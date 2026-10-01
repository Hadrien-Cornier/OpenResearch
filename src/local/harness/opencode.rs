//! OpenCode harness.
//!
//! Chat: talks to a lazily spawned `opencode serve` child (the `AgentHost` the
//! up server shares). serve is opencode's first-party embedding surface; HTTP
//! on loopback is just this adapter's transport, never exposed to the browser.
//! A turn = subscribe to the global `/event` SSE stream, POST the message
//! (which resolves when the turn ends), and translate this session's part
//! events into wire parts as they stream.
//!
//! Interactive prompts: unlike Claude (which ends its turn and resumes with a
//! new message), opencode approves *inline*. Its serve stream emits
//! `permission.asked` / `question.asked` while the `session.prompt` POST is
//! still open — the turn is paused, not finished. We surface those as
//! `permission` / `question` cards and reply over the live session
//! (`resume_from_prompt` → [`ResumeAction::Handled`]), which unblocks the same
//! POST. Auto-approve resolves native `ask` requests without a card; Default
//! surfaces them. Questions always need a human, so they always surface.
//!
//! Detection: opencode's `auth.json` is `{provider: {type}}`; the signed-in
//! providers are its account line, and `opencode models --verbose` is the model
//! list plus each model's reasoning `variants` (plain `opencode models` is the
//! fallback for a CLI too old for `--verbose`).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// A key probe is one short reply; a cold CLI start is most of it.
const KEY_PROBE_TIMEOUT: Duration = Duration::from_secs(15);

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::detect::{read_json, BinProbe, HarnessAuthState, HarnessInfo, ModelInfo};
use super::options::{
    HarnessOptions, OptionChoice, PermissionMode, PlanActivation, REASONING_DEFAULT_ID,
};
use super::{
    CompactCtx, CompactOutcome, Harness, OneShot, ResumeAction, TurnFailure, TurnOutcome,
    TurnResult, ORX_MAX_ATTEMPTS,
};
use crate::error::{anyhow, Result};
use crate::local::chat::{
    ContextUsage, DeliveryState, PromptAnswer, ResumeCtx, TurnCtx, WirePart, WirePrompt,
    WireQuestionOption, WireToolState,
};
use crate::local::local_models::is_loopback_url;
use crate::local::native_store::{self, NativeStore};
use crate::local::opencode::{find_opencode, AgentEndpoint, ResolvedBinary, SummarizeOutcome};

const OPENCODE_REINSTALL: &str =
    "Reinstall opencode (curl -fsSL https://opencode.ai/install | bash)";

#[path = "opencode_v2.rs"]
mod v2;

pub struct OpenCode;

impl OpenCode {
    /// `snapshot` skips every catalog probe — `models`/`debug config` children,
    /// the V2 server bring-up, local-server and dead-key checks — and answers
    /// install/auth only; readiness is deferred entirely and the entry is
    /// marked `catalog_pending` until a full pass replaces it. The one
    /// non-free check kept is the isolated DB lease preflight (file I/O, no
    /// child process), which surfaces `needs_config_repair` early.
    async fn detect_at(&self, snapshot: bool) -> Option<HarnessInfo> {
        let mut info = HarnessInfo::new(self.id(), self.name());
        let mut models = Vec::new();
        let mut public_models = HashSet::new();
        let mut config = Value::Null;
        let bin = find_opencode().ok();
        let mut resolved_binary = None;
        if let Some(discovered) = &bin {
            // The catalog children launch on the discovered binary while
            // `resolve_binary` probes every candidate — each spawn costs
            // seconds on Windows, and sequencing the catalog behind the
            // version sweep made the fill their sum. A resolution landing on
            // a different binary (a stale first candidate) re-runs the probes
            // on the winner; a V2 verdict never needs them. The snapshot
            // skips the spawns entirely — discovery already proves the file
            // is there, and a V2 install's real auth only answers through
            // the served API anyway.
            let speculated = (!snapshot).then(|| {
                super::detect::spawn_timed_probe(
                    "opencode",
                    "models",
                    opencode_models(discovered.clone()),
                )
            });
            let resolved = if snapshot {
                None
            } else {
                Some(
                    super::detect::timed_probe(
                        "opencode",
                        "resolve",
                        crate::local::opencode::resolve_binary(),
                    )
                    .await,
                )
            };
            let (bin, probe) = match resolved.as_ref() {
                Some(Ok(binary)) => (
                    binary.path.clone(),
                    BinProbe::Answered(Some(binary.version.clone())),
                ),
                // Nothing resolved: report what discovery first named, so the
                // UI shows a broken install rather than "not detected".
                Some(Err(error)) => (discovered.clone(), BinProbe::Broken(error.to_string())),
                None => (discovered.clone(), BinProbe::Unknown),
            };
            info.record_bin(&bin, probe);
            // The speculation only stands when resolution picked the binary
            // it ran against — anything else cuts the child loose
            // (kill_on_drop reaps it) and the winner is probed below.
            let mut spec_out = None;
            if let Some(models_task) = speculated {
                match resolved.as_ref() {
                    Some(Ok(binary))
                        if binary.protocol != crate::local::opencode::Protocol::V2
                            && binary.path == *discovered =>
                    {
                        spec_out = Some(models_task.await);
                    }
                    other => {
                        if std::env::var_os("ORX_DETECT_TIMING").is_some() {
                            eprintln!(
                                "orx detect: opencode spec miss (resolved={:?} spec={:?})",
                                other.map(|r| r.as_ref().map(|b| b.path.clone())),
                                discovered
                            );
                        }
                        models_task.abort();
                        // Await teardown so the aborted probe's timing row
                        // lands in the fill's sink before the pass drains it.
                        let _ = models_task.await;
                    }
                }
            }
            if let Some(Ok(binary)) = resolved {
                if binary.protocol == crate::local::opencode::Protocol::V2 {
                    return Some(v2::detect(binary, info).await);
                }
                resolved_binary = Some(binary);
            }
            if !info.install_broken {
                let db = native_store::opencode_db(NativeStore::Isolated);
                let preflight = tokio::task::spawn_blocking(move || {
                    native_store::opencode_database::DatabaseLease::acquire(&db, 1).map(drop)
                })
                .await
                .unwrap_or_else(|error| {
                    Err(anyhow!("OpenCode database inspection failed: {error}"))
                });
                if let Err(error) = preflight {
                    let busy = error
                        .downcast_ref::<native_store::opencode_database::DatabaseBusy>()
                        .is_some();
                    info.auth_state = if busy {
                        HarnessAuthState::Unknown
                    } else {
                        HarnessAuthState::Unsupported
                    };
                    // A database the CLI will not open is not an out-of-date
                    // binary: upgrading a current install cannot repair it, and
                    // orx must never delete a user's database to clear it.
                    info.needs_config_repair = !busy;
                    info.agent_note = Some(if busy {
                        error.to_string()
                    } else {
                        format!(
                            "{error}\nThis is an OpenCode database problem, not an out-of-date install. \
                             Close other OpenCode processes and re-check; if it persists, move the file aside so OpenCode can recreate it."
                        )
                    });
                    return Some(info);
                }
            }
            // A binary that failed `--version` has no catalog to give either.
            if let Some(binary) = resolved_binary.as_ref().filter(|_| !snapshot) {
                (models, public_models) = match spec_out {
                    Some(catalog) => catalog.ok().unwrap_or_default(),
                    // Resolution picked a different binary than the
                    // speculation ran on — rare enough to simply re-probe.
                    None => opencode_models(binary.path.clone()).await,
                };
                // `debug config --pure` answers the same fields this pass
                // consumes (provider gates, local providers, the default
                // model) — project/plugin layers are off under `--pure` and
                // the child runs from the home dir anyway — so the file read
                // stands in for a multi-second spawn. It falls back to the
                // child only for a config the strict parse cannot honor —
                // `opencode.jsonc` or a `.json` carrying comments — which
                // opencode accepts and a `Null` here would silently drop
                // (configured local providers would vanish).
                let unresolved;
                (config, unresolved) = snapshot_config();
                if unresolved {
                    if let Some(text) = super::detect::timed_probe(
                        "opencode",
                        "config",
                        run_models(binary.path.clone(), &["debug", "config", "--pure"]),
                    )
                    .await
                    .and_then(|text| serde_json::from_str(&text).ok())
                    {
                        config = text;
                    }
                }
            } else if snapshot {
                // `debug config` costs a child process; the snapshot reads
                // the config file directly.
                config = snapshot_config().0;
            }
        }
        apply_configured_labels(&mut models, &config);
        let providers: Vec<_> = opencode_providers()
            .into_iter()
            .filter(|id| provider_enabled(&config, id))
            .collect();
        if !providers.is_empty() {
            info.authenticated = true;
            info.auth_method = Some("oauth");
            info.account = Some(providers.join(", "));
        }
        // opencode also takes provider keys straight from the environment,
        // writing no auth.json — same fallback claude.rs has. Checked against
        // orx's synced env too, since that's a source the harness child gets
        // but this process may not. Measured, not assumed: `opencode models`
        // still lists free/bundled models when signed out, so a non-empty
        // model list can't stand in for a credential.
        const PROVIDER_KEYS: &[(&str, &str)] = &[
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("google", "GEMINI_API_KEY"),
            ("google", "GOOGLE_API_KEY"),
            ("groq", "GROQ_API_KEY"),
            ("xai", "XAI_API_KEY"),
            ("deepseek", "DEEPSEEK_API_KEY"),
        ];
        if !info.authenticated
            && PROVIDER_KEYS.iter().any(|(id, key)| {
                provider_enabled(&config, id) && super::detect::api_key(key).is_some()
            })
        {
            info.authenticated = true;
            info.auth_method = Some("apiKey");
        }

        let local = local_providers(&config);
        // Snapshot defers the local-server liveness probes with the catalog.
        let available = if snapshot {
            HashSet::new()
        } else {
            available_local_models(&local).await
        };
        let is_local =
            |model: &ModelInfo| local.iter().any(|(id, _)| model_provider(&model.id) == *id);
        let missing_local = models
            .iter()
            .any(|model| is_local(model) && !available.contains(&model.id));
        if missing_local {
            info.agent_note = Some("Some local models are unavailable. Start the server, load the configured model, and re-check OpenCode.".to_string());
        }
        models.retain(|model| {
            available.contains(&model.id)
                || (info.authenticated && !is_local(model))
                || (public_models.contains(&model.id)
                    && provider_enabled(&config, "opencode")
                    && !is_local(model))
        });
        // Onboarding and the composer seed their selection from the first model.
        let default = config.get("model").and_then(Value::as_str);
        models.sort_by_key(|model| {
            (
                Some(model.id.as_str()) != default,
                !model.id.starts_with("orx-local-"),
            )
        });
        if !info.authenticated && !local.is_empty() {
            info.auth_method = Some("local");
        }
        if !snapshot {
            // Readiness needs the catalog — `models` proves a credential
            // resolves to something runnable. The snapshot leaves it false;
            // `detect_one` marks the answer pending until the fill lands.
            info.agent_ready = info.installed && !info.install_broken && !models.is_empty();
        }
        if info.agent_ready {
            // Hide the models of providers whose stored key a live request rejects.
            let cloud_providers: Vec<_> = providers
                .iter()
                .filter(|id| !local.iter().any(|(local_id, _)| local_id == id))
                .cloned()
                .collect();
            let dead = match &resolved_binary {
                Some(binary) => dead_providers(binary, &cloud_providers, &models).await,
                None => Vec::new(),
            };
            if !dead.is_empty() {
                models.retain(|model| !dead.iter().any(|p| model_provider(&model.id) == p));
                info.account = Some(
                    providers
                        .iter()
                        .map(|p| {
                            if dead.contains(p) {
                                format!("{p} (key rejected)")
                            } else {
                                p.clone()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", "),
                );
                let note = format!(
                    "{} rejected the stored API key, so its models are hidden. Re-add it with `opencode auth login`.",
                    dead.join(", ")
                );
                info.agent_note = Some(match info.agent_note.take() {
                    Some(local_note) => format!("{local_note} {note}"),
                    None => note,
                });
            }
            // Every key rejected and nothing free left: nothing can run a turn.
            if models.is_empty() {
                info.agent_ready = false;
                info.auth_state = HarnessAuthState::NeedsLogin;
            }
            info.models = models;
        } else if info.install_broken {
            info.agent_note = Some(info.broken_note(OPENCODE_REINSTALL));
        } else if !info.installed {
            info.agent_note = Some(
                "Install opencode (curl -fsSL https://opencode.ai/install | bash), then configure a local model or sign in with `opencode auth login`."
                    .to_string(),
            );
        } else if !snapshot {
            // Installed but not ready — the snapshot leaves the diagnosis to
            // the fill, whose catalog decides which of these actually applies.
            if !local.is_empty() {
                info.agent_note = Some(if available.is_empty() {
                    "Local model server unavailable or configured model not found. Start the server, load your model, and re-check OpenCode."
                } else {
                    "The local server is reachable, but OpenCode did not list the configured model. Check `opencode models` and re-check OpenCode."
                }.to_string());
            } else if info.authenticated {
                info.agent_note = Some(
                    "OpenCode listed no models. Check `opencode models` and re-check OpenCode."
                        .to_string(),
                );
            } else {
                info.agent_note = Some(
                    "Configure a local model in OpenCode, or sign in with `opencode auth login`."
                        .to_string(),
                );
            }
        }
        if info.installed
            && !info.install_broken
            && !info.agent_ready
            && !snapshot
            && config.is_null()
        {
            info.agent_note = Some("Could not read OpenCode configuration. Update OpenCode and re-check to discover local models.".to_string());
        }
        if info.auth_state == HarnessAuthState::Unknown && !config.is_null() {
            info.auth_state = if info.authenticated || info.agent_ready {
                HarnessAuthState::Ready
            } else {
                HarnessAuthState::NeedsLogin
            };
        }
        if let Err(error) = crate::local::local_models::read() {
            info.agent_note = Some(error.to_string());
        }
        Some(info)
    }
}

#[async_trait]
impl Harness for OpenCode {
    fn id(&self) -> &'static str {
        "opencode"
    }

    fn name(&self) -> &'static str {
        "OpenCode"
    }

    fn supports_chat(&self) -> bool {
        true
    }

    /// opencode compacts through its own summarize endpoint, which needs a live
    /// server and a native session; without either, the shared fallback runs.
    async fn compact(&self, ctx: &CompactCtx) -> Result<CompactOutcome> {
        let Some(native_id) = ctx.native_session_id.as_deref() else {
            return Ok(CompactOutcome::Fallback);
        };
        // A live session is still resumable, so a failure to compact it in
        // place is reported rather than traded for a summary of its transcript.
        match ctx
            .host
            .opencode
            .summarize(&ctx.session_id, native_id, ctx.model.as_deref())
            .await?
        {
            SummarizeOutcome::Compacted => Ok(CompactOutcome::Native),
            SummarizeOutcome::NoServer => Err(anyhow!(
                "OpenCode is not running for this chat — send a message first, then compact"
            )),
            SummarizeOutcome::NoModel => Ok(CompactOutcome::Fallback),
        }
    }

    async fn one_shot(&self, request: OneShot<'_>) -> Option<String> {
        opencode_one_shot(
            &crate::local::opencode::resolve_binary().await.ok()?,
            request,
        )
        .await
    }

    async fn detect(&self) -> Option<HarnessInfo> {
        self.detect_at(false).await
    }

    async fn detect_snapshot(&self) -> Option<HarnessInfo> {
        self.detect_at(true).await
    }

    async fn run_turn(&self, ctx: &mut TurnCtx) -> TurnResult {
        run_turn(ctx)
            .await
            .map(|()| TurnOutcome::Completed)
            .map_err(|error| TurnFailure::adapter(error, ctx.delivery_state()))
    }

    fn options(&self) -> HarnessOptions {
        // OpenCode's agent (plan/build) is independent of permission handling.
        // Default honors configured allow/ask/deny rules; Auto-approve answers
        // only native `ask` requests and never overrides explicit denies.
        // Reasoning IS a model property in opencode, so there is no meaningful
        // harness-wide list: the real choices are each model's `variants`, read
        // from `opencode models --verbose` in `detect` and attached per-model.
        // Leaving this axis empty means a model with no variants shows no
        // picker at all, rather than falling back to a bogus union.
        HarnessOptions::none().with_permission_choices(
            vec![
                OptionChoice::described(
                    "default",
                    "Default",
                    "Ask before actions that need your approval",
                ),
                OptionChoice::described(
                    "auto-approve",
                    "Auto-approve",
                    "Approve requests automatically, except actions you have denied",
                ),
            ],
            "default",
            PlanActivation::Command,
        )
    }

    /// opencode is paused mid-turn on a `permission.asked` / `question.asked`;
    /// the answer is replied over the live serve session, which unblocks the
    /// still-open `session.prompt` POST. So this delivers the reply inline and
    /// returns [`ResumeAction::Handled`] — never the new-message path.
    async fn resume_from_prompt(
        &self,
        ctx: &ResumeCtx,
        prompt: &WirePrompt,
        answer: &PromptAnswer,
    ) -> Result<ResumeAction> {
        let plan_mode = plan_exit_transition(prompt, answer);
        if let Some(plan_mode) = plan_mode {
            // Persist the native answer's Plan transition before OpenCode
            // consumes it. If delivery fails, restore the active Plan state so
            // the still-actionable card and ORX continue to agree.
            ctx.host.set_plan_mode(&ctx.session_id, plan_mode).await?;
            if let Err(err) = reply_inline(ctx, prompt, answer).await {
                let _ = ctx.host.set_plan_mode(&ctx.session_id, true).await;
                return Err(err);
            }
            return Ok(ResumeAction::Handled { plan_mode: None });
        }
        reply_inline(ctx, prompt, answer).await?;
        Ok(ResumeAction::Handled { plan_mode: None })
    }

    fn config_home(&self) -> Option<PathBuf> {
        // OpenCode discovers skills under XDG config, staying XDG even on macOS.
        Some(super::xdg_config_home().join("opencode"))
    }

    fn skill_target(&self) -> Option<PathBuf> {
        Some(
            self.config_home()?
                .join("skills")
                .join("orx")
                .join("SKILL.md"),
        )
    }

    fn skill_shim(&self) -> Option<&'static str> {
        // OpenCode reads the same SKILL.md format as Claude Code.
        Some(super::CLAUDE_SKILL)
    }

    fn session_skills_dir(&self) -> Option<&'static str> {
        Some(".opencode/skills")
    }
}

fn opencode_auth_path() -> Option<PathBuf> {
    let base = crate::local::shell_env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("share")))?;
    Some(base.join("opencode").join("auth.json"))
}

/// The cheap config read for the snapshot pass: `OPENCODE_CONFIG_CONTENT`,
/// the file `OPENCODE_CONFIG` points at, else `opencode.json` under
/// `OPENCODE_CONFIG_DIR` or the stock `~/.config/opencode`. `debug config
/// --pure` additionally merges project and plugin layers, but it costs a
/// child process — the snapshot settles for the file and the full pass
/// re-reads it properly.
///
/// One merge the file read must not skip: orx's own connected local models,
/// which `local_models::prepare_env` folds into `OPENCODE_CONFIG_CONTENT` for
/// every spawned CLI — and inline content shadows the file entirely, so a
/// connections-only user sees exactly those providers and nothing else.
///
/// Returns the config plus whether a source exists the strict parse could
/// not honor (`.jsonc`, comments) — a cue for the full pass to ask the CLI.
fn snapshot_config() -> (Value, bool) {
    let connections = crate::local::local_models::read().unwrap_or_default();
    // `true` when a config source exists that the strict file read cannot
    // honor — opencode accepts `.jsonc` and comments, which parse as `Null`
    // here and would silently drop configured providers. The full pass
    // resolves those with `debug config --pure`.
    let mut unresolved = false;
    let mut config = if let Some(content) = crate::local::shell_env::var("OPENCODE_CONFIG_CONTENT")
    {
        // The env var overrides the file entirely — present-but-unparseable
        // means "no usable config", not "fall through to the file" — but its
        // comments still outrun the strict parse.
        let parsed = serde_json::from_str::<Value>(&content.to_string_lossy());
        unresolved = parsed.is_err();
        parsed.unwrap_or(Value::Null)
    } else if !connections.is_empty() {
        // The spawned CLI would get inline content built from the connections
        // alone; the file never enters the picture.
        json!({})
    } else if let Some(custom) = crate::local::shell_env::var("OPENCODE_CONFIG") {
        let path = PathBuf::from(custom);
        let parsed = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok());
        unresolved = path.exists() && parsed.is_none();
        parsed.unwrap_or(Value::Null)
    } else {
        let dir = crate::local::shell_env::var("OPENCODE_CONFIG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| super::xdg_config_home().join("opencode"));
        let json = dir.join("opencode.json");
        let parsed = std::fs::read_to_string(&json)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok());
        // opencode honors a sibling `.jsonc` alongside — and comments inside
        // — `.json`; either outruns what the file read can claim to cover.
        unresolved = dir.join("opencode.jsonc").exists() || (json.exists() && parsed.is_none());
        parsed.unwrap_or(Value::Null)
    };
    if !connections.is_empty() && config.is_object() {
        let _ = crate::local::local_models::merge_config(&mut config, &connections, None);
    }
    (config, unresolved)
}

/// Providers opencode is signed into (its auth.json is `{provider: {type}}`).
fn opencode_providers() -> Vec<String> {
    let Some(auth) = opencode_auth_path().and_then(read_json) else {
        return Vec::new();
    };
    match auth.as_object() {
        Some(map) => map.keys().cloned().collect(),
        None => Vec::new(),
    }
}

/// Providers whose stored key a live request rejects: one tiny read-only
/// request per provider on its first catalogued model. Verdicts are kept per
/// auth.json version so a paid probe runs once, not once per detection, and
/// only when the child actually answered — a timeout or spawn failure is not
/// remembered, so a dead key is still found on the next detection.
async fn dead_providers(
    binary: &ResolvedBinary,
    providers: &[String],
    models: &[ModelInfo],
) -> Vec<String> {
    static VERDICTS: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let verdicts = VERDICTS.get_or_init(Default::default);
    let version = opencode_auth_path()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let probes = providers.iter().filter_map(|provider| {
        let model = models
            .iter()
            .find(|model| model_provider(&model.id) == provider)?
            .id
            .clone();
        let key = format!("{version}:{provider}");
        let cached = verdicts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .copied();
        Some(async move {
            let dead = match cached {
                Some(dead) => dead,
                None => {
                    let dead = probe_rejects_key(binary, &model).await;
                    if let Some(dead) = dead {
                        verdicts
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(key, dead);
                    }
                    dead.unwrap_or(false)
                }
            };
            dead.then(|| provider.clone())
        })
    });
    futures::future::join_all(probes)
        .await
        .into_iter()
        .flatten()
        .collect()
}

/// `Some(true)` when the provider answered with an authentication error,
/// `Some(false)` for any other answer, `None` when the child never answered.
async fn probe_rejects_key(binary: &ResolvedBinary, model: &str) -> Option<bool> {
    let out = opencode_child(
        binary,
        Some(model),
        "Reply with the single word ok",
        KEY_PROBE_TIMEOUT,
    )
    .await?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Some(!out.status.success() && is_auth_rejection(&text))
}

/// Phrases the major providers use when a key is invalid, revoked, or expired.
/// A bare `401` is deliberately absent: line numbers and counts contain it.
const AUTH_REJECTION_MARKERS: &[&str] = &[
    "api key not valid",
    "invalid api key",
    "incorrect api key",
    "invalid x-api-key",
    "invalid_api_key",
    "authentication_error",
    "authentication failed",
    "unauthorized",
    "status 401",
    "http 401",
    "code 401",
    "error 401",
    "(401)",
    "key has expired",
    "api key expired",
    "no auth credentials",
];

fn is_auth_rejection(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    AUTH_REJECTION_MARKERS.iter().any(|m| lower.contains(m))
}

/// The `provider` half of an opencode `provider/model` id.
fn model_provider(id: &str) -> &str {
    id.split_once('/').map(|(p, _)| p).unwrap_or(id)
}

fn local_providers(config: &Value) -> Vec<(&str, &Value)> {
    config
        .get("provider")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter(|(id, provider)| {
            provider_enabled(config, id)
                && provider
                    .pointer("/options/baseURL")
                    .and_then(Value::as_str)
                    .is_some_and(is_loopback_url)
        })
        .map(|(id, provider)| (id.as_str(), provider))
        .collect()
}

fn provider_enabled(config: &Value, id: &str) -> bool {
    let contains = |key| {
        config.get(key).and_then(Value::as_array).map(|providers| {
            providers
                .iter()
                .any(|provider| provider.as_str() == Some(id))
        })
    };
    contains("enabled_providers").unwrap_or(true)
        && !contains("disabled_providers").unwrap_or(false)
}

fn apply_configured_labels(models: &mut [ModelInfo], config: &Value) {
    for model in models
        .iter_mut()
        .filter(|model| model.display_name.is_none())
    {
        if let Some((provider, id)) = model.id.split_once('/') {
            model.display_name = config
                .get("provider")
                .and_then(|providers| providers.get(provider))
                .and_then(|provider| provider.get("models"))
                .and_then(|models| models.get(id))
                .and_then(|model| model.get("name"))
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
    }
}

async fn available_local_models(providers: &[(&str, &Value)]) -> HashSet<String> {
    let probes = providers.iter().map(|(id, provider)| async move {
        let advertised = crate::local::local_models::discover(&crate::local::local_models::Probe {
            base_url: provider.pointer("/options/baseURL")?.as_str()?.to_owned(),
            api_key: provider
                .pointer("/options/apiKey")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        })
        .await
        .ok()?;
        let models = provider.get("models")?.as_object()?;
        Some(
            models
                .iter()
                .filter_map(|(model, options)| {
                    let api_id = options.get("id").and_then(Value::as_str).unwrap_or(model);
                    advertised
                        .iter()
                        .any(|entry| entry == api_id)
                        .then(|| format!("{id}/{model}"))
                })
                .collect::<Vec<_>>(),
        )
    });
    futures::future::join_all(probes)
        .await
        .into_iter()
        .flatten()
        .flatten()
        .collect()
}

/// `opencode models --verbose` — the ground truth for what the agent can run
/// *and* for each model's reasoning `variants`.
///
/// `--verbose` prints, per model, a `provider/model` header line followed by a
/// pretty-printed JSON object. We parse it for the `variants` map because
/// reasoning in opencode is a genuine per-model property (issue #123):
/// `gemini-3-flash` offers `minimal…high`, `deepseek-v4-flash` offers
/// `low…max`, and plenty of models offer none at all.
///
/// Falls back to the plain `opencode models` id list if `--verbose` is
/// unavailable or unparseable, so an older/newer opencode still yields models
/// (just without per-model variants).
async fn opencode_models(bin: PathBuf) -> (Vec<super::ModelInfo>, HashSet<String>) {
    let verbose = run_models(bin.clone(), &["models", "--verbose"]).await;
    if let Some(out) = &verbose {
        let parsed = parse_verbose_models(out);
        if !parsed.is_empty() {
            let public = parse_verbose_models_filtered(out, true)
                .into_iter()
                .map(|model| model.id)
                .collect();
            return (parsed, public);
        }
    }
    let Some(plain) = run_models(bin, &["models"]).await else {
        return (Vec::new(), HashSet::new());
    };
    (
        model_id_lines(&plain).map(super::ModelInfo::new).collect(),
        HashSet::new(),
    )
}

/// One headless request on a throwaway `opencode run` child on
/// `request.model`, else the user's default model. opencode's server no
/// longer retitles parent sessions itself (only sub-agent child sessions get
/// task-description titles), so titles run through here like the
/// claude/codex one-shot children. opencode has no system-prompt flag, so
/// `system` leads the message. Any failure lands on `None` and the caller
/// keeps its fallback.
async fn opencode_one_shot(binary: &ResolvedBinary, request: OneShot<'_>) -> Option<String> {
    let message = format!("{}\n\n{}", request.system, request.prompt);
    if binary.protocol == crate::local::opencode::Protocol::V2 {
        return v2::generate(
            binary.clone(),
            request.model.map(str::to_owned),
            message,
            request.timeout,
        )
        .await
        .ok();
    }
    let out = opencode_child(binary, request.model, &message, request.timeout).await?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Run one unattended `opencode run` to completion, or `None` if it could
/// not be started, had no isolated store, or ran past `timeout`.
///
/// The message embeds untrusted text, so the child must not be able to act on
/// it: the built-in read-only `plan` agent denies writes, `--pure` skips
/// external plugins, and the temp cwd keeps any residual reads away from real
/// repos. A tool call that still asks for permission just blocks the child
/// until the timeout kills it.
async fn opencode_child(
    binary: &ResolvedBinary,
    model: Option<&str>,
    message: &str,
    timeout: Duration,
) -> Option<std::process::Output> {
    let db = native_store::prepare_opencode(NativeStore::Isolated).ok()?;
    let lease = crate::local::opencode::prepare_database(binary, &db)
        .await
        .ok()?;
    let mut cmd = tokio::process::Command::new(&binary.path);
    cmd.args(["run", "--agent", "plan", "--pure"]);
    cmd.args(model.iter().flat_map(|model| ["--model", model]))
        .arg(message)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .current_dir(std::env::temp_dir());
    crate::local::local_models::prepare_env(&mut cmd, model).ok()?;
    // Stateless V1 calls must not race chat startup when initializing its database.
    cmd.env("OPENCODE_DB", ":memory:");
    cmd.env("OPENCODE_DISABLE_PROJECT_CONFIG", "1")
        .env("OPENCODE_CONFIG_PROJECT_DISABLE", "1")
        .env("OPENCODE_DISABLE_AUTOUPDATE", "1");
    // Plain text only — an ANSI-colorizing CLI (or a synced FORCE_COLOR) would
    // otherwise write escape codes straight into the reply.
    cmd.env("NO_COLOR", "1");
    let binary = binary.clone();
    tokio::spawn(async move {
        use tokio::io::AsyncReadExt;
        // The task owns the lease until the child is reaped, even if its caller is cancelled.
        let _lease = lease;
        binary.check_unchanged().ok()?;
        let (mut child, _permit) = super::detect::detect_spawn_child(cmd).await.ok()?;
        let mut stdout = child.stdout.take()?;
        let mut stderr = child.stderr.take()?;
        let output = tokio::spawn(async move {
            let mut bytes = Vec::new();
            stdout.read_to_end(&mut bytes).await.ok()?;
            Some(bytes)
        });
        let errors = tokio::spawn(async move {
            let mut bytes = Vec::new();
            stderr.read_to_end(&mut bytes).await.ok()?;
            Some(bytes)
        });
        let status = match tokio::time::timeout(timeout, child.wait()).await {
            Ok(Ok(status)) => Some(status),
            _ => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                None
            }
        };
        let stdout = output.await.ok().flatten();
        let stderr = errors.await.ok().flatten();
        Some(std::process::Output {
            status: status?,
            stdout: stdout?,
            stderr: stderr?,
        })
    })
    .await
    .ok()
    .flatten()
}

/// Run `opencode <args>` in the home dir, returning stdout on success.
/// File redirection avoids the truncated piped config output reported in #307.
/// Takes a bare path rather than `ResolvedBinary` so detection can spawn the
/// catalog children before resolution settles — a binary swapped mid-detect
/// just answers as whatever it now is, and the next pass re-verifies.
async fn run_models(bin: PathBuf, args: &[&str]) -> Option<String> {
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(args)
        .current_dir(dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")))
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    crate::local::local_models::prepare_env(&mut cmd, None).ok()?;
    cmd.env("OPENCODE_DB", ":memory:")
        .env("OPENCODE_DISABLE_AUTOUPDATE", "1")
        // The probe runs from the home dir, so project-layer discovery is
        // already empty — skipping the scan outright is ~1s off the child's
        // startup (measured on Windows).
        .env("OPENCODE_DISABLE_PROJECT_CONFIG", "1")
        .env("OPENCODE_CONFIG_PROJECT_DISABLE", "1")
        // Serve the catalog bundled into the binary instead of refreshing
        // models.dev — ~1.2s off a cold first-install run, and opencode
        // refreshes its own cache on the next real launch anyway.
        .env("OPENCODE_DISABLE_MODELS_FETCH", "1")
        // Nothing the catalog probe prints depends on plugins, LSP servers,
        // the claude-code bridge, or terminal chrome — each disabled piece is
        // startup work the child skips (~0.8s combined on Windows).
        .env("OPENCODE_DISABLE_DEFAULT_PLUGINS", "1")
        .env("OPENCODE_DISABLE_LSP_DOWNLOAD", "1")
        .env("OPENCODE_DISABLE_CLAUDE_CODE", "1")
        .env("OPENCODE_DISABLE_PRUNE", "1")
        .env("OPENCODE_DISABLE_AUTOCOMPACT", "1")
        .env("OPENCODE_DISABLE_TERMINAL_TITLE", "1")
        .env("NO_COLOR", "1");
    let path = std::env::temp_dir().join(format!("orx-opencode-stdout-{}", uuid::Uuid::new_v4()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    cmd.stdout(std::process::Stdio::from(options.open(&path).ok()?));
    let status = {
        // The lane is acquired before the deadline, but the spawn itself is
        // inside it — a `CreateProcess` wedged on an AV scan must not hold the
        // fill (and its single-flight flag) forever. On timeout the dropped
        // future kills the child via `kill_on_drop`.
        let permit = super::detect::detect_spawn_permit().await;
        tokio::time::timeout(Duration::from_secs(20), async move {
            let (mut child, _permit) = super::detect::spawn_with_permit(cmd, permit).await.ok()?;
            child.wait().await.ok()
        })
        .await
        .ok()
        .flatten()
    };
    let stdout = std::fs::read(&path).ok();
    std::fs::remove_file(&path).ok();
    let stdout = stdout?;
    matches!(status, Some(status) if status.success())
        .then(|| String::from_utf8_lossy(&stdout).into_owned())
}

/// The bare `provider/model` id lines of plain `opencode models` output.
fn model_id_lines(out: &str) -> impl Iterator<Item = &str> {
    out.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && l.contains('/'))
}

/// Parse `opencode models --verbose` into models + their variant ids.
///
/// The format is a repeating `header line` + `{ … }` JSON block. We walk lines,
/// treat any non-`{`-starting line containing `/` as a header, and accumulate
/// the following block until braces balance — brace counting (rather than
/// "next header") keeps a `}` inside a nested object from ending the block
/// early.
///
/// The counter skips braces inside JSON string literals. That is not
/// hypothetical tidiness: a single `{` in any free-text field (a model `name`
/// or description) would otherwise desynchronize the depth, and since it can
/// never balance again the loop would swallow the entire rest of the output —
/// dropping every later model, and quietly, because a partial parse doesn't
/// trigger the plain-list fallback.
fn parse_verbose_models(out: &str) -> Vec<super::ModelInfo> {
    parse_verbose_models_filtered(out, false)
}

fn parse_verbose_models_filtered(out: &str, public_only: bool) -> Vec<super::ModelInfo> {
    let mut models = Vec::new();
    let mut lines = out.lines().peekable();
    while let Some(line) = lines.next() {
        let header = line.trim();
        if header.is_empty() || !header.contains('/') || header.starts_with('{') {
            continue;
        }
        if !lines
            .peek()
            .is_some_and(|l| l.trim_start().starts_with('{'))
        {
            continue;
        }
        let mut block = String::new();
        let mut depth = 0usize;
        let mut in_str = false;
        let mut esc = false;
        for body in lines.by_ref() {
            for ch in body.chars() {
                match ch {
                    _ if esc => esc = false,
                    '\\' if in_str => esc = true,
                    '"' => in_str = !in_str,
                    '{' if !in_str => depth += 1,
                    '}' if !in_str => depth = depth.saturating_sub(1),
                    _ => {}
                }
            }
            // Neither a string literal nor an escape spans lines in this
            // output, so reset both: an unterminated quote would otherwise
            // invert `in_str` for every following line, stop brace counting
            // entirely, and swallow the rest of the output — the same silent
            // model-dropping failure the string tracking exists to prevent.
            esc = false;
            in_str = false;
            block.push_str(body);
            block.push('\n');
            if depth == 0 {
                break;
            }
        }
        // An unparseable block still yields the model, just without variants —
        // never drop a model the CLI reported.
        let parsed = serde_json::from_str::<Value>(&block).ok();
        if public_only
            && !(header.starts_with("opencode/")
                && parsed
                    .as_ref()
                    .and_then(|v| v.pointer("/cost/input"))
                    .and_then(Value::as_f64)
                    == Some(0.0)
                && parsed
                    .as_ref()
                    .and_then(|v| v.pointer("/cost/output"))
                    .and_then(Value::as_f64)
                    == Some(0.0))
        {
            continue;
        }
        let variants = parsed.as_ref().and_then(variant_ids);
        let name = parsed
            .as_ref()
            .and_then(|v| v.get("name"))
            .and_then(Value::as_str);
        let model = match variants {
            Some(ids) => {
                let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
                super::ModelInfo::new(header).with_reasoning(&refs)
            }
            None => super::ModelInfo::new(header),
        };
        models.push(model.with_label(name, None));
    }
    models
}

/// The variant ids of one model's verbose JSON, ordered weakest → strongest.
///
/// `Some(vec![])` (an empty `variants` map) is distinct from `None` (no
/// `variants` key at all): the former hides the picker, the latter falls back.
///
/// Ordering is imposed here rather than taken from the JSON: `serde_json`'s
/// default `Map` is a `BTreeMap`, so object keys arrive alphabetically
/// (`high, low, max, medium, xhigh`) and a picker in that order is nonsense.
/// Sorting by `OPENCODE_VARIANTS` restores the intended ramp.
fn variant_ids(model: &Value) -> Option<Vec<String>> {
    let variants = model.get("variants")?;
    let mut ids: Vec<String> = if let Some(map) = variants.as_object() {
        map.keys().cloned().collect()
    } else {
        // Tolerate an array form (`[]` is what an empty map serializes to in
        // some opencode builds — observed locally).
        variants
            .as_array()?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect()
    };
    // Known ids ramp in canonical order; anything unrecognized sorts after
    // them, alphabetically, so a new opencode variant still shows up.
    ids.sort_by_key(|id| {
        let rank = OPENCODE_VARIANTS
            .iter()
            .position(|v| v == id)
            .unwrap_or(OPENCODE_VARIANTS.len());
        (rank, id.clone())
    });
    Some(ids)
}

/// The variant ids opencode's catalog is known to use, weakest → strongest.
/// This ORDERS a model's variants for display (see `variant_ids`); it is not an
/// allowlist — opencode's catalog is the authority on what exists.
const OPENCODE_VARIANTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// Session reasoning id → opencode's top-level `variant` value.
///
/// Only the `default` sentinel (and an absent level) send nothing; every other
/// value is forwarded as-is. Deliberately NOT filtered against
/// `OPENCODE_VARIANTS`: the ids come from opencode's own catalog, and
/// `variant_ids` goes out of its way to keep ones this build doesn't recognize
/// so a new variant still reaches the picker. Filtering here would offer such a
/// choice and then silently ignore it. `run_turn` has only the model id and
/// must not re-shell `opencode models` (a 20s subprocess) per turn, so opencode
/// itself is the validator of last resort.
fn opencode_variant(level: Option<&str>) -> Option<&str> {
    level.filter(|l| *l != REASONING_DEFAULT_ID)
}

/// opencode part → wire part (the shapes are already close).
fn to_wire_part(part: &Value) -> Option<WirePart> {
    let id = part.get("id")?.as_str()?.to_string();
    let kind = part.get("type")?.as_str()?;
    match kind {
        "text" | "reasoning" => Some(WirePart {
            id,
            kind: kind.into(),
            text: part.get("text").and_then(Value::as_str).map(str::to_string),
            tool: None,
            state: None,
            prompt: None,
            phase: None,
            children: Vec::new(),
        }),
        "tool" => {
            let state = part.get("state");
            Some(WirePart {
                id,
                kind: "tool".into(),
                text: None,
                tool: part.get("tool").and_then(Value::as_str).map(str::to_string),
                state: Some(WireToolState {
                    status: state
                        .and_then(|s| s.get("status"))
                        .and_then(Value::as_str)
                        .unwrap_or("running")
                        .into(),
                    input: state.and_then(|s| s.get("input")).cloned(),
                    output: state
                        .and_then(|s| s.get("output"))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    error: state
                        .and_then(|s| s.get("error"))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    title: state
                        .and_then(|s| s.get("title"))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                }),
                prompt: None,
                phase: None,
                children: Vec::new(),
            })
        }
        _ => None,
    }
}

/// The id of the most-recent top-level `task` tool part not yet linked to a
/// child session — the row a freshly-spawned sub-agent session belongs to.
/// opencode's `session.created` carries the child's `parentID` (our session) but
/// not the spawning tool call, so we attribute to the latest unclaimed `task`
/// row; in the common single-task case this is exact.
///
/// Only top-level `task` rows are candidates, so nesting is one level deep: a
/// sub-agent that spawns its *own* sub-agent emits a `session.created` whose
/// `parentID` is the child session (not ours), so the grandchild isn't
/// registered and its events fall through to the foreign-session drop.
fn newest_task_part_id(parts: &[WirePart], claimed: &HashMap<String, String>) -> Option<String> {
    let taken: HashSet<&str> = claimed.values().map(String::as_str).collect();
    parts
        .iter()
        .rev()
        .find(|p| p.tool.as_deref() == Some("task") && !taken.contains(p.id.as_str()))
        .map(|p| p.id.clone())
}

/// opencode `permission.asked` payload → a `permission` card. The permission
/// request id rides on `native_id` so the reply can address
/// `POST /session/{sid}/permissions/{id}`. `permission` is opencode's tool
/// group (e.g. `bash`, `edit`); the metadata carries the concrete call detail.
fn permission_card(props: &Value) -> Option<WirePrompt> {
    let id = props.get("id").and_then(Value::as_str)?.to_string();
    Some(WirePrompt {
        kind: "permission".into(),
        tool: props
            .get("permission")
            .and_then(Value::as_str)
            .map(str::to_string),
        // The event's `metadata` is the closest thing to a tool input summary
        // the UI can render (command / file / etc., shape varies by tool).
        tool_input: props.get("metadata").filter(|m| !m.is_null()).cloned(),
        native_id: Some(id),
        ..Default::default()
    })
}

/// opencode `question.asked` payload → a `question` card. opencode's
/// `QuestionInfo` (`{question, header, options:[{label,description}], multiple}`)
/// is the same shape as Claude's AskUserQuestion, so it maps 1:1. Only the first
/// question is surfaced (the composer answers one at a time); its request id
/// rides on `native_id` for `POST /question/{id}/reply`.
fn question_card(props: &Value, plan_exit_calls: &HashSet<String>) -> Option<WirePrompt> {
    let id = props.get("id").and_then(Value::as_str)?.to_string();
    let q = props
        .get("questions")
        .and_then(Value::as_array)
        .and_then(|qs| qs.first())?;
    let options = q
        .get("options")
        .and_then(Value::as_array)
        .map(|opts| {
            opts.iter()
                .filter_map(|o| {
                    Some(WireQuestionOption {
                        label: o.get("label").and_then(Value::as_str)?.to_string(),
                        description: o
                            .get("description")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Some(WirePrompt {
        kind: "question".into(),
        question: q
            .get("question")
            .and_then(Value::as_str)
            .map(str::to_string),
        header: q.get("header").and_then(Value::as_str).map(str::to_string),
        options,
        multi_select: q.get("multiple").and_then(Value::as_bool).unwrap_or(false),
        plan_exit: props
            .get("tool")
            .and_then(|tool| tool.get("callID"))
            .and_then(Value::as_str)
            .is_some_and(|call_id| plan_exit_calls.contains(call_id)),
        native_id: Some(id),
        ..Default::default()
    })
}

/// POST a permission decision to the live serve session (v1 API). `response` is
/// `once` | `always` | `reject`.
async fn post_permission(
    http: &reqwest::Client,
    base: &str,
    native_session: &str,
    permission_id: &str,
    response: &str,
) -> Result<()> {
    http.post(format!(
        "{base}/session/{native_session}/permissions/{permission_id}"
    ))
    .json(&json!({ "response": response }))
    .send()
    .await?
    .error_for_status()?;
    Ok(())
}

fn submitted_answers<'a>(answers: &'a [String], note: Option<&'a String>) -> &'a [String] {
    if !answers.is_empty() {
        answers
    } else {
        note.filter(|note| !note.trim().is_empty())
            .map(std::slice::from_ref)
            .unwrap_or_default()
    }
}

/// Deliver an answered card's reply to the live serve session, unblocking the
/// paused `session.prompt` POST. Permission → `{response: once|always|reject}`;
/// question → `{answers: [[label,...]]}` (or reject). The reply target is the
/// card's `native_id` (the opencode permission/question request id).
async fn reply_inline(ctx: &ResumeCtx, prompt: &WirePrompt, answer: &PromptAnswer) -> Result<()> {
    let request_id = prompt
        .native_id
        .as_deref()
        .ok_or_else(|| anyhow!("opencode prompt has no reply id"))?;
    // The reply only lands if the turn is still paused waiting for it. If the
    // turn already ended (errored / interrupted), serve may still accept the
    // POST but no one is consuming the resumed stream, so the reply would be
    // lost and the card would falsely mark resolved. Reject it instead — the
    // card stays actionable and the user sees the turn is no longer live.
    if !ctx.is_busy().await {
        return Err(anyhow!(
            "this turn is no longer running — its prompt can't be answered"
        ));
    }
    // Reach this session's live serve child through the shared host, exactly
    // as `ChatHost::interrupt` does — the reply goes to the same loopback
    // serve whose `session.prompt` POST is paused on this prompt.
    let endpoint = ctx
        .host
        .opencode
        .endpoint_for(&ctx.session_id)
        .await
        .ok_or_else(|| anyhow!("OpenCode serve is not running"))?;
    if endpoint.protocol == crate::local::opencode::Protocol::V2 {
        return v2::reply(ctx, &endpoint, prompt, answer).await;
    }
    let base = endpoint.base_url;
    let http = &endpoint.client;

    match prompt.kind.as_str() {
        "permission" => {
            // approve → "always" (so the same tool won't re-prompt this turn);
            // reject closes it. The reply is session-scoped in opencode's v1 API.
            let native_session = ctx.native_session_id.as_deref().ok_or_else(|| {
                anyhow!("opencode session has no native id — cannot deliver the reply")
            })?;
            let response = if answer.approve { "always" } else { "reject" };
            post_permission(http, &base, native_session, request_id, response).await?;
        }
        "question" => {
            let submitted = submitted_answers(&answer.answers, answer.note.as_ref());
            if submitted.is_empty() {
                // No selection: reject the question rather than reply empty, so
                // opencode surfaces the model's fallback path.
                http.post(format!("{base}/question/{request_id}/reject"))
                    .json(&json!({}))
                    .send()
                    .await?
                    .error_for_status()?;
            } else {
                // opencode takes an array of answers, one per question; we only
                // surface the first question, so send a single answer array.
                http.post(format!("{base}/question/{request_id}/reply"))
                    .json(&json!({ "answers": [submitted] }))
                    .send()
                    .await?
                    .error_for_status()?;
            }
        }
        other => {
            return Err(anyhow!(
                "opencode cannot reply to a `{other}` prompt inline"
            ))
        }
    }
    Ok(())
}

/// Session mode → opencode built-in agent name. `Plan` runs the read-only
/// `plan` agent (denies edits, allows inspection); everything else runs the
/// default `build` agent. The permission-reply behavior (surface vs auto-reply)
/// is a separate axis handled in `handle_prompt_event`.
fn opencode_agent(plan_mode: bool) -> &'static str {
    if plan_mode {
        "plan"
    } else {
        "build"
    }
}

fn opencode_auto_approve(mode: Option<PermissionMode>) -> bool {
    matches!(mode, Some(PermissionMode::Auto))
}

fn plan_exit_transition(prompt: &WirePrompt, answer: &PromptAnswer) -> Option<bool> {
    prompt.plan_exit.then(|| {
        !answer
            .answers
            .iter()
            .any(|choice| choice.eq_ignore_ascii_case("yes"))
    })
}

#[derive(Debug)]
struct OpenCodeSetupHttpError {
    status: reqwest::StatusCode,
    retry_after: Option<Duration>,
}

impl std::fmt::Display for OpenCodeSetupHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpenCode setup returned HTTP {}", self.status)
    }
}

impl std::error::Error for OpenCodeSetupHttpError {}

fn opencode_setup_response(response: reqwest::Response) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let retry_after = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs);
    Err(OpenCodeSetupHttpError {
        status: response.status(),
        retry_after,
    }
    .into())
}

async fn ensure_runtime(
    ctx: &mut TurnCtx,
    store: NativeStore,
    binary: crate::local::opencode::ResolvedBinary,
    database: PathBuf,
) -> Result<crate::local::opencode::AgentStatus> {
    let host = ctx.host.clone();
    let project = ctx.project.clone();
    let session = ctx.session_id.clone();
    let model = ctx.model.clone();
    let (sender, mut progress) = tokio::sync::watch::channel("Preparing OpenCode".to_string());
    host.opencode.revive(&session);
    let setup = host.opencode.ensure(
        &project,
        &session,
        model.as_deref(),
        crate::local::opencode::ResolvedRuntime {
            binary,
            database,
            store,
        },
        sender,
    );
    tokio::pin!(setup);
    let mut progress_open = true;
    let result = loop {
        tokio::select! {
            result = &mut setup => break result,
            changed = progress.changed(), if progress_open => {
                if changed.is_err() { progress_open = false; continue; }
                let mut part = WirePart::tool("opencode-setup", "OpenCode", "running", None);
                if let Some(state) = &mut part.state { state.title = Some(progress.borrow().clone()); }
                ctx.upsert_part(part);
                ctx.flush()?;
            }
        }
    };
    if let Some(state) = ctx
        .assistant
        .parts
        .iter_mut()
        .find(|part| part.id == "opencode-setup")
        .and_then(|part| part.state.as_mut())
    {
        state.status = if result.is_ok() { "completed" } else { "error" }.into();
        state.error = result.as_ref().err().map(ToString::to_string);
    }
    ctx.flush()?;
    result
}

async fn opencode_setup_attempt(
    ctx: &mut TurnCtx,
    store: NativeStore,
    binary: &crate::local::opencode::ResolvedBinary,
    database: &Path,
) -> Result<(String, String, reqwest::Response)> {
    let status = ensure_runtime(ctx, store, binary.clone(), database.to_owned()).await?;
    let port = status
        .port
        .ok_or_else(|| anyhow!("opencode agent has no port"))?;
    let base = format!("http://127.0.0.1:{port}");
    let native_id = match &ctx.native_session_id {
        Some(id) => id.clone(),
        None => {
            let response = ctx
                .http()
                .post(format!("{base}/session"))
                .header("content-type", "application/json")
                .body("{}")
                .send()
                .await?;
            let session: Value = opencode_setup_response(response)?.json().await?;
            let id = session
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("opencode session response had no id"))?
                .to_string();
            ctx.persist_native_session_id(&id)?;
            id
        }
    };
    let events = ctx.http().get(format!("{base}/event")).send().await?;
    let events = opencode_setup_response(events)?;
    Ok((native_id, base, events))
}

async fn opencode_pre_accept_setup(
    ctx: &mut TurnCtx,
    store: NativeStore,
    binary: &crate::local::opencode::ResolvedBinary,
    database: &Path,
) -> Result<(String, String, reqwest::Response)> {
    loop {
        let remaining = ctx.orx_retry_remaining();
        let attempt = opencode_setup_attempt(ctx, store, binary, database);
        let result = match remaining {
            Some(remaining) => tokio::time::timeout(remaining, attempt)
                .await
                .map_err(|_| anyhow!("OpenCode setup exceeded the ORX retry budget"))?,
            None => attempt.await,
        };
        match result {
            Ok(setup) => {
                ctx.clear_retry_status();
                return Ok(setup);
            }
            Err(error) => {
                let (retryable, explicit) =
                    if let Some(http) = error.downcast_ref::<OpenCodeSetupHttpError>() {
                        (
                            http.status.as_u16() == 408
                                || http.status.as_u16() == 429
                                || http.status.is_server_error(),
                            http.retry_after,
                        )
                    } else if let Some(request) = error.downcast_ref::<reqwest::Error>() {
                        (
                            request.is_connect() || request.is_timeout() || request.is_request(),
                            None,
                        )
                    } else {
                        (false, None)
                    };
                let retry = retryable
                    .then(|| ctx.schedule_orx_retry(explicit))
                    .flatten();
                let Some((retry_number, delay)) = retry else {
                    ctx.mark_delivery(DeliveryState::NotSent);
                    ctx.mark_terminal_failure("opencode_setup", error.to_string());
                    return Err(error);
                };
                ctx.show_retry_status(
                    "orx",
                    "Reconnecting to OpenCode",
                    retry_number as i64 + 1,
                    Some(ORX_MAX_ATTEMPTS as i64),
                    Some(crate::store::now_ms() + delay.as_millis() as i64),
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
}

async fn run_turn(ctx: &mut TurnCtx) -> Result<()> {
    // Native permission and question requests die with their turn. Clear any
    // crash/restart leftovers before a new live request can be surfaced.
    ctx.host
        .resolve_stale_prompts(&ctx.session_id, true)
        .await?;

    let native_session = match ctx.native_session_id.clone() {
        Some(id) => tokio::task::spawn_blocking(move || native_store::opencode_session(&id))
            .await
            .map_err(|error| anyhow!("OpenCode session lookup failed: {error}"))??,
        None => None,
    };
    let store = native_session
        .as_ref()
        .map(|session| session.store)
        .unwrap_or(NativeStore::Isolated);
    if ctx.native_session_id.is_some() && native_session.is_none() {
        if let Some(recovery) = super::native_recovery_context(ctx, "OpenCode") {
            ctx.text = format!("{recovery}\n\n{}", ctx.text);
        }
        ctx.native_session_id = None;
    }

    let binary = crate::local::opencode::resolve_binary().await?;
    let database = match native_session {
        Some(session) => session.path,
        None => native_store::prepare_opencode(store)?,
    };
    if binary.protocol == crate::local::opencode::Protocol::V2 {
        let result = v2::run_turn(ctx, store, binary, database).await;
        ctx.host.opencode.untrack(&ctx.session_id);
        return result;
    }
    let (native_id, base, events) =
        opencode_pre_accept_setup(ctx, store, &binary, &database).await?;
    let mut stream = events.bytes_stream();

    let mut body = json!({
        "parts": [{ "type": "text", "text": ctx.text }],
        // Select opencode's built-in agent from the session's mode: `plan` (the
        // read-only planning agent — allows inspection, denies edits) vs `build`
        // (the default). The message endpoint takes `agent` directly (verified),
        // so no separate switch call is needed.
        "agent": opencode_agent(ctx.plan_mode),
    });
    if let Some(model) = &ctx.model {
        if let Some((provider, model_id)) = model.split_once('/') {
            body["model"] = json!({ "providerID": provider, "modelID": model_id });
        }
    }
    // Reasoning → opencode's provider-specific `variant` (the serve API's
    // session-message field, mirroring `opencode run --variant`). Omitted for
    // `Default`, so the model's own reasoning default stands (issue #123).
    if let Some(variant) = opencode_variant(ctx.reasoning_level.as_deref()) {
        body["variant"] = json!(variant);
    }
    let prompt_id = v1_prompt_id();
    body["messageID"] = json!(prompt_id);
    let turn_started_at = crate::store::now_ms();
    persist_scope(
        ctx,
        json!({"native": native_id, "startedAt": turn_started_at, "roots": [], "prompt": prompt_id,
            "endpoint": base}),
    );
    track_turn(ctx, &native_id, turn_started_at);
    let send = ctx
        .http()
        .post(format!("{base}/session/{native_id}/message"))
        .json(&body)
        .send();
    ctx.persist_delivery(DeliveryState::Unknown)?;
    tokio::pin!(send);

    // Parts are attributed via message.updated role info; a part arriving
    // before its message would be misfiled, and assistant messages are always
    // announced before their parts stream.
    let mut assistant_msgs: HashSet<String> = HashSet::new();
    // Sub-agent child sessions spawned by a `task` tool this turn: child
    // sessionID → the task spawn part's id. Their events (a foreign sessionID)
    // route into that part's `children` instead of being dropped.
    let mut sub_sessions: HashMap<String, String> = HashMap::new();
    // Native plan exit is an ordinary `question.asked`; connect it to the
    // preceding `plan_exit` tool through the question's `tool.callID`.
    let mut plan_exit_calls: HashSet<String> = HashSet::new();
    let mut buf = String::new();

    let result: Result<()> = async {
        loop {
            tokio::select! {
                chunk = stream.next() => {
                    let Some(chunk) = chunk else {
                        return Err(anyhow!("opencode event stream ended mid-turn"));
                    };
                    buf.push_str(&String::from_utf8_lossy(&chunk?));
                    while let Some(pos) = buf.find('\n') {
                        let line = buf[..pos].trim().to_string();
                        buf.drain(..=pos);
                        let Some(data) = line.strip_prefix("data: ") else { continue };
                        let Ok(event) = serde_json::from_str::<Value>(data) else { continue };
                        if let Some(part) = event
                            .get("properties")
                            .and_then(|props| props.get("part"))
                            .filter(|part| {
                                part.get("sessionID").and_then(Value::as_str) == Some(native_id.as_str())
                                    && part.get("type").and_then(Value::as_str) == Some("tool")
                                    && part.get("tool").and_then(Value::as_str) == Some("plan_exit")
                            })
                        {
                            if let Some(call_id) = part.get("callID").and_then(Value::as_str) {
                                plan_exit_calls.insert(call_id.to_string());
                            }
                        }
                        // Interactive prompts (permission/question) pause the turn and
                        // are handled async (emit a card, or auto-reply per mode); all
                        // other events are message/part updates handled synchronously.
                        if !handle_prompt_event(
                            ctx,
                            &native_id,
                            &base,
                            &event,
                            &plan_exit_calls,
                        )
                        .await?
                        {
                            handle_event(ctx, &native_id, &event, &mut assistant_msgs, &mut sub_sessions);
                        }
                    }
                }
                resp = &mut send => {
                    // Turn done — the response body is the final assistant message;
                    // merge its parts as the authoritative versions.
                    let resp = resp?.error_for_status()?;
                    ctx.mark_delivery(DeliveryState::Accepted);
                    let message = resp.json::<Value>().await?;
                    if let Some(error) = opencode_response_error(&message) {
                        ctx.mark_native_retry_exhausted();
                        ctx.mark_terminal_failure("opencode_terminal", error);
                        return Err(anyhow!("{error}"));
                    }
                    ctx.clear_retry_status();
                    if !opencode_response_is_current(&message, turn_started_at) {
                        let message = "OpenCode returned an earlier assistant message instead of replying to this turn. Update OpenCode or start a new chat.";
                        ctx.mark_terminal_failure("opencode_stale_response", message);
                        return Err(anyhow!(message));
                    }
                    if let Some(parts) = message.get("parts").and_then(Value::as_array) {
                        for part in parts {
                            if let Some(wire) = to_wire_part(part) {
                                // Preserve children: the final `task` part carries
                                // none, but its row already streamed the sub-agent
                                // transcript into `children`.
                                ctx.upsert_part_preserving_children(wire);
                            }
                        }
                        ctx.mark_final_text_tail();
                    }
                    return Ok(());
                }
            }
        }
    }
    .await;
    let captured = capture_v1(
        &*ctx,
        History::Server(ctx.http(), &base),
        vec![(native_id.clone(), None)],
        (turn_started_at, None),
        None,
    )
    .await;
    ctx.host.opencode.untrack(&ctx.session_id);
    // A failed turn or read may have missed root steps: keep watching the prompt's run.
    let root = result.is_err() || captured.failed.is_some();
    hold_and_watch(
        ctx,
        (root || !captured.background.is_empty()).then(|| {
            json!({"native": native_id, "startedAt": turn_started_at,
                "roots": captured.background, "parents": captured.parents,
                "prompt": root.then_some(&prompt_id), "endpoint": base})
        }),
        &base,
    );
    result
}

/// Records this turn's steps and tool invokers from native history, descending into every
/// subagent session a `task` part names (grandchildren included). Runs after the live stream,
/// which can miss the final events, after an interrupt dropped the turn, and for background
/// subagents that outlive it. `roots` pairs each session with the `WirePart.id` of the task part
/// that spawned it (`None` for the chat's own session, bounded to `prompt`'s run when given).
/// A watcher passes `(woken_only, owned)`: sessions read only for the runs its `owned` subagents'
/// results natively woke, and the subagents it awaits.
async fn capture_v1(
    sink: &impl UsageSink,
    history: History<'_>,
    roots: Vec<(String, Option<String>)>,
    (started_at, prompt): (i64, Option<&str>),
    watched: Option<(&HashSet<String>, &HashSet<String>)>,
) -> V1Capture {
    let (woken_only, owned) = watched.unzip();
    // Woken-only sessions go first, so they are read last: a whole read of the same session wins.
    let mut pending: Vec<_> = woken_only
        .into_iter()
        .flatten()
        .map(|session| (session.clone(), None))
        .chain(roots)
        .collect();
    let mut captured = V1Capture::default();
    let mut tree = TreeState::default();
    while let Some((session, spawn)) = pending.pop() {
        // A model-chosen `task_id` can resume any session, even an ancestor.
        if !captured.visited.insert(session.clone()) {
            continue;
        }
        let messages = match v1_history(history, &session).await {
            Ok(messages) => messages,
            Err(error) => {
                eprintln!("orx up: could not reconcile OpenCode usage for {session}: {error}");
                captured.failed = Some(error);
                continue;
            }
        };
        let deliveries = v1_deliveries(&messages);
        captured.delivered.extend(deliveries.values().cloned());
        let woken_only = spawn.is_none() && woken_only.is_some_and(|set| set.contains(&session));
        let run = prompt
            .filter(|_| spawn.is_none())
            .map(|prompt| v1_prompt_run(&messages, prompt));
        for message in v1_turn_messages(&messages, started_at) {
            // A run a background result woke natively belongs to the execution owning that subagent.
            let woken_by = v1_woken_by(message, &deliveries);
            match woken_by.map(|child| {
                owned.is_some_and(|owned| owned.contains(child))
                    || captured.background.iter().any(|(bg, _)| bg == child)
            }) {
                Some(false) => continue,
                Some(true) => {}
                None if woken_only => continue,
                None => {
                    let id = message.pointer("/info/id").and_then(Value::as_str);
                    if run
                        .as_ref()
                        .is_some_and(|run| !run.contains(id.unwrap_or("")))
                    {
                        continue;
                    }
                }
            }
            let identity = v1_identity(&message["info"]).map(|(_, identity)| identity);
            for part in message["parts"].as_array().into_iter().flatten() {
                if woken_by.is_some() {
                    if let Some(wire) = to_wire_part(part).filter(|wire| wire.kind == "tool") {
                        sink.tool_evidence(&wire);
                    }
                }
                let Some(id) = part.get("id").and_then(Value::as_str) else {
                    continue;
                };
                // No transcript holds a woken run, so its parts keep their native ids (as evidence does).
                let wire = match (&spawn, woken_by) {
                    (Some(spawn), None) => format!("{spawn}:{id}"),
                    _ => id.to_string(),
                };
                if let Some(child) = task_session(part) {
                    if part.pointer("/state/metadata/background") == Some(&json!(true)) {
                        captured
                            .background
                            .push((child.to_string(), Some(wire.clone())));
                        captured.parents.insert(session.clone());
                    }
                    pending.push((child.to_string(), Some(wire.clone())));
                }
                record_v1_part(
                    sink,
                    part,
                    identity.as_ref(),
                    Some(wire),
                    spawn.is_some(),
                    &mut tree,
                );
            }
        }
    }
    captured
}

/// Where native history is read: a session's server, or its database when no server may run.
#[derive(Clone, Copy)]
pub(super) enum History<'a> {
    Server(&'a reqwest::Client, &'a str),
    Database(&'a std::path::Path),
}

impl<'a> From<&'a AgentEndpoint> for History<'a> {
    fn from(endpoint: &'a AgentEndpoint) -> Self {
        Self::Server(&endpoint.client, &endpoint.base_url)
    }
}

/// A session its native database no longer holds.
#[derive(Debug)]
struct SessionGone(String);

impl std::fmt::Display for SessionGone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpenCode session {} no longer exists", self.0)
    }
}

impl std::error::Error for SessionGone {}

/// Reads a session's history from its native database off the async runtime.
async fn stored(
    database: &std::path::Path,
    session: &str,
    read: fn(&std::path::Path, &str) -> Result<Option<Value>>,
) -> Result<Value> {
    let (path, id) = (database.to_path_buf(), session.to_string());
    tokio::task::spawn_blocking(move || read(&path, &id))
        .await??
        .ok_or_else(|| SessionGone(session.to_string()).into())
}

async fn v1_history(history: History<'_>, session: &str) -> Result<Value> {
    let (http, base) = match history {
        History::Server(http, base) => (http, base),
        History::Database(path) => {
            return stored(path, session, native_store::opencode_database::v1_history).await
        }
    };
    let request = http.get(format!("{base}/session/{session}/message")).send();
    tokio::time::timeout(Duration::from_secs(10), async {
        Ok(request.await?.error_for_status()?.json::<Value>().await?)
    })
    .await
    .map_err(|_| anyhow!("timed out reading OpenCode session {session}"))?
}

#[derive(Default)]
struct V1Capture {
    visited: HashSet<String>,
    /// Background subagents this turn spawned, which can outlive it.
    background: Vec<(String, Option<String>)>,
    /// Sessions those subagents report their results to, natively waking a run there.
    parents: HashSet<String>,
    /// Subagents whose results reached a session read.
    delivered: HashSet<String>,
    /// Why a session's history could not be read.
    failed: Option<crate::error::Error>,
}

/// One watcher poll of a held execution's background subagents and the runs their results natively
/// woke (no app turn) in each session they report to: the tree's busy sessions (native
/// `GET /session/status` lists only non-idle sessions), and whether a result is undelivered.
async fn poll_v1_background(
    sink: &impl UsageSink,
    endpoint: &AgentEndpoint,
    (started_at, prompt): (i64, Option<&str>),
    roots: &[(String, Option<String>)],
    parents: &mut HashSet<String>,
    owned: &mut HashSet<String>,
) -> Result<(Vec<String>, bool)> {
    // Read before capturing, so a session idle here has persisted everything captured below.
    let busy: Value = endpoint
        .client
        .get(format!("{}/session/status", endpoint.base_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let woken_only = parents
        .iter()
        .filter(|parent| !roots.iter().any(|(root, _)| root == *parent))
        .cloned()
        .collect();
    let captured = capture_v1(
        sink,
        endpoint.into(),
        roots.to_vec(),
        (started_at, prompt),
        Some((&woken_only, owned)),
    )
    .await;
    if let Some(error) = captured.failed {
        return Err(error);
    }
    // Background subagents the tree spawned since the turn: await their results where they report.
    owned.extend(captured.background.into_iter().map(|(child, _)| child));
    parents.extend(captured.parents);
    let active = captured
        .visited
        .iter()
        .chain(parents.iter())
        .filter(|session| busy.get(session.as_str()).is_some())
        .cloned()
        .collect();
    let undelivered = owned
        .iter()
        .any(|child| !captured.delivered.contains(child));
    Ok((active, undelivered))
}

/// How long an idle tree waits for a finished subagent's result to reach and wake its parent.
const DELIVERY_GRACE_POLLS: usize = 15;

/// Native background-result deliveries in a V1 listing: user message id → the subagent session its
/// synthetic `<task id="…">` part reports.
fn v1_deliveries(messages: &Value) -> HashMap<String, String> {
    messages
        .as_array()
        .into_iter()
        .flatten()
        .filter(|message| message.pointer("/info/role").and_then(Value::as_str) == Some("user"))
        .filter_map(|message| {
            let child = message["parts"]
                .as_array()?
                .iter()
                .find_map(|part| v1_delivered_child(part))?;
            Some((
                message.pointer("/info/id")?.as_str()?.to_string(),
                child.to_string(),
            ))
        })
        .collect()
}

fn v1_delivered_child(part: &Value) -> Option<&str> {
    if part.get("synthetic").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    let rest = part.get("text")?.as_str()?.strip_prefix("<task id=\"")?;
    rest.split_once('"').map(|(child, _)| child)
}

/// The subagent whose delivered result woke this assistant message, if one did.
fn v1_woken_by<'a>(message: &Value, deliveries: &'a HashMap<String, String>) -> Option<&'a str> {
    deliveries
        .get(message.pointer("/info/parentID")?.as_str()?)
        .map(String::as_str)
}

const BACKGROUND_POLL: Duration = if cfg!(test) {
    Duration::from_millis(10)
} else {
    Duration::from_secs(2)
};

/// Assistant messages this turn created, in a native `GET /session/{id}/message` listing.
fn v1_turn_messages(messages: &Value, started_at: i64) -> impl Iterator<Item = &Value> {
    messages
        .as_array()
        .into_iter()
        .flatten()
        .filter(move |message| {
            message.pointer("/info/role").and_then(Value::as_str) == Some("assistant")
                && message
                    .pointer("/info/time/created")
                    .and_then(Value::as_i64)
                    .is_some_and(|created| created >= started_at)
        })
}

/// A V1 prompt id orx submits: native's ascending time prefix, then a `-` native ids never hold.
fn v1_prompt_id() -> String {
    let time = (crate::store::now_ms() as u64).wrapping_mul(0x1000) & 0xffff_ffff_ffff;
    let random = uuid::Uuid::new_v4().simple().to_string();
    format!("msg_{time:012x}-orx{}", &random[..10])
}

/// Message ids of `prompt`'s run: everything after it until orx's next prompt. Native's own user
/// messages (compaction, subtask summaries, result deliveries) stay inside the run.
fn v1_prompt_run<'a>(messages: &'a Value, prompt: &str) -> HashSet<&'a str> {
    messages
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|message| message.pointer("/info/id")?.as_str())
        .skip_while(|id| *id != prompt)
        .skip(1)
        .take_while(|id| !id.contains("-orx"))
        .collect()
}

/// The native model OpenCode ran an assistant message on (`modelID`/`providerID`).
fn v1_identity(info: &Value) -> Option<(String, crate::store::InvocationIdentity)> {
    Some((
        info.get("id")?.as_str()?.to_string(),
        crate::store::InvocationIdentity {
            harness: "opencode".into(),
            model: info.get("modelID")?.as_str()?.to_string(),
            provider: Some(info.get("providerID")?.as_str()?.to_string()),
        },
    ))
}

/// The `WirePart.id` a native tool part is stored under, if it is in the transcript.
fn v1_tool_part_id(
    part: &Value,
    native_id: &str,
    sub_sessions: &HashMap<String, String>,
) -> Option<String> {
    let id = part.get("id")?.as_str()?;
    let session = part.get("sessionID")?.as_str()?;
    if session == native_id {
        Some(id.to_string())
    } else {
        sub_sessions
            .get(session)
            .map(|spawn| format!("{spawn}:{id}"))
    }
}

/// The usage sample a V1 part proves. A `step-start` means the provider began streaming a request;
/// its `step-finish` carries that request's tokens under the same sample id, so a request that never
/// finishes (cancel, error, a native retry) stays a known request without counters.
fn v1_step_sample(
    part: &Value,
    identity: Option<&crate::store::InvocationIdentity>,
    child: bool,
    steps: &mut HashMap<String, String>,
) -> Option<(
    String,
    crate::store::Attribution,
    crate::store::TokenUsage,
    bool,
)> {
    let attribution = crate::store::Attribution::native(
        "opencode",
        identity.map(|identity| identity.model.as_str()),
        identity.and_then(|identity| identity.provider.as_deref()),
        crate::store::Missing::unidentified(child),
    );
    let id = part.get("id")?.as_str()?;
    let message = part.get("messageID")?.as_str()?;
    match part.get("type").and_then(Value::as_str)? {
        "step-start" => {
            steps.insert(message.to_string(), id.to_string());
            Some((
                id.to_string(),
                attribution,
                crate::store::TokenUsage::default(),
                false,
            ))
        }
        "step-finish" => {
            let (usage, complete) = opencode_sample(Some(part.get("tokens")?));
            // Part ids ascend, so a message's latest step-start is this step's.
            let step = steps.get(message).map_or(id, String::as_str);
            Some((step.to_string(), attribution, usage, complete))
        }
        _ => None,
    }
}

fn record_v1_part(
    sink: &impl UsageSink,
    part: &Value,
    identity: Option<&crate::store::InvocationIdentity>,
    tool_part: Option<String>,
    child: bool,
    tree: &mut TreeState,
) {
    if let Some((sample, attribution, usage, complete)) =
        v1_step_sample(part, identity, child, &mut tree.steps)
    {
        sink.sample(&sample, attribution, usage, complete);
    }
    if let (Some(tool_part), Some(identity)) = (tool_part, identity) {
        if part.get("type").and_then(Value::as_str) == Some("tool")
            && tree.invokers.insert(tool_part.clone())
        {
            sink.invoker(&tool_part, &identity.model, identity.provider.as_deref());
        }
    }
}

/// The subagent session a V1 `task` part ran (also a resumed `task_id`, which gets no
/// `session.created`).
fn task_session(part: &Value) -> Option<&str> {
    (part.get("tool").and_then(Value::as_str) == Some("task"))
        .then(|| part.pointer("/state/metadata/sessionId")?.as_str())
        .flatten()
}

/// V1 evidence already recorded, and each message's latest `step-start` part id.
#[derive(Default)]
struct TreeState {
    invokers: HashSet<String>,
    steps: HashMap<String, String>,
}

/// Destination for native usage evidence: the live turn, or its execution after an interrupt
/// dropped the turn.
pub(crate) trait UsageSink: Sync {
    fn sample(
        &self,
        id: &str,
        attribution: crate::store::Attribution,
        usage: crate::store::TokenUsage,
        complete: bool,
    );
    fn invoker(&self, part_id: &str, model: &str, provider: Option<&str>);
    /// A tool part no transcript holds (a natively woken run): kept so a run it launched binds.
    fn tool_evidence(&self, _part: &WirePart) {}
}

impl UsageSink for TurnCtx {
    fn sample(
        &self,
        id: &str,
        attribution: crate::store::Attribution,
        usage: crate::store::TokenUsage,
        complete: bool,
    ) {
        self.record_attributed_usage(id, attribution, usage, complete);
    }

    fn invoker(&self, part_id: &str, model: &str, provider: Option<&str>) {
        self.record_tool_invoker(part_id, model, provider);
    }
}

/// A turn's usage execution, written by [`capture_interrupted`] after the turn's future is gone.
pub(crate) struct ExecutionSink {
    execution_id: String,
    session_id: String,
}

impl UsageSink for ExecutionSink {
    fn sample(
        &self,
        id: &str,
        attribution: crate::store::Attribution,
        usage: crate::store::TokenUsage,
        complete: bool,
    ) {
        if let Err(error) = crate::store::Store::open().and_then(|store| {
            store.record_attributed_sample(
                &self.execution_id,
                id,
                "opencode",
                &attribution,
                &usage,
                complete,
            )
        }) {
            eprintln!("orx up: could not persist interrupted OpenCode usage: {error}");
        }
    }

    fn invoker(&self, part_id: &str, model: &str, provider: Option<&str>) {
        let identity = crate::store::InvocationIdentity {
            harness: "opencode".into(),
            model: model.to_string(),
            provider: provider.map(str::to_string),
        };
        if let Err(error) = crate::store::Store::open().and_then(|store| {
            store.record_native_invocation(part_id, &identity, Some(&self.session_id))
        }) {
            eprintln!("orx up: could not capture interrupted OpenCode tool identity: {error}");
        }
    }

    fn tool_evidence(&self, part: &WirePart) {
        if let Err(error) = crate::store::Store::open().and_then(|store| {
            store.set_native_scope(
                &self.execution_id,
                &format!("tool-part:{}", part.id),
                &serde_json::to_value(part)?,
            )
        }) {
            eprintln!("orx up: could not keep OpenCode tool evidence: {error}");
        }
    }
}

impl ExecutionSink {
    /// Closes a held execution now if its turn already ended; its tree is read, so a restart has
    /// nothing left to adopt.
    fn release(&self) {
        if let Err(error) = crate::store::Store::open().and_then(|store| {
            store.clear_native_scope(&self.execution_id, BACKGROUND_SCOPE)?;
            store.release_usage_execution(&self.execution_id)
        }) {
            eprintln!("orx up: could not release OpenCode usage: {error}");
        }
    }
}

fn execution_sink(ctx: &TurnCtx) -> Option<ExecutionSink> {
    let execution_id = crate::store::Store::open()
        .and_then(|store| store.open_usage_execution(&ctx.turn_id))
        .ok()??;
    Some(ExecutionSink {
        execution_id,
        session_id: ctx.session_id.clone(),
    })
}

/// Persists what a restart adopts to finish this turn's capture; written before submission, so a
/// crash at any point leaves it.
fn persist_scope(ctx: &TurnCtx, mut scope: Value) -> Value {
    scope["session"] = json!(ctx.session_id);
    scope["model"] = json!(ctx.model);
    if let Err(error) = execution_sink(ctx)
        .ok_or_else(|| anyhow!("no open usage execution"))
        .and_then(|sink| {
            crate::store::Store::open()?.set_native_scope(
                &sink.execution_id,
                BACKGROUND_SCOPE,
                &scope,
            )
        })
    {
        eprintln!("orx up: could not persist OpenCode scope: {error}");
    }
    scope
}

/// Holds and watches this turn's execution while `scope` has native history left to read; the
/// watcher releases it once settled. `None` clears the scope: the turn captured everything.
fn hold_and_watch(ctx: &TurnCtx, scope: Option<Value>, original: &str) {
    let Some(sink) = execution_sink(ctx) else {
        return;
    };
    let Ok(store) = crate::store::Store::open() else {
        return;
    };
    let Some(scope) = scope else {
        if let Err(error) = store.clear_native_scope(&sink.execution_id, BACKGROUND_SCOPE) {
            eprintln!("orx up: could not clear OpenCode scope: {error}");
        }
        return;
    };
    if !store
        .hold_usage_execution(&sink.execution_id)
        .unwrap_or(false)
    {
        return;
    }
    let scope = persist_scope(ctx, scope);
    let original = Some(original.to_string());
    tokio::spawn(watch_held(ctx.host.opencode.clone(), sink, scope, original));
}

const BACKGROUND_SCOPE: &str = "opencode-background";

/// Held executions a dead process left, adopted at startup so recovery cannot close them before
/// [`recover_adopted_background`] reads their native history.
static ADOPTED: Mutex<Vec<(String, Value)>> = Mutex::new(Vec::new());

pub(crate) fn adopt_orphaned_background(store: &crate::store::Store) -> Result<()> {
    for (execution, _, scope, orphaned) in store.native_scopes(BACKGROUND_SCOPE)? {
        if orphaned && store.hold_usage_execution(&execution)? {
            ADOPTED
                .lock()
                .map_err(|_| anyhow!("OpenCode recovery lock failed"))?
                .push((execution, scope));
        }
    }
    Ok(())
}

/// Resumes watching each adopted execution's background tree from native history.
pub(crate) fn recover_adopted_background(agent: std::sync::Arc<crate::local::opencode::AgentHost>) {
    let adopted = ADOPTED
        .lock()
        .map(|mut adopted| std::mem::take(&mut *adopted));
    for (execution_id, scope) in adopted.unwrap_or_default() {
        let sink = ExecutionSink {
            execution_id,
            session_id: scope["session"].as_str().unwrap_or_default().to_string(),
        };
        tokio::spawn(watch_held(agent.clone(), sink, scope, None));
    }
}

/// `original`: the base URL of the server that ran the turn, when this process started it.
async fn watch_held(
    agent: std::sync::Arc<crate::local::opencode::AgentHost>,
    sink: ExecutionSink,
    scope: Value,
    original: Option<String>,
) {
    if scope["prompt"].is_string() && scope["roots"].as_array().is_some_and(Vec::is_empty) {
        // Starting a server would resume native work, so a root scope reads the native database.
        let session = sink.session_id.as_str();
        let native = scope["native"].as_str().unwrap_or_default().to_string();
        let database = loop {
            let id = native.clone();
            match tokio::task::spawn_blocking(move || native_store::opencode_session(&id))
                .await
                .map_err(crate::error::Error::from)
                .and_then(|found| found)
            {
                Ok(location) => break location,
                Err(error) => eprintln!("orx up: retrying OpenCode root capture: {error}"),
            }
            tokio::time::sleep(BACKGROUND_POLL).await;
        };
        let Some(database) = database else {
            record_unrecoverable(&sink, &SessionGone(native).into(), false);
            return sink.release();
        };
        let owner = || async {
            let Some(original) = &original else {
                return persisted_owner(&scope).await;
            };
            match agent.endpoint_for(session).await {
                Some(endpoint) if endpoint.base_url == *original => Owner::Live(endpoint),
                _ => Owner::Gone,
            }
        };
        settle_stored(&sink, &scope, &database.path, owner).await;
    } else {
        let endpoint = |missing| background_endpoint(&agent, &sink.session_id, &scope, missing);
        settle_watch(&sink, &scope, endpoint).await;
    }
    sink.release();
}

/// How a root scope's native work stood at one read of its database.
enum Stored {
    /// Nothing more will run; `unfinished`: a session native never finished.
    Settled { unfinished: Option<String> },
    /// The turn's own server still runs its tree.
    Running,
}

/// Who can still run a root scope's native work.
enum Owner {
    /// The turn's original server: its status says when the tree settled.
    Live(AgentEndpoint),
    /// The original server is gone, so the database holds all it wrote.
    Gone,
}

/// After a restart: the turn's original V1 server, reconnected read-only at its persisted endpoint
/// (a refused connection: gone). V2 servers exit with orx once their stdin closes.
async fn persisted_owner(scope: &Value) -> Owner {
    if scope["v2"] == true {
        return Owner::Gone;
    }
    let Some(base) = scope["endpoint"].as_str() else {
        return Owner::Gone;
    };
    let endpoint = AgentEndpoint {
        base_url: base.to_string(),
        client: reqwest::Client::new(),
        protocol: crate::local::opencode::Protocol::V1,
        legacy_v2_api: false,
    };
    match endpoint
        .client
        .get(format!("{base}/session/status"))
        .send()
        .await
    {
        Err(error) if error.is_connect() => Owner::Gone,
        _ => Owner::Live(endpoint),
    }
}

/// Captures a root scope's prompt run and descendants from its native database until native
/// evidence says it ended; work no process can finish seals it with a partial marker.
async fn settle_stored<F: std::future::Future<Output = Owner>>(
    sink: &impl UsageSink,
    scope: &Value,
    database: &std::path::Path,
    owner: impl Fn() -> F,
) {
    let (Some(native), Some(started_at), Some(prompt)) = (
        scope["native"].as_str(),
        scope["startedAt"].as_i64(),
        scope["prompt"].as_str(),
    ) else {
        return record_unrecoverable(sink, &anyhow!("invalid OpenCode scope"), false);
    };
    let history = History::Database(database);
    let mut captured = v2::Captured::default();
    loop {
        let polled: Result<Stored> = async {
            // Read before capturing, so a tree found idle has persisted everything captured below.
            let owner = owner().await;
            let active = match &owner {
                Owner::Live(endpoint) => Some(active_sessions(endpoint).await?),
                Owner::Gone => None,
            };
            let sessions: Vec<String> = if scope["v2"] == true {
                v2::capture_tree(
                    sink,
                    history,
                    native,
                    started_at,
                    &mut captured,
                    true,
                    Some(prompt),
                )
                .await?;
                captured.sessions(native)
            } else {
                let capture = capture_v1(
                    sink,
                    history,
                    vec![(native.to_string(), None)],
                    (started_at, Some(prompt)),
                    None,
                )
                .await;
                if let Some(error) = capture.failed {
                    return Err(error);
                }
                capture.visited.into_iter().collect()
            };
            if let Some(active) = &active {
                if sessions.iter().any(|session| active.contains(session)) {
                    return Ok(Stored::Running);
                }
            }
            // V1 never resumes: a settled V1 database is final.
            let unfinished = if scope["v2"] == true {
                unfinished_session(database, sessions, native).await?
            } else {
                None
            };
            Ok(Stored::Settled { unfinished })
        }
        .await;
        match polled {
            Ok(Stored::Settled { unfinished }) => {
                if let Some(session) = unfinished {
                    let why = anyhow!(
                        "OpenCode left {session} unfinished; resumed work is not this execution's"
                    );
                    record_unrecoverable(sink, &why, session != native);
                }
                return;
            }
            Ok(Stored::Running) => {}
            Err(error) => match error.downcast_ref::<SessionGone>() {
                Some(SessionGone(session)) => {
                    return record_unrecoverable(sink, &error, session != native)
                }
                None => eprintln!("orx up: retrying OpenCode root capture: {error}"),
            },
        }
        tokio::time::sleep(BACKGROUND_POLL).await;
    }
}

/// The first of `sessions` (the root first) holding an unfinished V2 execution claim.
async fn unfinished_session(
    database: &std::path::Path,
    mut sessions: Vec<String>,
    native: &str,
) -> Result<Option<String>> {
    sessions.sort_by_key(|session| session != native);
    let path = database.to_path_buf();
    tokio::task::spawn_blocking(move || {
        for session in sessions {
            if native_store::opencode_database::v2_unfinished(&path, &session)? {
                return Ok(Some(session));
            }
        }
        Ok(None)
    })
    .await?
}

/// Sessions the server is running now.
async fn active_sessions(endpoint: &AgentEndpoint) -> Result<HashSet<String>> {
    let (path, pointer) = match endpoint.protocol {
        crate::local::opencode::Protocol::V1 => ("/session/status", ""),
        crate::local::opencode::Protocol::V2 => ("/api/session/active", "/data"),
    };
    let status: Value = endpoint
        .client
        .get(format!("{}{path}", endpoint.base_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(status
        .pointer(pointer)
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("OpenCode session status is invalid"))?
        .keys()
        .cloned()
        .collect())
}

/// Watches until the tree settles. Native failures retry on the next poll; history that is gone for
/// good leaves an explicit unresolved request beside what was already captured. `endpoint` gets the
/// session a read found missing, to confirm natively.
async fn settle_watch<F: std::future::Future<Output = Result<Option<AgentEndpoint>>>>(
    sink: &impl UsageSink,
    scope: &Value,
    endpoint: impl Fn(Option<String>) -> F,
) {
    let mut confirming = None;
    let watched = async {
        let mut watch = Watch::from_scope(scope)?;
        let mut waited = 0;
        let mut missing = None;
        loop {
            confirming = missing.take();
            if let Some(endpoint) = endpoint(confirming.clone()).await? {
                match watch.poll(sink, &endpoint).await {
                    Ok((busy, undelivered))
                        if busy.is_empty() && (!undelivered || waited >= DELIVERY_GRACE_POLLS) =>
                    {
                        return Ok(())
                    }
                    Ok((busy, _)) => waited += usize::from(busy.is_empty()),
                    Err(error) => {
                        missing = not_found_session(&error);
                        eprintln!("orx up: retrying OpenCode background capture: {error}");
                    }
                }
            }
            tokio::time::sleep(BACKGROUND_POLL).await;
        }
    };
    if let Err(gone) = watched.await {
        // The turn's own history is lost only while its prompt's run was still to read.
        let root = scope["prompt"].is_string()
            && confirming
                .as_deref()
                .is_none_or(|session| Some(session) == scope["native"].as_str());
        record_unrecoverable(sink, &gone, !root);
    }
}

/// The session a native read found missing: HTTP 404 on `…/session/{id}/…`.
fn not_found_session(error: &crate::error::Error) -> Option<String> {
    let error = error.downcast_ref::<reqwest::Error>()?;
    if error.status()? != reqwest::StatusCode::NOT_FOUND {
        return None;
    }
    let mut segments = error.url()?.path_segments()?;
    segments.find(|segment| *segment == "session")?;
    // `…/session/status` and `…/session/active` name no session.
    segments
        .next()
        .filter(|session| !matches!(*session, "status" | "active"))
        .map(str::to_string)
}

/// Stands in for the requests whose native history could not be read: the turn's own (`child`
/// false) or its subagents'.
fn record_unrecoverable(sink: &impl UsageSink, why: &crate::error::Error, child: bool) {
    eprintln!("orx up: OpenCode usage is unrecoverable: {why}");
    sink.sample(
        if child {
            "opencode-background:unrecoverable"
        } else {
            "opencode-root:unrecoverable"
        },
        crate::store::Attribution::Unresolved {
            reason: crate::store::Missing::unidentified(child),
        },
        crate::store::TokenUsage::default(),
        false,
    );
}

/// A held execution's background tree, as its watcher tracks it between polls.
struct Watch {
    native_id: String,
    started_at: i64,
    tree: WatchTree,
}

enum WatchTree {
    V1 {
        /// Sessions captured whole, each with the `WirePart.id` of the task part that spawned it.
        roots: Vec<(String, Option<String>)>,
        prompt: Option<String>,
        /// Sessions the subagents report their results to.
        parents: HashSet<String>,
        owned: HashSet<String>,
    },
    V2 {
        roots: Vec<(String, String)>,
        captured: v2::Captured,
        /// The turn's submitted prompt, when its own run is still to capture.
        prompt: Option<String>,
    },
}

impl Watch {
    fn from_scope(scope: &Value) -> Result<Self> {
        let (Some(native_id), Some(started_at), Some(roots)) = (
            scope["native"].as_str(),
            scope["startedAt"].as_i64(),
            scope["roots"].as_array(),
        ) else {
            return Err(anyhow!("invalid OpenCode background scope"));
        };
        let pairs = roots
            .iter()
            .filter_map(|root| Some((root.get(0)?.as_str()?.to_string(), root.get(1)?)));
        let prompt = scope["prompt"].as_str().map(str::to_string);
        let tree = if scope["v2"] == true {
            let roots: Vec<(String, String)> = pairs
                .filter_map(|(child, parent)| Some((child, parent.as_str()?.to_string())))
                .collect();
            WatchTree::V2 {
                captured: v2::Captured::watching(&roots),
                roots,
                prompt,
            }
        } else {
            let mut roots: Vec<(String, Option<String>)> = pairs
                .map(|(child, spawn)| (child, spawn.as_str().map(str::to_string)))
                .collect();
            let owned = roots.iter().map(|(child, _)| child.clone()).collect();
            // The chat session itself: its prompt's run, which `capture_v1` bounds by `prompt`.
            if prompt.is_some() {
                roots.push((native_id.to_string(), None));
            }
            WatchTree::V1 {
                prompt,
                owned,
                parents: scope["parents"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .chain([native_id])
                    .map(str::to_string)
                    .collect(),
                roots,
            }
        };
        Ok(Self {
            native_id: native_id.to_string(),
            started_at,
            tree,
        })
    }

    /// The tree's busy sessions, and whether a subagent's result is undelivered.
    async fn poll(
        &mut self,
        sink: &impl UsageSink,
        endpoint: &AgentEndpoint,
    ) -> Result<(Vec<String>, bool)> {
        match &mut self.tree {
            WatchTree::V1 {
                roots,
                prompt,
                parents,
                owned,
            } => {
                poll_v1_background(
                    sink,
                    endpoint,
                    (self.started_at, prompt.as_deref()),
                    roots,
                    parents,
                    owned,
                )
                .await
            }
            WatchTree::V2 {
                roots,
                captured,
                prompt,
            } => {
                v2::poll_background(
                    sink,
                    endpoint,
                    (&self.native_id, prompt.as_deref()),
                    roots,
                    self.started_at,
                    captured,
                )
                .await
            }
        }
    }
}

/// The session's live server, else one started from its stored runtime (it died, was replaced, or
/// this process restarted). `None` retries on the next poll; `Err` means the session was deleted
/// or its native history no longer exists.
async fn background_endpoint(
    agent: &crate::local::opencode::AgentHost,
    session_id: &str,
    scope: &Value,
    missing: Option<String>,
) -> Result<Option<AgentEndpoint>> {
    // A 404 alone may be transient; only the native store's own record makes it permanent.
    if let Some(missing) = missing {
        let id = missing.clone();
        if let Ok(Ok(None)) =
            tokio::task::spawn_blocking(move || native_store::opencode_session(&id)).await
        {
            return Err(anyhow!("OpenCode session {missing} no longer exists"));
        }
    }
    if let Some(endpoint) = agent.endpoint_for(session_id).await {
        return Ok(Some(endpoint));
    }
    match start_background_server(agent, session_id, scope).await {
        Ok(started) => started,
        Err(error) => {
            eprintln!("orx up: OpenCode is unavailable for background capture: {error}");
            Ok(None)
        }
    }
}

/// Outer `Err`: a transient failure; inner `Err`: the session or its native history is gone.
async fn start_background_server(
    agent: &crate::local::opencode::AgentHost,
    session_id: &str,
    scope: &Value,
) -> Result<Result<Option<AgentEndpoint>>> {
    let project = {
        let store = crate::store::Store::open()?;
        let Some(session) = store.get_chat_session(session_id)? else {
            return Ok(Err(anyhow!("chat session no longer exists")));
        };
        let Some(project) = store.get_local_project(&session.project_id)? else {
            return Ok(Err(anyhow!("project no longer exists")));
        };
        project
    };
    // Deleting: wait for the deletion to commit (gone above) or fail (a turn revives the server).
    if agent.is_retired(session_id) {
        return Ok(Ok(None));
    }
    let native = scope["native"].as_str().unwrap_or_default().to_string();
    let Some(native_session) =
        tokio::task::spawn_blocking(move || native_store::opencode_session(&native)).await??
    else {
        return Ok(Err(anyhow!("OpenCode session no longer exists")));
    };
    let binary = crate::local::opencode::resolve_binary().await?;
    agent
        .ensure(
            &project,
            session_id,
            scope["model"].as_str(),
            crate::local::opencode::ResolvedRuntime {
                binary,
                database: native_session.path,
                store: native_session.store,
            },
            tokio::sync::watch::channel(String::new()).0,
        )
        .await?;
    Ok(Ok(agent.endpoint_for(session_id).await))
}

/// A deleted session's server is about to stop: stop each background tree it holds natively, read
/// it to the end, then release it. One that did not stop, or a failed read, stays held for the
/// watcher, which closes it once the deletion commits or resumes if the deletion fails.
pub(crate) async fn reconcile_retiring(endpoint: &AgentEndpoint, session_id: &str) {
    let scopes =
        match crate::store::Store::open().and_then(|store| store.native_scopes(BACKGROUND_SCOPE)) {
            Ok(scopes) => scopes,
            Err(error) => {
                return eprintln!("orx up: could not reconcile OpenCode background usage: {error}")
            }
        };
    for (execution_id, _, scope, orphaned) in scopes {
        if orphaned || scope["session"] != session_id {
            continue;
        }
        let sink = ExecutionSink {
            execution_id,
            session_id: session_id.to_string(),
        };
        let stopped = tokio::time::timeout(Duration::from_secs(10), async {
            stop_tree(&mut Watch::from_scope(&scope)?, &sink, endpoint).await
        })
        .await
        .unwrap_or_else(|_| Err(anyhow!("timed out")));
        match stopped {
            Ok(true) => sink.release(),
            Ok(false) => eprintln!("orx up: OpenCode background subagents did not stop; kept held"),
            Err(error) => eprintln!("orx up: OpenCode background usage stays held: {error}"),
        }
    }
}

/// Aborts the tree's busy sessions (a stopped subagent's result can wake its parent, so repeat)
/// until a read finds none busy, which settles every run it captured. `false`: still running.
async fn stop_tree(
    watch: &mut Watch,
    sink: &impl UsageSink,
    endpoint: &AgentEndpoint,
) -> Result<bool> {
    for _ in 0..20 {
        let (busy, _) = watch.poll(sink, endpoint).await?;
        if busy.is_empty() {
            return Ok(true);
        }
        for session in &busy {
            crate::local::opencode::abort_session(endpoint, session).await?;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(false)
}

/// What an interrupt needs to capture the turn it aborts, held by the session's `AgentHost`.
pub(crate) struct TrackedTurn {
    pub(crate) native_id: String,
    started_at: i64,
    sink: ExecutionSink,
}

fn track_turn(ctx: &TurnCtx, native_id: &str, started_at: i64) {
    if let Some(sink) = execution_sink(ctx) {
        ctx.host.opencode.track(
            &ctx.session_id,
            TrackedTurn {
                native_id: native_id.to_string(),
                started_at,
                sink,
            },
        );
    }
}

/// Captures an interrupted turn's native evidence before the shared interrupt finalizes its usage.
/// V1 abort persists synchronously and cascades to every subagent; V2 settles asynchronously.
pub(crate) async fn capture_interrupted(
    agent: &std::sync::Arc<crate::local::opencode::AgentHost>,
    endpoint: &crate::local::opencode::AgentEndpoint,
    turn: &TrackedTurn,
) {
    let captured = match endpoint.protocol {
        crate::local::opencode::Protocol::V1 => capture_v1(
            &turn.sink,
            endpoint.into(),
            vec![(turn.native_id.clone(), None)],
            (turn.started_at, None),
            None,
        )
        .await
        .failed
        .is_none(),
        crate::local::opencode::Protocol::V2 => {
            v2::capture_interrupted(endpoint, &turn.sink, &turn.native_id, turn.started_at)
                .await
                .is_ok()
        }
    };
    let execution = &turn.sink.execution_id;
    let retry = crate::store::Store::open().and_then(|store| {
        if captured {
            // The interrupt stopped and read the whole tree, so a restart has nothing to adopt.
            store.clear_native_scope(execution, BACKGROUND_SCOPE)?;
            return Ok(None);
        }
        hold_for_retry(&store, execution)
    });
    match retry {
        Ok(Some(scope)) => {
            let sink = ExecutionSink {
                execution_id: execution.clone(),
                session_id: turn.sink.session_id.clone(),
            };
            let original = Some(endpoint.base_url.clone());
            tokio::spawn(watch_held(agent.clone(), sink, scope, original));
        }
        Ok(None) => {}
        Err(error) => eprintln!("orx up: could not keep OpenCode scope for retry: {error}"),
    }
}

/// After a failed capture: holds the execution so its finalize waits, and returns the scope
/// persisted before submission for a watcher to retry from.
fn hold_for_retry(store: &crate::store::Store, execution: &str) -> Result<Option<Value>> {
    let scope = store
        .native_scopes(BACKGROUND_SCOPE)?
        .into_iter()
        .find_map(|(id, _, scope, _)| (id == execution).then_some(scope));
    Ok(scope.filter(|_| store.hold_usage_execution(execution).unwrap_or(false)))
}

/// Native `tokens` as a sample that is complete when it measures the request's input and output.
fn opencode_sample(tokens: Option<&Value>) -> (crate::store::TokenUsage, bool) {
    let usage = tokens
        .filter(|tokens| !tokens.is_null())
        .map(opencode_native_usage)
        .unwrap_or_default();
    let complete = usage.input_tokens.is_some() && usage.output_tokens.is_some();
    (usage, complete)
}

fn opencode_response_error(message: &Value) -> Option<&str> {
    if let Some(error) = message
        .pointer("/info/error")
        .filter(|error| !error.is_null())
    {
        return Some(
            error
                .pointer("/data/message")
                .or_else(|| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("OpenCode reported an error"),
        );
    }
    (message.pointer("/info/summary").and_then(Value::as_bool) == Some(true)).then_some(
        "OpenCode compacted the context but did not resume this turn. Continue the chat to resume.",
    )
}

fn opencode_response_is_current(message: &Value, turn_started_at: i64) -> bool {
    message
        .pointer("/info/time/created")
        .and_then(Value::as_i64)
        .is_some_and(|created| created >= turn_started_at)
}

/// OpenCode assistant `tokens` occupying the context window:
/// `input + output + reasoning + cache.read + cache.write`. Returns `None` when
/// the object is absent, and `None` (not `Some(0)`) when every field is zero —
/// the early `message.updated` events carry an all-zero placeholder.
fn opencode_native_usage(tokens: &Value) -> crate::store::TokenUsage {
    let field = |name| tokens.get(name).and_then(Value::as_u64);
    let cache_read_tokens = tokens.pointer("/cache/read").and_then(Value::as_u64);
    let cache_write_tokens = tokens.pointer("/cache/write").and_then(Value::as_u64);
    let reasoning_tokens = field("reasoning");
    crate::store::TokenUsage {
        input_tokens: field("input").and_then(|input| {
            input
                .checked_add(cache_read_tokens?)
                .and_then(|n| n.checked_add(cache_write_tokens?))
        }),
        output_tokens: field("output").and_then(|output| output.checked_add(reasoning_tokens?)),
        cache_read_tokens,
        cache_write_tokens,
        reasoning_tokens,
    }
}

fn opencode_used_tokens(tokens: Option<&Value>) -> Option<u64> {
    let tokens = tokens?;
    let field = |v: &Value, name: &str| v.get(name).and_then(Value::as_u64).unwrap_or(0);
    let cache = tokens.get("cache").unwrap_or(&Value::Null);
    let total = field(tokens, "input")
        + field(tokens, "output")
        + field(tokens, "reasoning")
        + field(cache, "read")
        + field(cache, "write");
    (total > 0).then_some(total)
}

/// Whether a `session.updated` title is opencode's placeholder rather than a
/// real summary. The server seeds every session with `New session - <ISO
/// timestamp>` at creation and overwrites it once its own summarizer answers,
/// so the seed is a title to skip, not adopt.
fn is_opencode_seed_title(title: &str) -> bool {
    title.trim_start().starts_with("New session - ")
}

fn handle_event(
    ctx: &mut TurnCtx,
    native_id: &str,
    event: &Value,
    assistant_msgs: &mut HashSet<String>,
    sub_sessions: &mut HashMap<String, String>,
) {
    let props = event.get("properties").unwrap_or(&Value::Null);
    match event.get("type").and_then(Value::as_str) {
        Some("session.status") => {
            if props.get("sessionID").and_then(Value::as_str) != Some(native_id) {
                return;
            }
            let status = props.get("status").unwrap_or(&Value::Null);
            let status_type = status.get("type").and_then(Value::as_str);
            if status_type == Some("retry") {
                ctx.mark_delivery(DeliveryState::Accepted);
                let attempt = status.get("attempt").and_then(Value::as_i64).unwrap_or(1) + 1;
                let next = status.get("next").and_then(Value::as_i64).map(|next| {
                    if next > 1_000_000_000_000 {
                        next
                    } else {
                        crate::store::now_ms() + next
                    }
                });
                let message = status
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("OpenCode is retrying");
                ctx.show_retry_status("native", message, attempt, None, next);
            } else {
                if status_type == Some("busy") {
                    ctx.mark_delivery(DeliveryState::Accepted);
                }
                ctx.clear_retry_status();
            }
        }
        Some("session.error") => {
            if props.get("sessionID").and_then(Value::as_str) != Some(native_id) {
                return;
            }
            // OpenCode can emit this before recovering through automatic compaction.
            ctx.mark_delivery(DeliveryState::Accepted);
        }
        // A `task` tool spawns a sub-agent in a child session; opencode announces
        // it with `session.created` carrying the child's `parentID` = our
        // session. Link that child session to the spawning `task` tool row so its
        // events stream into that row's `children`.
        Some("session.created") => {
            let info = props.get("info").unwrap_or(&Value::Null);
            if info.get("parentID").and_then(Value::as_str) == Some(native_id) {
                if let Some(child_id) = info.get("id").and_then(Value::as_str) {
                    if let Some(spawn) = newest_task_part_id(&ctx.assistant.parts, sub_sessions) {
                        sub_sessions.insert(child_id.to_string(), spawn);
                    }
                }
            }
        }
        Some("message.updated") => {
            let info = props.get("info").unwrap_or(&Value::Null);
            let session = info.get("sessionID").and_then(Value::as_str);
            let is_assistant = info.get("role").and_then(Value::as_str) == Some("assistant")
                && info.get("summary").and_then(Value::as_bool) != Some(true);
            if session == Some(native_id)
                && info.get("role").and_then(Value::as_str) == Some("user")
            {
                ctx.mark_delivery(DeliveryState::Accepted);
            }
            // Record assistant message ids for the main session AND registered
            // sub-sessions, so a session's user parts (e.g. the task prompt echo)
            // can be filtered out — for both the transcript and sub-agent nesting.
            let ours =
                session == Some(native_id) || session.is_some_and(|s| sub_sessions.contains_key(s));
            if ours && is_assistant {
                if let Some(id) = info.get("id").and_then(Value::as_str) {
                    assistant_msgs.insert(id.to_string());
                }
            }
            // Usage is recorded from native history once the turn ends (`capture_v1`).
            // Only the MAIN session's tokens drive the context meter; a
            // sub-agent's smaller counts must not overwrite it.
            if session == Some(native_id) && is_assistant {
                // Several `message.updated` fire per message; the early ones have
                // no tokens yet, so skip a report until real numbers land. The
                // context window isn't in this event (provider config only), so
                // report the token count without one.
                if let Some(used) = opencode_used_tokens(info.get("tokens")) {
                    ctx.report_usage(ContextUsage {
                        used_tokens: used,
                        context_window: None,
                    });
                }
            }
        }
        Some("message.part.updated") => {
            let part = props.get("part").unwrap_or(&Value::Null);
            let session = part.get("sessionID").and_then(Value::as_str);
            let owned_by_assistant = part
                .get("messageID")
                .and_then(Value::as_str)
                .is_some_and(|mid| assistant_msgs.contains(mid));
            // A grandchild nests under its parent's task row.
            if let (Some(child), Some(spawn)) = (
                task_session(part),
                v1_tool_part_id(part, native_id, sub_sessions),
            ) {
                sub_sessions.entry(child.to_string()).or_insert(spawn);
            }
            // A sub-agent's part (foreign sessionID we've registered) streams
            // into its owning `task` row's children, with a namespaced id — but
            // only assistant-owned parts (skip the child's user prompt echo).
            if let Some(spawn) = session.and_then(|s| sub_sessions.get(s)).cloned() {
                if owned_by_assistant {
                    if let Some(mut wire) = to_wire_part(part) {
                        wire.id = format!("{spawn}:{}", wire.id);
                        ctx.upsert_child(&spawn, wire);
                        ctx.maybe_flush();
                    }
                }
                return;
            }
            if session != Some(native_id) || !owned_by_assistant {
                return;
            }
            if part.get("type").and_then(Value::as_str) == Some("step-finish")
                && part.get("reason").and_then(Value::as_str) == Some("stop")
            {
                ctx.mark_final_text_tail();
                ctx.maybe_flush();
            }
            if let Some(wire) = to_wire_part(part) {
                ctx.upsert_part(wire);
                ctx.maybe_flush();
            }
        }
        Some("message.part.delta") => {
            if props.get("field").and_then(Value::as_str) != Some("text")
                || !props
                    .get("messageID")
                    .and_then(Value::as_str)
                    .is_some_and(|id| assistant_msgs.contains(id))
            {
                return;
            }
            let session = props.get("sessionID").and_then(Value::as_str);
            let (Some(part_id), Some(delta)) = (
                props.get("partID").and_then(Value::as_str),
                props.get("delta").and_then(Value::as_str),
            ) else {
                return;
            };
            // Route a sub-agent's text delta into the owning task row's child.
            if let Some(spawn) = session.and_then(|s| sub_sessions.get(s)).cloned() {
                let child_id = format!("{spawn}:{part_id}");
                ctx.append_child_text(&spawn, &child_id, delta, || {
                    WirePart::text(child_id.clone(), "")
                });
                ctx.maybe_flush();
                return;
            }
            if session != Some(native_id) {
                return;
            }
            ctx.append_part_text(part_id, delta);
            ctx.maybe_flush();
        }
        Some("session.updated") => {
            // Adopt opencode's auto-generated titles. The creation seed arrives
            // in the first `session.updated` and the real title in a later one;
            // adopting the seed would latch it as 'generated' and permanently
            // reject the real one.
            let info = props.get("info").unwrap_or(&Value::Null);
            if info.get("id").and_then(Value::as_str) == Some(native_id) {
                if let Some(title) = info
                    .get("title")
                    .and_then(Value::as_str)
                    .filter(|t| !is_opencode_seed_title(t))
                {
                    ctx.set_title(title);
                }
            }
        }
        _ => {}
    }
}

/// Surface a prompt card and flush it so it renders immediately (before the
/// turn resumes). The card's `native_id` (the reply target) is also its
/// `WirePart` id, so the user's answer round-trips back to the right request.
fn surface_card(ctx: &mut TurnCtx, card: WirePrompt) {
    // `native_id` is always set by permission_card/question_card (opencode
    // requires the request id); the fallback id only guards a malformed payload.
    let part_id = card
        .native_id
        .clone()
        .unwrap_or_else(|| format!("prompt-{}", ctx.assistant.parts.len()));
    ctx.upsert_part(WirePart::prompt(part_id, card));
    let _ = ctx.flush();
}

/// Handle an interactive-prompt SSE event (`permission.asked` / `question.asked`)
/// for this session. Returns `true` if it consumed the event (so the caller
/// skips `handle_event`), `false` otherwise.
///
/// Permissions honor the session's policy: Auto-approve replies `always` to an
/// `ask` request, while Default surfaces a card. Explicit denies never emit an
/// approval request, so neither policy overrides them. Questions always surface — there's no
/// sensible auto-answer. A single flaky auto-reply must not lose the whole turn,
/// so on POST failure we fall back to surfacing the card rather than erroring.
async fn handle_prompt_event(
    ctx: &mut TurnCtx,
    native_id: &str,
    base: &str,
    event: &Value,
    plan_exit_calls: &HashSet<String>,
) -> Result<bool> {
    let props = event.get("properties").unwrap_or(&Value::Null);
    // Only this session's prompts (the /event stream is global across sessions).
    if props.get("sessionID").and_then(Value::as_str) != Some(native_id) {
        // Not a match — but if it *is* a prompt event for another session, still
        // report "not consumed" so handle_event ignores it too (it will, by id).
        return Ok(false);
    }
    match event.get("type").and_then(Value::as_str) {
        Some("permission.asked") => {
            let Some(card) = permission_card(props) else {
                // No request id to reply to — surface it as an error so the turn
                // isn't silently wedged waiting on an answer no one can give.
                ctx.push_error("opencode asked for a permission we couldn't parse".into());
                let _ = ctx.flush();
                return Ok(true);
            };
            // Auto-approve handles native `ask` requests; Default surfaces them.
            let auto_approve = opencode_auto_approve(ctx.permission_mode);
            match (auto_approve, card.native_id.as_deref()) {
                (true, Some(id)) => {
                    // Reply without surfacing a card — keep the turn flowing. If
                    // the reply POST fails, don't kill the turn: fall back to a
                    // card so the user can decide.
                    if let Err(err) =
                        post_permission(ctx.http(), base, native_id, id, "always").await
                    {
                        eprintln!("orx up: opencode auto-approve failed, surfacing card: {err}");
                        surface_card(ctx, card);
                    }
                }
                _ => surface_card(ctx, card),
            }
            Ok(true)
        }
        Some("question.asked") => {
            match question_card(props, plan_exit_calls) {
                Some(card) => surface_card(ctx, card),
                None => {
                    ctx.push_error("opencode asked a question we couldn't parse".into());
                    let _ = ctx.flush();
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_models_require_an_enabled_loopback_server_and_matching_model() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                axum::Router::new().route(
                    "/v1/models",
                    axum::routing::get(|| async {
                        axum::Json(json!({"data": [{"id": "loaded"}]}))
                    }),
                ),
            )
            .await
            .unwrap();
        });
        let mut config = json!({
            "enabled_providers": ["local"],
            "provider": {
                "local": {"options": {"baseURL": base}, "models": {
                    "loaded": {}, "missing": {}, "alias": {"id": "loaded"}
                }},
                "disabled": {"options": {"baseURL": base}},
                "cloud": {"options": {"baseURL": "https://example.com/v1"}}
            }
        });
        let providers = local_providers(&config);
        assert_eq!(providers.len(), 1);
        let available = available_local_models(&providers).await;
        assert_eq!(
            available,
            HashSet::from(["local/loaded".into(), "local/alias".into()])
        );
        assert!(!provider_enabled(&config, "openai"));
        server.abort();
        let _ = server.await;
        assert!(available_local_models(&providers).await.is_empty());
        config["disabled_providers"] = json!(["local"]);
        assert!(local_providers(&config).is_empty());
        for url in [
            "http://localhost:1234/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:11434/v1",
        ] {
            assert!(is_loopback_url(url));
        }
        for url in [
            "https://localhost.example/v1",
            "http://127.0.0.1@example.com/v1",
            "http://192.168.1.1/v1",
            "file:///tmp/models",
        ] {
            assert!(!is_loopback_url(url));
        }
    }

    /// Trimmed-down real `opencode models --verbose` output (1.17.15): a header
    /// line per model followed by its pretty-printed JSON. Covers the three
    /// cases that matter — a rich variants map, a *different* one on another
    /// model, and an empty one.
    const VERBOSE_SAMPLE: &str = r#"opencode/claude-fable-5
{
  "id": "claude-fable-5",
  "providerID": "opencode",
  "capabilities": {
    "reasoning": true,
    "input": { "text": true }
  },
  "variants": {
    "low": { "effort": "low" },
    "medium": { "effort": "medium" },
    "high": { "effort": "high" },
    "xhigh": { "effort": "xhigh" },
    "max": { "effort": "max" }
  }
}
opencode/gemini-3-flash
{
  "id": "gemini-3-flash",
  "providerID": "opencode",
  "variants": {
    "minimal": { "effort": "minimal" },
    "low": { "effort": "low" },
    "medium": { "effort": "medium" },
    "high": { "effort": "high" }
  }
}
opencode/glm-5
{
  "id": "glm-5",
  "providerID": "opencode",
  "variants": {}
}
"#;

    fn ids(m: &super::super::ModelInfo) -> Option<Vec<&str>> {
        m.reasoning_levels
            .as_ref()
            .map(|c| c.iter().map(|c| c.id.as_str()).collect())
    }

    #[test]
    fn public_models_require_opencode_and_explicit_zero_cost() {
        let catalog = r#"opencode/free
{"cost":{"input":0,"output":0}}
opencode/paid
{"cost":{"input":1,"output":2}}
other/free
{"cost":{"input":0,"output":0}}
opencode/unknown
{}
"#;
        let models = parse_verbose_models_filtered(catalog, true);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "opencode/free");
        assert_eq!(parse_verbose_models(catalog).len(), 4);
    }

    #[test]
    fn native_stop_marks_only_the_final_text_tail() {
        let mut ctx = TurnCtx::test_stub();
        ctx.upsert_part(WirePart::text("progress", "Reading"));
        ctx.upsert_part(WirePart::tool("tool", "read", "completed", None));
        ctx.upsert_part(WirePart::text("answer", "Done"));
        let mut messages = HashSet::from(["message".to_string()]);
        handle_event(
            &mut ctx,
            "session",
            &json!({"type":"message.part.updated","properties":{"part":{"id":"finish","messageID":"message","sessionID":"session","type":"step-finish","reason":"stop"}}}),
            &mut messages,
            &mut HashMap::new(),
        );
        assert_eq!(
            ctx.assistant.parts[0].phase,
            Some(crate::local::chat::MessagePhase::Commentary)
        );
        assert_eq!(
            ctx.assistant.parts[2].phase,
            Some(crate::local::chat::MessagePhase::FinalAnswer)
        );
        ctx.upsert_part(WirePart::text("progress", "Reading"));
        ctx.upsert_part(WirePart::text("answer", "Done."));
        ctx.mark_final_text_tail();
        assert_eq!(
            ctx.assistant.parts[0].phase,
            Some(crate::local::chat::MessagePhase::Commentary)
        );
        assert_eq!(
            ctx.assistant.parts[2].phase,
            Some(crate::local::chat::MessagePhase::FinalAnswer)
        );
    }

    /// The core of issue #123 for opencode: variants are genuinely per-model,
    /// so each model gets its own list rather than a hard-coded union.
    #[test]
    fn plain_catalog_keeps_configured_local_labels() {
        let mut models = vec![
            ModelInfo::new("local/mlx/qwen"),
            ModelInfo::new("cloud/claude").with_label(Some("Claude"), None),
        ];
        apply_configured_labels(
            &mut models,
            &json!({"provider":{"local":{"models":{"mlx/qwen":{"name":"Qwen · LM Studio (local)"}}}}}),
        );
        assert_eq!(
            models[0].display_name.as_deref(),
            Some("Qwen · LM Studio (local)")
        );
        assert_eq!(models[1].display_name.as_deref(), Some("Claude"));
    }

    #[test]
    fn verbose_models_parse_per_model_variants() {
        let models = parse_verbose_models(VERBOSE_SAMPLE);
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            [
                "opencode/claude-fable-5",
                "opencode/gemini-3-flash",
                "opencode/glm-5"
            ]
        );
        // Nested `{ … }` inside the variants map must not end the block early.
        assert_eq!(
            ids(&models[0]),
            Some(vec!["default", "low", "medium", "high", "xhigh", "max"])
        );
        // A different model, a genuinely different set (note `minimal`, and no
        // `xhigh`/`max`) — the whole point of being model-aware.
        assert_eq!(
            ids(&models[1]),
            Some(vec!["default", "minimal", "low", "medium", "high"])
        );
    }

    /// Regression: `serde_json`'s default map is a `BTreeMap`, so raw key order
    /// is alphabetical (`high, low, max, medium, xhigh`) — a meaningless ramp
    /// in the picker. Variants must come out weakest → strongest regardless of
    /// the order they appear in the JSON.
    #[test]
    fn variants_are_ordered_weakest_to_strongest() {
        let model = serde_json::json!({
            "variants": { "max": {}, "low": {}, "xhigh": {}, "high": {}, "medium": {} }
        });
        assert_eq!(
            variant_ids(&model).unwrap(),
            ["low", "medium", "high", "xhigh", "max"]
        );
        // Unknown ids still survive, sorted after the known ramp.
        let odd = serde_json::json!({ "variants": { "zzz": {}, "high": {}, "aaa": {} } });
        assert_eq!(variant_ids(&odd).unwrap(), ["high", "aaa", "zzz"]);
    }

    /// A native variant literally named `default` must not produce a second
    /// row identical to the sentinel — that row would read as "no override" and
    /// make the real variant unselectable.
    #[test]
    fn a_native_default_variant_does_not_duplicate_the_sentinel() {
        let out = "prov/a\n{\n  \"variants\": { \"default\": {}, \"high\": {} }\n}\n";
        let models = parse_verbose_models(out);
        assert_eq!(ids(&models[0]), Some(vec!["default", "high"]));
    }

    /// An empty `variants` map means "checked, none supported" → an empty list,
    /// which hides the picker. It must NOT be `None`, which would fall back to
    /// the harness-wide list.
    #[test]
    fn empty_variants_map_hides_the_picker() {
        let models = parse_verbose_models(VERBOSE_SAMPLE);
        assert_eq!(ids(&models[2]), Some(vec![]));
        assert!(models[2].reasoning_levels.is_some());
    }

    /// Garbage or a `--verbose` flag the installed CLI doesn't support yields
    /// no models, which sends `opencode_models` to the plain-list fallback.
    #[test]
    fn unparseable_verbose_output_yields_nothing() {
        assert!(parse_verbose_models("").is_empty());
        assert!(parse_verbose_models("error: unknown flag --verbose").is_empty());
        // Header with no JSON block is skipped, not half-parsed.
        assert!(parse_verbose_models("opencode/foo\nnot json\n").is_empty());
    }

    /// The plain-list fallback still yields models, just without variants.
    #[test]
    fn plain_model_lines_have_no_variants() {
        let list: Vec<_> = model_id_lines("opencode/a\n\n  github-copilot/b  \njunk\n").collect();
        assert_eq!(list, ["opencode/a", "github-copilot/b"]);
        assert!(super::super::ModelInfo::new("opencode/a")
            .reasoning_levels
            .is_none());
    }

    /// A `{` inside a JSON string value must not desynchronize the brace
    /// counter. Before this was handled, one such brace consumed the rest of
    /// the output and every later model vanished — silently, since a partial
    /// parse is non-empty and so never reaches the plain-list fallback.
    #[test]
    fn brace_inside_a_string_does_not_swallow_later_models() {
        let out = concat!(
            "prov/a\n{\n  \"name\": \"Weird { name\",\n  \"variants\": { \"high\": {} }\n}\n",
            "prov/b\n{\n  \"name\": \"esc \\\" and } brace\",\n  \"variants\": {}\n}\n",
            "prov/c\n{\n  \"variants\": { \"low\": {}, \"max\": {} }\n}\n",
        );
        let models = parse_verbose_models(out);
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["prov/a", "prov/b", "prov/c"]
        );
        assert_eq!(ids(&models[0]), Some(vec!["default", "high"]));
        assert_eq!(ids(&models[1]), Some(vec![]));
        assert_eq!(ids(&models[2]), Some(vec!["default", "low", "max"]));
    }

    /// Only the sentinel is withheld. An unrecognized id is forwarded, because
    /// `variant_ids` deliberately keeps unknown variants so a new one still
    /// reaches the picker — offering it and then dropping it here would ignore
    /// the user's selection.
    #[test]
    fn variant_is_sent_unless_it_is_the_default_sentinel() {
        assert_eq!(opencode_variant(Some("high")), Some("high"));
        assert_eq!(opencode_variant(Some("minimal")), Some("minimal"));
        assert_eq!(opencode_variant(Some("none")), Some("none"));
        assert_eq!(opencode_variant(Some("brand-new")), Some("brand-new"));
        assert_eq!(opencode_variant(Some(REASONING_DEFAULT_ID)), None);
        assert_eq!(opencode_variant(None), None);
    }

    /// Every variant id detection advertises must survive the mapper — the
    /// picker can never offer a value `run_turn` would silently drop. Includes
    /// an unknown id, which is exactly the case a mapper-side allowlist broke.
    #[test]
    fn advertised_variants_all_map_back() {
        let unknown = "prov/x\n{\n  \"variants\": { \"high\": {}, \"turbo\": {} }\n}\n";
        for model in parse_verbose_models(VERBOSE_SAMPLE)
            .into_iter()
            .chain(parse_verbose_models(unknown))
        {
            for choice in model.reasoning_levels.into_iter().flatten() {
                if choice.id == REASONING_DEFAULT_ID {
                    continue;
                }
                assert_eq!(
                    opencode_variant(Some(&choice.id)),
                    Some(choice.id.as_str()),
                    "{} advertises {} but the mapper drops it",
                    model.id,
                    choice.id
                );
            }
        }
    }

    #[test]
    fn plan_mode_uses_the_plan_agent_others_build() {
        assert_eq!(opencode_agent(true), "plan");
        assert_eq!(opencode_agent(false), "build");
    }

    #[test]
    fn plan_and_permissions_form_four_independent_combinations() {
        for (plan_mode, permission_mode, agent, auto_approve) in [
            (false, Some(PermissionMode::Ask), "build", false),
            (false, Some(PermissionMode::Auto), "build", true),
            (true, Some(PermissionMode::Ask), "plan", false),
            (true, Some(PermissionMode::Auto), "plan", true),
        ] {
            assert_eq!(opencode_agent(plan_mode), agent);
            assert_eq!(opencode_auto_approve(permission_mode), auto_approve);
        }
    }

    // `properties` payloads shaped exactly like the live `permission.asked` /
    // `question.asked` events (verified against opencode serve). These pin the
    // field names the parsers read — the kind that silently yields a `None` card
    // at runtime if opencode ever renames one.
    #[test]
    fn permission_card_reads_id_permission_metadata() {
        let props = json!({
            "id": "per_abc123",
            "sessionID": "ses_x",
            "permission": "bash",
            "patterns": [],
            "metadata": { "command": "orx runs r1" },
            "always": [],
            "tool": { "messageID": "m1", "callID": "c1" }
        });
        let card = permission_card(&props).expect("should parse");
        assert_eq!(card.kind, "permission");
        assert_eq!(card.tool.as_deref(), Some("bash"));
        assert_eq!(card.native_id.as_deref(), Some("per_abc123")); // the reply target
        assert_eq!(
            card.tool_input
                .as_ref()
                .and_then(|m| m.get("command"))
                .and_then(|c| c.as_str()),
            Some("orx runs r1")
        );
        // No id → no reply target → no card.
        assert!(permission_card(&json!({ "permission": "bash" })).is_none());
    }

    #[test]
    fn question_card_reads_first_question_and_opencode_multiple_field() {
        let props = json!({
            "id": "que_xyz",
            "sessionID": "ses_x",
            "questions": [{
                "question": "Which backend?",
                "header": "Backend",
                "options": [
                    { "label": "modal", "description": "per-second" },
                    { "label": "k8s", "description": "your cluster" }
                ],
                "multiple": true
            }]
        });
        let card = question_card(&props, &HashSet::new()).expect("should parse");
        assert_eq!(card.kind, "question");
        assert_eq!(card.native_id.as_deref(), Some("que_xyz"));
        assert_eq!(card.question.as_deref(), Some("Which backend?"));
        assert_eq!(card.header.as_deref(), Some("Backend"));
        assert_eq!(card.options.len(), 2);
        assert_eq!(card.options[0].label, "modal");
        assert_eq!(card.options[0].description.as_deref(), Some("per-second"));
        // opencode's field is `multiple`, NOT Claude's `multiSelect`.
        assert!(card.multi_select);
        // A `multiSelect` (Claude's name) is NOT read → defaults to false.
        let claude_shaped = json!({
            "id": "que_1",
            "questions": [{ "question": "q", "header": "h", "options": [], "multiSelect": true }]
        });
        assert!(
            !question_card(&claude_shaped, &HashSet::new())
                .unwrap()
                .multi_select
        );
        // No questions → no card.
        assert!(question_card(&json!({ "id": "que_1" }), &HashSet::new()).is_none());
    }

    #[test]
    fn question_card_recognizes_plan_exit_by_tool_call_id() {
        let props = json!({
            "id": "que_exit",
            "tool": { "messageID": "m1", "callID": "call_plan" },
            "questions": [{
                "question": "Switch to build?",
                "header": "Build Agent",
                "options": [{ "label": "Yes" }, { "label": "No" }],
                "multiple": false
            }]
        });
        let calls = HashSet::from(["call_plan".to_string()]);
        assert!(question_card(&props, &calls).unwrap().plan_exit);
        assert!(!question_card(&props, &HashSet::new()).unwrap().plan_exit);
    }

    #[test]
    fn native_plan_exit_yes_leaves_plan_no_keeps_it() {
        let mut prompt = WirePrompt {
            plan_exit: true,
            ..Default::default()
        };
        let response = |choice: &str| PromptAnswer {
            session_id: "session".into(),
            prompt_id: "question".into(),
            approve: true,
            resume_mode: None,
            answers: vec![choice.into()],
            note: None,
            annotations: Vec::new(),
        };
        assert_eq!(plan_exit_transition(&prompt, &response("Yes")), Some(false));
        assert_eq!(plan_exit_transition(&prompt, &response("No")), Some(true));
        prompt.plan_exit = false;
        assert_eq!(plan_exit_transition(&prompt, &response("Yes")), None);
    }

    #[test]
    fn native_step_tokens_restore_inclusive_counters() {
        let fixture: Value =
            serde_json::from_str(include_str!("fixtures/opencode-mock-usage.json")).unwrap();
        let usage = opencode_native_usage(&fixture[0]["tokens"]);
        usage.validate().unwrap();
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(usage.reasoning_tokens, Some(5));
        assert_eq!(usage.total(), Some(120));
    }

    #[test]
    fn message_updated_reports_summed_tokens_without_window() {
        let mut ctx = TurnCtx::test_stub();
        let mut msgs = HashSet::new();
        let event = json!({
            "type": "message.updated",
            "properties": { "info": {
                "id": "msg_1",
                "sessionID": "ses_x",
                "role": "assistant",
                "tokens": { "input": 1200, "output": 340, "reasoning": 50, "cache": { "read": 8000, "write": 200 } }
            }}
        });
        handle_event(&mut ctx, "ses_x", &event, &mut msgs, &mut HashMap::new());
        let usage = ctx.context_usage.expect("usage reported");
        assert_eq!(usage.used_tokens, 1200 + 340 + 50 + 8000 + 200);
        assert_eq!(usage.context_window, None);
    }

    #[test]
    fn message_updated_without_tokens_reports_nothing() {
        let mut ctx = TurnCtx::test_stub();
        let mut msgs = HashSet::new();
        // Early message.updated: assistant role, but no tokens yet.
        let no_tokens = json!({
            "type": "message.updated",
            "properties": { "info": { "id": "msg_1", "sessionID": "ses_x", "role": "assistant" }}
        });
        handle_event(
            &mut ctx,
            "ses_x",
            &no_tokens,
            &mut msgs,
            &mut HashMap::new(),
        );
        assert!(ctx.context_usage.is_none());
        // All-zero placeholder tokens must also be ignored.
        let zero_tokens = json!({
            "type": "message.updated",
            "properties": { "info": { "id": "msg_1", "sessionID": "ses_x", "role": "assistant",
                "tokens": { "input": 0, "output": 0, "reasoning": 0, "cache": { "read": 0, "write": 0 } }}}
        });
        handle_event(
            &mut ctx,
            "ses_x",
            &zero_tokens,
            &mut msgs,
            &mut HashMap::new(),
        );
        assert!(ctx.context_usage.is_none());
    }

    #[test]
    fn compaction_recovery_does_not_surface_a_terminal_error_or_summary() {
        let mut ctx = TurnCtx::test_stub();
        let mut messages = HashSet::new();
        let mut sessions = HashMap::new();
        for event in [
            json!({"type":"session.error","properties":{"sessionID":"ses_x","error":{"name":"ContextOverflowError","data":{"message":"Too many tokens"}}}}),
            json!({"type":"message.updated","properties":{"info":{"id":"summary","sessionID":"ses_x","role":"assistant","summary":true}}}),
            json!({"type":"message.part.updated","properties":{"part":{"id":"summary_text","messageID":"summary","sessionID":"ses_x","type":"text","text":"Internal summary"}}}),
            json!({"type":"message.part.delta","properties":{"sessionID":"ses_x","messageID":"summary","partID":"summary_text","field":"text","delta":"hidden"}}),
            json!({"type":"message.updated","properties":{"info":{"id":"answer","sessionID":"ses_x","role":"assistant"}}}),
            json!({"type":"message.part.updated","properties":{"part":{"id":"answer_text","messageID":"answer","sessionID":"ses_x","type":"text","text":"Run finished"}}}),
        ] {
            handle_event(&mut ctx, "ses_x", &event, &mut messages, &mut sessions);
        }
        assert_eq!(ctx.delivery_state(), DeliveryState::Accepted);
        assert_eq!(ctx.assistant.parts.len(), 1);
        assert_eq!(ctx.assistant.parts[0].text.as_deref(), Some("Run finished"));
        assert_eq!(
            opencode_response_error(&json!({"info":{"role":"assistant","finish":"stop"}})),
            None
        );
        assert_eq!(
            opencode_response_error(
                &json!({"info":{"error":{"name":"APIError","data":{"message":"Invalid API key"}}}})
            ),
            Some("Invalid API key")
        );
        assert!(opencode_response_error(&json!({"info":{"summary":true}})).is_some());
    }

    #[test]
    fn stale_final_response_is_rejected() {
        assert!(opencode_response_is_current(
            &json!({"info":{"time":{"created":100}}}),
            100
        ));
        assert!(!opencode_response_is_current(
            &json!({"info":{"time":{"created":99}}}),
            100
        ));
    }

    #[test]
    fn auth_rejections_are_recognized_but_other_failures_are_not() {
        assert!(is_auth_rejection(
            "Error: API key not valid. Please pass a valid API key."
        ));
        assert!(is_auth_rejection("Incorrect API key provided: sk-abc"));
        assert!(is_auth_rejection("{\"type\":\"authentication_error\"}"));
        assert!(is_auth_rejection("HTTP 401 Unauthorized"));
        assert!(!is_auth_rejection("Error: fetch failed (ENOTFOUND)"));
        assert!(!is_auth_rejection(
            "    at plugin.ts:401:12\ncontext: 1401 tokens"
        ));
        assert!(!is_auth_rejection("ok"));
        assert!(!is_auth_rejection("model not found: google/nope"));
    }

    #[test]
    fn model_provider_is_the_prefix() {
        assert_eq!(model_provider("google/gemini-2.5-flash"), "google");
        assert_eq!(model_provider("opencode/big-pickle"), "opencode");
        assert_eq!(model_provider("bare"), "bare");
    }

    #[test]
    fn seed_title_is_recognized_but_real_titles_pass() {
        // The exact shape opencode stamps at session creation.
        assert!(is_opencode_seed_title(
            "New session - 2026-07-09T23:50:40.501Z"
        ));
        assert!(is_opencode_seed_title(
            "  New session - 2026-07-09T23:50:40.501Z"
        ));
        // What the summarizer actually produces — must reach `set_title`.
        assert!(!is_opencode_seed_title("Fix the login redirect"));
        assert!(!is_opencode_seed_title("New session handling in the store"));
        assert!(!is_opencode_seed_title(""));
    }

    #[test]
    fn only_active_session_statuses_confirm_turn_acceptance() {
        let mut ctx = TurnCtx::test_stub();
        ctx.mark_delivery(DeliveryState::Unknown);
        let mut messages = HashSet::new();
        let mut sessions = HashMap::new();
        handle_event(
            &mut ctx,
            "ses_x",
            &json!({"type":"session.status","properties":{"sessionID":"ses_x","status":{"type":"idle"}}}),
            &mut messages,
            &mut sessions,
        );
        assert_eq!(ctx.delivery_state(), DeliveryState::Unknown);
        handle_event(
            &mut ctx,
            "ses_x",
            &json!({"type":"session.status","properties":{"sessionID":"ses_x","status":{"type":"retry","attempt":1}}}),
            &mut messages,
            &mut sessions,
        );
        assert_eq!(ctx.delivery_state(), DeliveryState::Accepted);
    }

    /// A `task` tool spawns a child session (announced via `session.created` with
    /// `parentID` = our session); the sub-agent's parts stream into the task
    /// row's `children`, not the top-level transcript.
    #[test]
    fn subagent_parts_stream_into_the_task_row_children() {
        let mut ctx = TurnCtx::test_stub();
        let mut msgs: HashSet<String> = HashSet::new();
        let mut subs: HashMap<String, String> = HashMap::new();
        // The main assistant message + its `task` tool call (top-level).
        handle_event(
            &mut ctx,
            "ses_main",
            &json!({"type":"message.updated","properties":{"info":{"id":"msg_1","sessionID":"ses_main","role":"assistant"}}}),
            &mut msgs,
            &mut subs,
        );
        handle_event(
            &mut ctx,
            "ses_main",
            &json!({"type":"message.part.updated","properties":{"part":{
                "id":"prt_task","type":"tool","tool":"task","sessionID":"ses_main","messageID":"msg_1",
                "state":{"status":"running","input":{"description":"analyze"}}}}}),
            &mut msgs,
            &mut subs,
        );
        // opencode announces the spawned child session (parentID = our session).
        handle_event(
            &mut ctx,
            "ses_main",
            &json!({"type":"session.created","properties":{"info":{"id":"ses_child","parentID":"ses_main"}}}),
            &mut msgs,
            &mut subs,
        );
        assert_eq!(subs.get("ses_child").map(String::as_str), Some("prt_task"));
        handle_event(
            &mut ctx,
            "ses_main",
            &json!({"type":"session.created","properties":{"info":{"id":"ses_grandchild","parentID":"ses_child"}}}),
            &mut msgs,
            &mut subs,
        );
        handle_event(
            &mut ctx,
            "ses_main",
            &json!({"type":"message.updated","properties":{"info":{"id":"msg_grandchild","sessionID":"ses_grandchild","role":"assistant","modelID":"child-model","providerID":"fixture"}}}),
            &mut msgs,
            &mut subs,
        );
        // The child session's assistant message + a tool part → nests under task.
        handle_event(
            &mut ctx,
            "ses_main",
            &json!({"type":"message.updated","properties":{"info":{"id":"msg_c","sessionID":"ses_child","role":"assistant"}}}),
            &mut msgs,
            &mut subs,
        );
        handle_event(
            &mut ctx,
            "ses_main",
            &json!({"type":"message.part.updated","properties":{"part":{
                "id":"prt_bash","type":"tool","tool":"bash","sessionID":"ses_child","messageID":"msg_c",
                "state":{"status":"completed","input":{"command":"ls"},"output":"a.rs"}}}}),
            &mut msgs,
            &mut subs,
        );
        // Only the task row is top-level; the sub bash nested under it (namespaced).
        assert_eq!(ctx.assistant.parts.len(), 1, "{:?}", ctx.assistant.parts);
        let task = &ctx.assistant.parts[0];
        assert_eq!(task.id, "prt_task");
        assert_eq!(task.tool.as_deref(), Some("task"));
        let bash = task
            .children
            .iter()
            .find(|p| p.id == "prt_task:prt_bash")
            .expect("sub bash nested under the task row");
        assert_eq!(bash.state.as_ref().unwrap().output.as_deref(), Some("a.rs"));

        // The turn-end merge re-upserts the main message's parts (incl. the task
        // row, rebuilt with empty children) authoritatively. It MUST preserve the
        // accrued children — a plain upsert would wipe the sub-agent transcript.
        let final_task = to_wire_part(&json!({
            "id":"prt_task","type":"tool","tool":"task","sessionID":"ses_main","messageID":"msg_1",
            "state":{"status":"completed","input":{"description":"analyze"},"output":"done"}
        }))
        .unwrap();
        assert!(
            final_task.children.is_empty(),
            "rebuilt part has no children"
        );
        ctx.upsert_part_preserving_children(final_task);
        let task = &ctx.assistant.parts[0];
        assert_eq!(task.state.as_ref().unwrap().status, "completed");
        assert_eq!(task.children.len(), 1, "children survive the final merge");
    }

    /// Native evidence as the store keeps it: an identity-only write never erases counters.
    #[derive(Default)]
    pub(super) struct Recorded {
        pub(super) samples:
            Mutex<HashMap<String, (crate::store::Attribution, crate::store::TokenUsage, bool)>>,
        pub(super) invokers: Mutex<HashMap<String, String>>,
        pub(super) evidence: Mutex<Vec<WirePart>>,
    }

    impl UsageSink for Recorded {
        fn sample(
            &self,
            id: &str,
            attribution: crate::store::Attribution,
            usage: crate::store::TokenUsage,
            complete: bool,
        ) {
            let mut samples = self.samples.lock().unwrap();
            let entry = samples.entry(id.to_string()).or_insert((
                attribution.clone(),
                usage.clone(),
                complete,
            ));
            entry.0 = attribution;
            if usage != crate::store::TokenUsage::default() {
                (entry.1, entry.2) = (usage, complete);
            }
        }

        fn invoker(&self, part_id: &str, model: &str, _: Option<&str>) {
            self.invokers
                .lock()
                .unwrap()
                .insert(part_id.to_string(), model.to_string());
        }

        fn tool_evidence(&self, part: &WirePart) {
            self.evidence.lock().unwrap().push(part.clone());
        }
    }

    /// Records into execution `exec` of a real store, as [`ExecutionSink`] does.
    pub(super) struct StoreSink(pub(super) Mutex<crate::store::Store>);

    impl UsageSink for StoreSink {
        fn sample(
            &self,
            id: &str,
            attribution: crate::store::Attribution,
            usage: crate::store::TokenUsage,
            complete: bool,
        ) {
            self.0
                .lock()
                .unwrap()
                .record_attributed_sample("exec", id, "opencode", &attribution, &usage, complete)
                .unwrap();
        }
        fn invoker(&self, _: &str, _: &str, _: Option<&str>) {}
    }

    /// An open execution `exec` in a fresh store, and a connection to read its ledger rows.
    pub(super) fn usage_store() -> (crate::store::Store, rusqlite::Connection) {
        let dir = std::env::temp_dir().join(format!("orx-oc-usage-{}", uuid::Uuid::new_v4()));
        let store = crate::store::Store::open_at(dir.clone()).unwrap();
        let db = rusqlite::Connection::open(dir.join("orx.db")).unwrap();
        db.execute_batch("INSERT INTO chat_turns (id, session_id, assistant_message_id, client_turn_id, request_hash, prepared_input, settings_json, state, delivery_state, created_at, updated_at) VALUES ('turn', 'session', 'message', 'c', 'h', '', '{}', 'running', 'accepted', 1, 1);").unwrap();
        store
            .begin_usage_execution("exec", "turn", "opencode")
            .unwrap();
        (store, db)
    }

    /// Each ledger row of `exec`: sample id → (attribution, usage), as reports read them.
    pub(super) fn ledger(db: &rusqlite::Connection) -> Vec<(String, String, String)> {
        db.prepare("SELECT sample_id, attribution_json, usage_json FROM chat_usage_samples WHERE execution_id = 'exec' ORDER BY sample_id")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    /// A scope persisted before submission is never held, so a crash leaves it adoptable; a turn that
    /// captured everything clears it.
    #[test]
    fn a_pre_submission_scope_is_adoptable_until_cleared() {
        let (store, _db) = usage_store();
        let scope = json!({"native":"ses_main","startedAt":10,"roots":[],"prompt":"msg_p"});
        store
            .set_native_scope("exec", BACKGROUND_SCOPE, &scope)
            .unwrap();
        assert_eq!(
            store.native_scopes(BACKGROUND_SCOPE).unwrap(),
            [("exec".into(), BACKGROUND_SCOPE.into(), scope, true)]
        );
        store.clear_native_scope("exec", BACKGROUND_SCOPE).unwrap();
        assert!(store.native_scopes(BACKGROUND_SCOPE).unwrap().is_empty());
    }

    /// Writes V1 `/session/{id}/message` entries as native `message` and `part` rows.
    fn insert_v1(db: &rusqlite::Connection, session: &str, messages: Vec<Value>) {
        for mut message in messages {
            let info = message["info"].as_object_mut().unwrap();
            let id = info.remove("id").unwrap();
            info.remove("sessionID");
            let created = info["time"]["created"].as_i64();
            let data = Value::Object(info.clone()).to_string();
            db.execute(
                "INSERT INTO message VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![id.as_str(), session, created, data],
            )
            .unwrap();
            for mut part in message["parts"].as_array().unwrap().clone() {
                let part = part.as_object_mut().unwrap();
                let part_id = part.remove("id").unwrap();
                part.remove("messageID");
                part.remove("sessionID");
                let data = Value::Object(part.clone()).to_string();
                db.execute(
                    "INSERT INTO part VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![part_id.as_str(), id.as_str(), session, data],
                )
                .unwrap();
            }
        }
    }

    /// A V1 database holding a prompt whose run stopped after a `tool-calls` step.
    fn v1_unfinished_run(prompt: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("orx-oc-v1db-{}.db", uuid::Uuid::new_v4()));
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute_batch("CREATE TABLE session (id TEXT PRIMARY KEY); CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT); CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT); INSERT INTO session VALUES ('ses_main');").unwrap();
        let prompt = json!({"info":{"id":prompt,"sessionID":"ses_main","role":"user",
            "time":{"created":12}},"parts":[]});
        insert_v1(
            &db,
            "ses_main",
            vec![prompt, v1_step("msg_a", "tool-calls")],
        );
        path
    }

    fn v1_step(id: &str, finish: &str) -> Value {
        json!({"info":{"id":id,"sessionID":"ses_main","role":"assistant","modelID":"m",
            "providerID":"p","finish":finish,"time":{"created":13,"completed":14}},
            "parts":[{"id":format!("{id}_1"),"messageID":id,"sessionID":"ses_main","type":"step-start"}]})
    }

    /// After a restart the turn's surviving V1 server is reconnected at its persisted endpoint: while
    /// it runs the prompt's tree, recovery keeps reading past the old grace, then settles unmarked.
    #[tokio::test]
    async fn v1_recovery_reads_the_live_original_server_past_the_grace() {
        use axum::{routing::get, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        const PROMPT: &str = "msg_0f4a81d5e000-orxabcdefghij";
        let path = v1_unfinished_run(PROMPT);
        // Each poll probes and then reads the status; the run ends well after the old grace.
        let late = 4 * DELIVERY_GRACE_POLLS;
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let app = Router::new().route(
            "/session/status",
            get({
                let (calls, path) = (calls.clone(), path.clone());
                move || {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    if call == late {
                        let db = rusqlite::Connection::open(&path).unwrap();
                        insert_v1(&db, "ses_main", vec![v1_step("msg_b", "stop")]);
                    }
                    async move {
                        Json(if call < late {
                            json!({"ses_main":{"type":"busy"}})
                        } else {
                            json!({})
                        })
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let recorded = Recorded::default();
        let scope = json!({"native":"ses_main","startedAt":10,"roots":[],"prompt":PROMPT,
            "endpoint":base});
        settle_stored(&recorded, &scope, &path, || persisted_owner(&scope)).await;
        server.abort();
        let mut ids: Vec<_> = recorded.samples.into_inner().unwrap().into_keys().collect();
        ids.sort();
        assert_eq!(ids, ["msg_a_1", "msg_b_1"]);
        assert!(calls.load(Ordering::SeqCst) > late);
    }

    /// Without a live original server nothing waits on other database holders: a gone server (or a
    /// scope with no recorded owner) leaves a final snapshot.
    #[tokio::test]
    async fn v1_recovery_never_waits_without_a_live_original_server() {
        const PROMPT: &str = "msg_0f4a81d5e000-orxabcdefghij";
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let gone = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        for endpoint in [Some(gone), None] {
            let path = v1_unfinished_run(PROMPT);
            // Another holder keeps the database open throughout.
            let _holder = rusqlite::Connection::open(&path).unwrap();
            let recorded = Recorded::default();
            let scope = json!({"native":"ses_main","startedAt":10,"roots":[],"prompt":PROMPT,
                "endpoint":endpoint});
            tokio::time::timeout(
                Duration::from_secs(5),
                settle_stored(&recorded, &scope, &path, || persisted_owner(&scope)),
            )
            .await
            .expect("no live owner, no wait");
            let samples = recorded.samples.into_inner().unwrap();
            assert!(samples.contains_key("msg_a_1"));
            assert!(
                !samples.contains_key("opencode-root:unrecoverable"),
                "{scope}"
            );
        }
    }

    /// A cancellation whose native read failed keeps its persisted scope and holds the execution, so
    /// the interrupt's finalize waits for the retry instead of closing without the missing usage.
    #[test]
    fn a_failed_interrupt_capture_holds_its_scope_for_retry() {
        let (store, db) = usage_store();
        let scope = json!({"native":"ses_main","startedAt":10,"roots":[],"prompt":"msg_p"});
        store
            .set_native_scope("exec", BACKGROUND_SCOPE, &scope)
            .unwrap();
        assert_eq!(hold_for_retry(&store, "exec").unwrap(), Some(scope.clone()));
        store.finalize_turn_usage("turn", "cancelled").unwrap();
        let (outcome, pending): (Option<String>, Option<String>) = db
            .query_row(
                "SELECT outcome, pending_outcome FROM chat_usage_executions",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!((outcome, pending.as_deref()), (None, Some("cancelled")));
        assert_eq!(store.native_scopes(BACKGROUND_SCOPE).unwrap()[0].2, scope);
    }

    /// V1 crash recovery reads the native database, starting no server: the prompt's run (across
    /// native compaction) and its child only; an unreadable database retries, replays add nothing.
    #[tokio::test]
    async fn v1_restart_recovers_the_prompt_run_from_the_native_database() {
        const PROMPT: &str = "msg_0f4a81d5e000-orxabcdefghij";
        let step = |session: &str, id: &str, model: &str, created: i64, tool: Option<Value>| {
            let mut parts = vec![
                json!({"id":format!("{id}_1"),"messageID":id,"sessionID":session,"type":"step-start"}),
            ];
            parts.extend(tool);
            parts.push(json!({"id":format!("{id}_9"),"messageID":id,"sessionID":session,"type":"step-finish",
                "tokens":{"input":7,"output":3,"reasoning":0,"cache":{"read":0,"write":0}}}));
            json!({"info":{"id":id,"sessionID":session,"role":"assistant","modelID":model,"providerID":"p",
                "time":{"created":created}},"parts":parts})
        };
        let user = |id: &str, created: i64| json!({"info":{"id":id,"sessionID":"ses_main","role":"user","time":{"created":created}},"parts":[]});
        let task = json!({"id":"msg_root_5","type":"tool","tool":"task","state":{"status":"completed",
            "input":{},"output":"","metadata":{"sessionId":"ses_child","background":false}}});
        let main = vec![
            user("msg_0f4a00000000-orxolderolder", 5),
            step("ses_main", "msg_old", "old-model", 11, None),
            user(PROMPT, 12),
            step("ses_main", "msg_root", "root-model", 13, Some(task)),
            user("msg_0f4a81d5e0011NativeCompac", 14),
            step("ses_main", "msg_compacted", "compact-model", 15, None),
            user("msg_0f4a81d5f000-orxlaterlater", 20),
            step("ses_main", "msg_later", "later-model", 21, None),
        ];
        let child = vec![step("ses_child", "msg_child", "child-model", 13, None)];
        let dir = std::env::temp_dir().join(format!("orx-oc-v1db-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let (staged, database) = (dir.join("staged.db"), dir.join("opencode.db"));
        let db = rusqlite::Connection::open(&staged).unwrap();
        db.execute_batch("CREATE TABLE session (id TEXT PRIMARY KEY); CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT); CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT);").unwrap();
        for (session, messages) in [("ses_main", main), ("ses_child", child)] {
            db.execute("INSERT INTO session VALUES (?1)", [session])
                .unwrap();
            insert_v1(&db, session, messages);
        }
        drop(db);
        // The database is unreadable until it appears.
        let appear = tokio::spawn({
            let database = database.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                std::fs::rename(staged, database).unwrap();
            }
        });
        let (store, db) = usage_store();
        let sink = StoreSink(Mutex::new(store));
        let scope = json!({"native":"ses_main","startedAt":10,"roots":[],"prompt":PROMPT});
        let owner = || async { Owner::Gone };
        settle_stored(&sink, &scope, &database, owner).await;
        appear.await.unwrap();
        let first = ledger(&db);
        settle_stored(&sink, &scope, &database, owner).await;
        assert_eq!(ledger(&db), first);
        let models: Vec<(String, crate::store::Attribution)> = first
            .into_iter()
            .map(|(id, attribution, _)| (id, serde_json::from_str(&attribution).unwrap()))
            .collect();
        let exact = |model: &str| crate::store::Attribution::Exact {
            model: model.into(),
            provider: Some("p".into()),
        };
        assert_eq!(
            models,
            [
                ("msg_child_1".to_string(), exact("child-model")),
                ("msg_compacted_1".to_string(), exact("compact-model")),
                ("msg_root_1".to_string(), exact("root-model")),
            ]
        );
    }

    /// A hold a dead process left (background subagents still to read) is adopted at startup, so
    /// generic recovery cannot close it on that process's last snapshot before its native history
    /// is captured; its scope carries what recovery needs. History gone for good closes it with the
    /// captured usage beside an explicit unresolved request.
    #[tokio::test]
    async fn orphaned_background_holds_are_adopted_not_closed_at_startup() {
        let dir = std::env::temp_dir().join(format!("orx-oc-adopt-{}", uuid::Uuid::new_v4()));
        let store = crate::store::Store::open_at(dir.clone()).unwrap();
        let db = rusqlite::Connection::open(dir.join("orx.db")).unwrap();
        db.execute_batch("INSERT INTO chat_turns (id, session_id, assistant_message_id, client_turn_id, request_hash, prepared_input, settings_json, state, delivery_state, created_at, updated_at) VALUES ('turn', 'session', 'message', 'c', 'h', '', '{}', 'completed', 'accepted', 1, 1);").unwrap();
        store
            .begin_usage_execution("exec", "turn", "opencode")
            .unwrap();
        store.hold_usage_execution("exec").unwrap();
        store.finalize_turn_usage("turn", "done").unwrap();
        let scope = json!({"session":"session","native":"ses_main","startedAt":100,"roots":[["ses_bg","prt_task"]]});
        store
            .set_native_scope("exec", BACKGROUND_SCOPE, &scope)
            .unwrap();
        db.execute(
            "UPDATE chat_usage_executions SET held_by = 'dead-process'",
            [],
        )
        .unwrap();
        adopt_orphaned_background(&store).unwrap();
        store.recover_terminal_usage().unwrap();
        let outcome: Option<String> = db
            .query_row("SELECT outcome FROM chat_usage_executions", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(outcome, None, "kept open for native recovery");
        let adopted = std::mem::take(&mut *ADOPTED.lock().unwrap());
        assert_eq!(adopted, [("exec".to_string(), scope.clone())]);
        // A watch still retrying when the process dies is adopted again at the next start.
        db.execute(
            "UPDATE chat_usage_executions SET held_by = 'dead-process'",
            [],
        )
        .unwrap();
        adopt_orphaned_background(&store).unwrap();
        store.recover_terminal_usage().unwrap();
        let adopted = std::mem::take(&mut *ADOPTED.lock().unwrap());
        assert_eq!(adopted, [("exec".to_string(), scope.clone())]);

        let partial = crate::store::TokenUsage {
            input_tokens: Some(5),
            output_tokens: Some(2),
            ..Default::default()
        };
        let sink = StoreSink(Mutex::new(store));
        let exact = crate::store::Attribution::Exact {
            model: "bg-model".into(),
            provider: Some("p".into()),
        };
        sink.sample("prt_bg_s", exact, partial, true);
        settle_watch(&sink, &scope, |_| {
            std::future::ready(Err(anyhow!("OpenCode session no longer exists")))
        })
        .await;
        let store = sink.0.into_inner().unwrap();
        store.release_usage_execution("exec").unwrap();
        let outcome: Option<String> = db
            .query_row("SELECT outcome FROM chat_usage_executions", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(outcome.as_deref(), Some("done"));
        let rows: Vec<(String, Value, Value, bool)> = db
            .prepare("SELECT sample_id, attribution_json, usage_json, complete FROM chat_usage_samples WHERE execution_id = 'exec' ORDER BY sample_id")
            .unwrap()
            .query_map([], |row| {
                let json = |i| row.get::<_, String>(i).map(|text| serde_json::from_str(&text).unwrap());
                Ok((row.get(0)?, json(1)?, json(2)?, row.get(3)?))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, "opencode-background:unrecoverable");
        assert_eq!(rows[0].1["reason"], "child_model_unknown");
        assert!(!rows[0].3);
        assert_eq!(rows[1].0, "prt_bg_s");
        assert_eq!(rows[1].1["model"], "bg-model");
        assert_eq!(rows[1].2["inputTokens"], 5);
        // Windows cannot delete the database while a connection holds it open.
        drop(store);
        drop(db);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Live V1 1.18.33 shape (oc-research v1probe): a background task's result is delivered as a
    /// synthetic `<task id=…>` user message that natively wakes the parent with no app turn. That
    /// run, its tool and its printed run id belong to the turn that spawned the task; a later app
    /// turn listing the same history never accounts it.
    #[tokio::test]
    async fn v1_woken_parent_run_belongs_to_the_turn_that_spawned_its_task() {
        use axum::{extract::Path, routing::get, Json, Router};
        let step = |message: &str, part: &str, tokens: Option<Value>| {
            let mut finish = json!({"id":format!("{part}_f"),"messageID":message,"sessionID":"ses_main","type":"step-finish"});
            if let Some(tokens) = tokens {
                finish["tokens"] = tokens;
            }
            vec![
                json!({"id":format!("{part}_s"),"messageID":message,"sessionID":"ses_main","type":"step-start"}),
                finish,
            ]
        };
        let tokens = json!({"input":5,"output":2,"reasoning":0,"cache":{"read":0,"write":0}});
        let assistant = |id: &str, parent: &str, created: i64, parts: Vec<Value>| {
            json!({"info":{"id":id,"sessionID":"ses_main","role":"assistant","parentID":parent,
                "modelID":"big-pickle","providerID":"opencode","time":{"created":created}},"parts":parts})
        };
        let mut spawn = step("msg_turn", "prt_turn", Some(tokens.clone()));
        spawn.push(json!({"id":"prt_task","messageID":"msg_turn","sessionID":"ses_main","type":"tool","tool":"task",
            "state":{"status":"completed","metadata":{"sessionId":"ses_bg","background":true}}}));
        let mut woken = step("msg_woken", "prt_woken", Some(tokens.clone()));
        woken.push(json!({"id":"prt_bash","messageID":"msg_woken","sessionID":"ses_main","type":"tool","tool":"bash",
            "state":{"status":"completed","input":{"command":"orx exp run exp"},"output":"  run  7c1a\n"}}));
        let main = json!([
            {"info":{"id":"msg_user","sessionID":"ses_main","role":"user","time":{"created":100}},"parts":[]},
            assistant("msg_turn", "msg_user", 101, spawn),
            {"info":{"id":"msg_delivery","sessionID":"ses_main","role":"user","time":{"created":200}},
                "parts":[{"id":"prt_d","type":"text","synthetic":true,
                    "text":"<task id=\"ses_bg\" state=\"completed\">\n<summary>done</summary>"}]},
            assistant("msg_woken", "msg_delivery", 201, woken),
            {"info":{"id":"msg_later","sessionID":"ses_main","role":"user","time":{"created":300}},"parts":[]},
            assistant("msg_next", "msg_later", 301, step("msg_next", "prt_next", Some(tokens))),
        ]);
        let background = json!([{"info":{"id":"msg_bg","sessionID":"ses_bg","role":"assistant","modelID":"bg-model",
            "providerID":"p","time":{"created":150}},"parts":[
            {"id":"prt_bg","messageID":"msg_bg","sessionID":"ses_bg","type":"step-start"}]}]);
        let app = Router::new()
            .route("/session/status", get(|| async { Json(json!({})) }))
            .route(
                "/session/{id}/message",
                get(move |Path(id): Path<String>| {
                    let listing = if id == "ses_bg" {
                        background.clone()
                    } else {
                        main.clone()
                    };
                    async move { Json(listing) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let http = reqwest::Client::new();

        // No later app turn: the held turn's watcher alone accounts the woken run.
        let held = Recorded::default();
        settle_watch(
            &held,
            &json!({"native":"ses_main","startedAt":100,"roots":[["ses_bg","prt_task"]]}),
            live(v1_endpoint(&base)),
        )
        .await;
        let samples = held.samples.into_inner().unwrap();
        assert!(
            samples["prt_woken_s"].2,
            "woken run counted with its tokens"
        );
        assert!(samples.contains_key("prt_bg"));
        assert!(
            !samples.contains_key("prt_next_s"),
            "a later turn's own step"
        );
        assert!(
            !samples.contains_key("prt_turn_s"),
            "the turn itself already counted it"
        );
        assert_eq!(
            held.invokers.into_inner().unwrap()["prt_bash"],
            "big-pickle"
        );
        let evidence = held.evidence.into_inner().unwrap();
        assert_eq!(evidence[0].id, "prt_bash");
        assert!(evidence[0]
            .state
            .as_ref()
            .unwrap()
            .output
            .as_deref()
            .unwrap()
            .contains("7c1a"));

        // The later app turn (started before the wake) never accounts the woken run.
        let later = Recorded::default();
        capture_v1(
            &later,
            History::Server(&http, &base),
            vec![("ses_main".to_string(), None)],
            (150, None),
            None,
        )
        .await;
        server.abort();
        let samples = later.samples.into_inner().unwrap();
        assert!(samples.contains_key("prt_next_s"));
        assert!(!samples.contains_key("prt_woken_s"));
    }

    /// Replays a captured OpenCode 1.18.33 turn: bash, then a `task` subagent that runs bash.
    #[test]
    fn native_v1_turn_attributes_every_step_and_tool_to_its_own_message_model() {
        use crate::store::{Attribution, Missing};
        const MAIN: &str = "ses_f0c30aab3ffev2glu5jL6rRnqj";
        const CHILD: &str = "ses_f0c30789cffe5d20g8AoPPYRBr";
        const TASK: &str = "prt_0f3cf875e001Klz5seORXpbK4K";
        let events: Vec<Value> = include_str!("fixtures/opencode-v1-turn-events.jsonl")
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let mut ctx = TurnCtx::test_stub();
        let (mut messages, mut subs) = (HashSet::new(), HashMap::new());
        for event in &events {
            handle_event(&mut ctx, MAIN, event, &mut messages, &mut subs);
        }
        assert_eq!(subs.get(CHILD).map(String::as_str), Some(TASK));
        // Each part's message model, as native history reports it.
        let models: HashMap<String, crate::store::InvocationIdentity> = events
            .iter()
            .filter(|event| event["type"] == "message.updated")
            .filter_map(|event| v1_identity(&event["properties"]["info"]))
            .collect();
        let recorded = Recorded::default();
        let mut replay = TreeState::default();
        for part in events
            .iter()
            .filter_map(|event| event.pointer("/properties/part"))
        {
            let identity = part["messageID"].as_str().and_then(|id| models.get(id));
            record_v1_part(
                &recorded,
                part,
                identity,
                v1_tool_part_id(part, MAIN, &subs),
                part["sessionID"] != MAIN,
                &mut replay,
            );
        }
        assert_eq!(
            replay.invokers,
            HashSet::from([
                "prt_0f3cf7e3f001MTz4d5J06KbOBZ".to_string(),
                TASK.to_string(),
                format!("{TASK}:prt_0f3cf9941001wZ3V6C7leaXefR"),
            ])
        );
        // Five steps (three parent, two child): identity and tokens share one sample each.
        let samples = recorded.samples.into_inner().unwrap();
        assert_eq!(samples.len(), 5);
        let exact = Attribution::Exact {
            model: "big-pickle".into(),
            provider: Some("opencode".into()),
        };
        assert!(samples
            .values()
            .all(|(attribution, _, complete)| attribution == &exact && *complete));
        let first = &samples["prt_0f3cf7dc9001hTNuf5YeAipgt2"].1;
        assert_eq!(first.input_tokens, Some(6283 + 1941));
        assert_eq!(first.output_tokens, Some(31));

        // A child's step whose model is not known yet never borrows its parent's model.
        let child_finish = events
            .iter()
            .filter_map(|event| event.pointer("/properties/part"))
            .find(|part| part["sessionID"] == CHILD && part["type"] == "step-finish")
            .unwrap();
        let (_, attribution, ..) =
            v1_step_sample(child_finish, None, true, &mut HashMap::new()).unwrap();
        assert_eq!(
            attribution,
            Attribution::Unresolved {
                reason: Missing::ChildModelUnknown
            }
        );
    }

    /// Replays the same turn's native history with a grandchild, a native retry's unfinished
    /// attempt, a resumed child's earlier message, and a `task_id` that resumes an ancestor.
    #[tokio::test]
    async fn v1_capture_descends_into_every_subagent_once() {
        use axum::{extract::Path, routing::get, Json, Router};
        const MAIN: &str = "ses_f0c30aab3ffev2glu5jL6rRnqj";
        const CHILD: &str = "ses_f0c30789cffe5d20g8AoPPYRBr";
        const TASK: &str = "prt_0f3cf875e001Klz5seORXpbK4K";
        let fixture: Value =
            serde_json::from_str(include_str!("fixtures/opencode-v1-messages.json")).unwrap();
        let started_at = fixture["main"][0]["info"]["time"]["created"]
            .as_i64()
            .unwrap();
        let mut main = fixture["main"].clone();
        // The first attempt of a natively retried request streamed but never finished.
        main[1]["parts"].as_array_mut().unwrap().insert(
            0,
            json!({"id":"prt_0f3cf7dc9000retried00000","messageID":"msg_0f3cf6e4f001VaKSOLsRFi3dsY","sessionID":MAIN,"type":"step-start"}),
        );
        let mut child = fixture["child"].clone();
        let mut earlier = child[1].clone();
        earlier["info"]["id"] = json!("msg_earlier_turn");
        earlier["info"]["time"]["created"] = json!(started_at - 1);
        child.as_array_mut().unwrap().insert(0, earlier);
        let grand_task = json!({"id":"prt_grand_task","messageID":"msg_0f3cf8773001sB85ZJIXdf6L7F","sessionID":CHILD,"type":"tool","tool":"task","callID":"call_grand","state":{"status":"completed","input":{},"output":"","metadata":{"sessionId":"ses_grandchild","background":true}}});
        child[2]["parts"].as_array_mut().unwrap().push(grand_task);
        let grandchild = json!([{
            "info":{"id":"msg_grand","sessionID":"ses_grandchild","role":"assistant","modelID":"grand-model","providerID":"grand","time":{"created":started_at + 1}},
            "parts":[
                {"id":"prt_grand_start","messageID":"msg_grand","sessionID":"ses_grandchild","type":"step-start"},
                {"id":"prt_grand_bash","messageID":"msg_grand","sessionID":"ses_grandchild","type":"tool","tool":"bash","state":{"status":"completed","input":{"command":"true"},"output":""}},
                {"id":"prt_grand_resume","messageID":"msg_grand","sessionID":"ses_grandchild","type":"tool","tool":"task","state":{"status":"completed","input":{},"output":"","metadata":{"sessionId":MAIN}}}
            ]
        }]);
        let sessions = HashMap::from([
            (MAIN.to_string(), main),
            (CHILD.to_string(), child),
            ("ses_grandchild".to_string(), grandchild),
        ]);
        let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
        let log = requests.clone();
        let app = Router::new().route(
            "/session/{id}/message",
            get(move |Path(id): Path<String>| {
                log.lock().unwrap().push(id.clone());
                let listing = sessions[&id].clone();
                async move { Json(listing) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let recorded = Recorded::default();
        let captured = capture_v1(
            &recorded,
            History::Server(&reqwest::Client::new(), &base),
            vec![(MAIN.to_string(), None)],
            (started_at, None),
            None,
        )
        .await;
        server.abort();
        assert_eq!(requests.lock().unwrap().len(), 3, "each session read once");
        assert_eq!(
            captured.background,
            [(
                "ses_grandchild".to_string(),
                Some(format!("{TASK}:prt_grand_task"))
            )]
        );

        let samples = recorded.samples.into_inner().unwrap();
        // Five finished steps, the grandchild's unfinished one, and the retried attempt.
        assert_eq!(samples.len(), 7);
        let (retried_model, retried_usage, retried_complete) =
            &samples["prt_0f3cf7dc9000retried00000"];
        assert!(
            matches!(retried_model, crate::store::Attribution::Exact { model, .. } if model == "big-pickle")
        );
        assert_eq!(retried_usage, &crate::store::TokenUsage::default());
        assert!(!retried_complete);
        assert!(samples["prt_0f3cf7dc9001hTNuf5YeAipgt2"].2);
        let (grand_model, grand_usage, _) = &samples["prt_grand_start"];
        assert!(
            matches!(grand_model, crate::store::Attribution::Exact { model, .. } if model == "grand-model")
        );
        assert_eq!(grand_usage, &crate::store::TokenUsage::default());
        assert!(!samples.keys().any(|id| id.contains("earlier")));

        let invokers = recorded.invokers.into_inner().unwrap();
        assert_eq!(invokers[TASK], "big-pickle");
        assert_eq!(
            invokers[&format!("{TASK}:prt_0f3cf9941001wZ3V6C7leaXefR")],
            "big-pickle"
        );
        assert_eq!(
            invokers[&format!("{TASK}:prt_grand_task:prt_grand_bash")],
            "grand-model"
        );
    }

    /// A real OpenCode 1.18.33 abort mid-subagent: `POST /abort` had already persisted both aborted
    /// steps, which streamed but never finished, so each is a known request without counters.
    #[tokio::test]
    async fn aborted_v1_turn_keeps_every_started_step_without_inventing_counters() {
        use axum::{extract::Path, routing::get, Json, Router};
        let fixture: Value =
            serde_json::from_str(include_str!("fixtures/opencode-v1-abort-messages.json")).unwrap();
        let main = fixture["main"].as_str().unwrap().to_string();
        let started_at = fixture["sessions"][&main][0]["info"]["time"]["created"]
            .as_i64()
            .unwrap();
        let sessions = fixture["sessions"].clone();
        let app = Router::new().route(
            "/session/{id}/message",
            get(move |Path(id): Path<String>| {
                let listing = sessions[&id].clone();
                async move { Json(listing) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let recorded = Recorded::default();
        capture_v1(
            &recorded,
            History::Server(&reqwest::Client::new(), &base),
            vec![(main, None)],
            (started_at, None),
            None,
        )
        .await;
        server.abort();
        let samples = recorded.samples.into_inner().unwrap();
        let exact = crate::store::Attribution::Exact {
            model: "big-pickle".into(),
            provider: Some("opencode".into()),
        };
        for step in [
            "prt_0f3fd25150011uI3i2udhGGxOn",
            "prt_0f3fd2e27001T46c4latb67zMo",
        ] {
            assert_eq!(
                samples[step],
                (exact.clone(), crate::store::TokenUsage::default(), false)
            );
        }
        assert_eq!(samples.len(), 2);
        assert_eq!(
            recorded.invokers.into_inner().unwrap()["prt_0f3fd2519001es0eud2F7drRcq"],
            "big-pickle"
        );
    }

    /// 1.18.33 writes `step-start` only when the provider starts a step (processor.ts), and a
    /// provider error or abort halts the same message. A step that failed after starting keeps its
    /// observed model; a request that failed or was cancelled before any step records no model.
    #[tokio::test]
    async fn v1_failures_keep_started_steps_and_record_nothing_before_execution() {
        use axum::{extract::Path, routing::get, Json, Router};
        let mut fixture: Value =
            serde_json::from_str(include_str!("fixtures/opencode-v1-abort-messages.json")).unwrap();
        let main = fixture["main"].as_str().unwrap().to_string();
        let listing = fixture["sessions"][&main].as_array_mut().unwrap();
        let started_at = listing[0]["info"]["time"]["created"].as_i64().unwrap();
        listing[1]["info"]["error"] = json!({"name":"APIError","data":{"message":"Internal Server Error","statusCode":500,"isRetryable":false}});
        for (id, error) in [
            (
                "msg_failed_before",
                json!({"name":"ProviderAuthError","data":{"providerID":"opencode","message":"Unauthorized"}}),
            ),
            (
                "msg_cancelled_before",
                json!({"name":"MessageAbortedError","data":{"message":"Aborted"}}),
            ),
        ] {
            let mut message = listing[1].clone();
            message["info"]["id"] = json!(id);
            message["info"]["error"] = error;
            message["parts"] = json!([]);
            listing.push(message);
        }
        let sessions = fixture["sessions"].clone();
        let app = Router::new().route(
            "/session/{id}/message",
            get(move |Path(id): Path<String>| {
                let listing = sessions[&id].clone();
                async move { Json(listing) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let recorded = Recorded::default();
        capture_v1(
            &recorded,
            History::Server(&reqwest::Client::new(), &base),
            vec![(main, None)],
            (started_at, None),
            None,
        )
        .await;
        server.abort();
        let samples = recorded.samples.into_inner().unwrap();
        let exact = crate::store::Attribution::Exact {
            model: "big-pickle".into(),
            provider: Some("opencode".into()),
        };
        assert_eq!(
            samples["prt_0f3fd25150011uI3i2udhGGxOn"],
            (exact, crate::store::TokenUsage::default(), false),
            "the step that failed after starting"
        );
        assert_eq!(samples.len(), 2, "no sample for either unstarted request");
    }

    /// A background subagent keeps being captured after its turn until no session is busy.
    #[tokio::test]
    async fn v1_background_watch_captures_until_nothing_is_busy() {
        use axum::{routing::get, Json, Router};
        let polls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let app = Router::new()
            .route(
                "/session/status",
                get(move || {
                    let poll = polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    async move {
                        Json(if poll < 2 {
                            json!({"ses_bg":{"type":"busy"}})
                        } else {
                            json!({})
                        })
                    }
                }),
            )
            .route(
                "/session/{id}/message",
                get(|| async {
                    Json(json!([{
                        "info":{"id":"msg_bg","sessionID":"ses_bg","role":"assistant","modelID":"bg-model","providerID":"p","time":{"created":10}},
                        "parts":[
                            {"id":"prt_start","messageID":"msg_bg","sessionID":"ses_bg","type":"step-start"},
                            {"id":"prt_finish","messageID":"msg_bg","sessionID":"ses_bg","type":"step-finish","tokens":{"input":3,"output":1,"reasoning":0,"cache":{"read":0,"write":0}}}
                        ]
                    }]))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let recorded = Recorded::default();
        settle_watch(
            &recorded,
            &json!({"native":"ses_main","startedAt":10,"roots":[["ses_bg","prt_task"]]}),
            live(v1_endpoint(&base)),
        )
        .await;
        server.abort();
        let samples = recorded.samples.into_inner().unwrap();
        assert!(samples["prt_start"].2);
    }

    /// V1: a run a background grandchild's result woke in a continued child session belongs to the
    /// turn that spawned the grandchild, not to a later turn using that child session.
    #[tokio::test]
    async fn v1_grandchild_woken_child_run_belongs_to_the_turn_that_spawned_the_grandchild() {
        use axum::{extract::Path, routing::get, Json, Router};
        let assistant = |session: &str,
                         id: &str,
                         parent: &str,
                         created: i64,
                         mut parts: Vec<Value>| {
            parts.insert(0, json!({"id":format!("{id}_s"),"messageID":id,"sessionID":session,"type":"step-start"}));
            json!({"info":{"id":id,"sessionID":session,"role":"assistant","parentID":parent,
                "modelID":format!("{session}-model"),"providerID":"p","time":{"created":created}},"parts":parts})
        };
        let user = |session: &str, id: &str, created: i64, parts: Vec<Value>| json!({"info":{"id":id,"sessionID":session,"role":"user","time":{"created":created}},"parts":parts});
        let task = |id: &str, session: &str, background: bool| {
            json!({"id":id,"type":"tool","tool":"task","state":{"status":"completed","input":{},"output":"",
                "metadata":{"sessionId":session,"background":background}}})
        };
        let sessions = HashMap::from([
            (
                "ses_main".to_string(),
                json!([
                    user("ses_main", "msg_user_a", 10, vec![]),
                    assistant(
                        "ses_main",
                        "msg_a",
                        "msg_user_a",
                        11,
                        vec![task("prt_task_a", "ses_child", false)]
                    ),
                    user("ses_main", "msg_user_b", 20, vec![]),
                    assistant(
                        "ses_main",
                        "msg_b",
                        "msg_user_b",
                        21,
                        vec![task("prt_task_b", "ses_child", false)]
                    ),
                ]),
            ),
            (
                "ses_child".to_string(),
                json!([
                    assistant(
                        "ses_child",
                        "msg_child_a",
                        "msg_prompt_a",
                        12,
                        vec![task("prt_grand_task", "ses_grand", true)]
                    ),
                    assistant("ses_child", "msg_child_b", "msg_prompt_b", 22, vec![]),
                    user(
                        "ses_child",
                        "msg_delivery",
                        30,
                        vec![json!({"id":"prt_d","type":"text","synthetic":true,
                    "text":"<task id=\"ses_grand\" state=\"completed\">"})]
                    ),
                    assistant(
                        "ses_child",
                        "msg_child_woken",
                        "msg_delivery",
                        31,
                        vec![json!({"id":"prt_run","type":"tool","tool":"bash",
                    "state":{"status":"completed","input":{"command":"orx exp run exp"},"output":"  run  7c1a\n"}})]
                    ),
                ]),
            ),
            (
                "ses_grand".to_string(),
                json!([assistant(
                    "ses_grand",
                    "msg_grand",
                    "msg_prompt_g",
                    13,
                    vec![]
                )]),
            ),
        ]);
        let app = Router::new()
            .route("/session/status", get(|| async { Json(json!({})) }))
            .route(
                "/session/{id}/message",
                get(move |Path(id): Path<String>| {
                    let listing = sessions[&id].clone();
                    async move { Json(listing) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let http = reqwest::Client::new();

        // Turn A's own capture holds the grandchild and notes the session its result reports to.
        let turn_a = capture_v1(
            &Recorded::default(),
            History::Server(&http, &base),
            vec![("ses_main".into(), None)],
            (10, None),
            None,
        )
        .await;
        assert_eq!(turn_a.parents, HashSet::from(["ses_child".to_string()]));

        // Turn B continues the child session and never accounts the woken run.
        let later = Recorded::default();
        capture_v1(
            &later,
            History::Server(&http, &base),
            vec![("ses_main".into(), None)],
            (20, None),
            None,
        )
        .await;
        let later = later.samples.into_inner().unwrap();
        assert!(later.contains_key("msg_child_b_s"));
        assert!(!later.contains_key("msg_child_woken_s"));

        // Turn A's held watcher accounts it, with the tool part a run it launched binds through.
        let held = Recorded::default();
        settle_watch(
            &held,
            &json!({"native":"ses_main","startedAt":10,"roots":turn_a.background,"parents":turn_a.parents}),
            live(v1_endpoint(&base)),
        )
        .await;
        server.abort();
        let samples = held.samples.into_inner().unwrap();
        assert!(samples.contains_key("msg_child_woken_s"));
        assert!(samples.contains_key("msg_grand_s"));
        assert!(!samples.contains_key("msg_child_b_s"));
        assert!(held
            .evidence
            .into_inner()
            .unwrap()
            .iter()
            .any(|part| part.id == "prt_run"));
        assert_eq!(
            held.invokers.into_inner().unwrap()["prt_run"],
            "ses_child-model"
        );
    }

    /// A live server answering 404 for a subagent session forever ends the watch only once the
    /// native store confirms that session is gone, keeping every captured identity beside the
    /// explicit unresolved request.
    #[tokio::test]
    async fn a_session_missing_natively_closes_the_watch_explicitly() {
        use axum::{extract::Path, http::StatusCode, routing::get, Json, Router};
        let app = Router::new()
            .route(
                "/session/status",
                get(|| async { Json(json!({"ses_bg":{"type":"busy"}})) }),
            )
            .route(
                "/session/{id}/message",
                get(|Path(id): Path<String>| async move {
                    (id != "ses_bg")
                        .then_some(Json(json!([])))
                        .ok_or(StatusCode::NOT_FOUND)
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let endpoint = v1_endpoint(&base);
        let recorded = Recorded::default();
        let exact = crate::store::Attribution::Exact {
            model: "main-model".into(),
            provider: Some("p".into()),
        };
        recorded.sample(
            "prt_turn_s",
            exact.clone(),
            crate::store::TokenUsage::default(),
            false,
        );
        let asked = Mutex::new(Vec::new());
        settle_watch(
            &recorded,
            &json!({"native":"ses_main","startedAt":10,"roots":[["ses_bg","prt_task"]]}),
            |missing: Option<String>| {
                asked.lock().unwrap().push(missing.clone());
                // Stands in for `native_store::opencode_session`: ses_bg's row is gone.
                std::future::ready(match missing.as_deref() {
                    Some("ses_bg") => Err(anyhow!("OpenCode session ses_bg no longer exists")),
                    _ => Ok(Some(endpoint.clone())),
                })
            },
        )
        .await;
        server.abort();
        assert_eq!(
            asked.into_inner().unwrap(),
            [None, Some("ses_bg".to_string())]
        );
        let samples = recorded.samples.into_inner().unwrap();
        assert_eq!(samples["prt_turn_s"].0, exact);
        assert_eq!(
            samples["opencode-background:unrecoverable"].0,
            crate::store::Attribution::Unresolved {
                reason: crate::store::Missing::ChildModelUnknown
            }
        );
    }

    /// A background grandchild the watched subagent spawns only after the first poll is tracked:
    /// the watch waits for its result, which natively wakes the subagent, and captures that run.
    #[tokio::test]
    async fn v1_watch_tracks_a_background_grandchild_spawned_after_the_first_poll() {
        use axum::{extract::Path, routing::get, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let polls = std::sync::Arc::new(AtomicUsize::new(0));
        let seen = polls.clone();
        let step = |session: &str, message: &str, part: &str| {
            vec![
                json!({"id":format!("{part}_s"),"messageID":message,"sessionID":session,"type":"step-start"}),
                json!({"id":format!("{part}_f"),"messageID":message,"sessionID":session,"type":"step-finish",
                    "tokens":{"input":3,"output":1,"reasoning":0,"cache":{"read":0,"write":0}}}),
            ]
        };
        let assistant = move |session: &str,
                              id: &str,
                              parent: &str,
                              created: i64,
                              parts: Vec<Value>| {
            json!({"info":{"id":id,"sessionID":session,"role":"assistant","parentID":parent,
                "modelID":format!("{session}-model"),"providerID":"p","time":{"created":created}},"parts":parts})
        };
        let delivery = |session: &str, id: &str, child: &str, created: i64| {
            json!({"info":{"id":id,"sessionID":session,"role":"user","time":{"created":created}},
                "parts":[{"id":format!("{id}_p"),"type":"text","synthetic":true,
                    "text":format!("<task id=\"{child}\" state=\"completed\">")}]})
        };
        let app = Router::new()
            .route(
                "/session/status",
                get(move || {
                    let poll = polls.fetch_add(1, Ordering::SeqCst);
                    async move {
                        Json(match poll {
                            0 => json!({"ses_b":{"type":"busy"}}),
                            1 => json!({"ses_x":{"type":"busy"}}),
                            _ => json!({}),
                        })
                    }
                }),
            )
            .route(
                "/session/{id}/message",
                get(move |Path(id): Path<String>| {
                    let polls = seen.load(Ordering::SeqCst);
                    let listing = match id.as_str() {
                        "ses_main" => json!([delivery("ses_main", "msg_delivery_b", "ses_b", 25)]),
                        "ses_x" => json!([assistant("ses_x", "msg_x", "msg_px", 20, step("ses_x", "msg_x", "prt_x"))]),
                        _ => {
                            let mut parts = step("ses_b", "msg_b", "prt_b");
                            // Spawned only from the second poll on; its result arrives after idle.
                            if polls >= 2 {
                                parts.push(json!({"id":"prt_task_x","messageID":"msg_b","sessionID":"ses_b","type":"tool","tool":"task",
                                    "state":{"status":"completed","input":{},"output":"","metadata":{"sessionId":"ses_x","background":true}}}));
                            }
                            let mut listing = vec![assistant("ses_b", "msg_b", "msg_pb", 11, parts)];
                            if polls >= 4 {
                                listing.push(delivery("ses_b", "msg_delivery_x", "ses_x", 30));
                                listing.push(assistant("ses_b", "msg_b_woken", "msg_delivery_x", 31, step("ses_b", "msg_b_woken", "prt_bw")));
                            }
                            json!(listing)
                        }
                    };
                    async move { Json(listing) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let recorded = Recorded::default();
        settle_watch(
            &recorded,
            &json!({"native":"ses_main","startedAt":10,"roots":[["ses_b","prt_task_b"]]}),
            live(v1_endpoint(&base)),
        )
        .await;
        server.abort();
        let samples = recorded.samples.into_inner().unwrap();
        assert!(samples["prt_x_s"].2, "the grandchild's step");
        assert!(samples["prt_bw_s"].2, "the run its result woke");
    }

    /// Deletion stops the held tree natively before the server dies; the execution is released
    /// only once a read finds nothing still running.
    #[tokio::test]
    async fn retirement_releases_only_a_tree_that_stopped() {
        use axum::{extract::Path, routing::get, routing::post, Json, Router};
        use std::sync::atomic::{AtomicBool, Ordering};
        let honored = std::sync::Arc::new(AtomicBool::new(false));
        let stopped = std::sync::Arc::new(AtomicBool::new(false));
        let aborts = std::sync::Arc::new(Mutex::new(Vec::new()));
        let (honor, stop, log) = (honored.clone(), stopped.clone(), aborts.clone());
        let app = Router::new()
            .route(
                "/session/status",
                get(move || {
                    let busy = !stopped.load(Ordering::SeqCst);
                    async move {
                        Json(if busy { json!({"ses_bg":{"type":"busy"}}) } else { json!({}) })
                    }
                }),
            )
            .route(
                "/session/{id}/abort",
                post(move |Path(id): Path<String>| {
                    log.lock().unwrap().push(id);
                    if honor.load(Ordering::SeqCst) {
                        stop.store(true, Ordering::SeqCst);
                    }
                    async { Json(json!(true)) }
                }),
            )
            .route(
                "/session/{id}/message",
                get(|Path(id): Path<String>| async move {
                    Json(if id == "ses_bg" {
                        json!([{"info":{"id":"msg_bg","sessionID":"ses_bg","role":"assistant","modelID":"bg-model",
                            "providerID":"p","time":{"created":10}},"parts":[
                            {"id":"prt_start","messageID":"msg_bg","sessionID":"ses_bg","type":"step-start"}]}])
                    } else {
                        json!([])
                    })
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = v1_endpoint(&format!("http://{}", listener.local_addr().unwrap()));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let scope = json!({"native":"ses_main","startedAt":10,"roots":[["ses_bg","prt_task"]]});

        // A read succeeds, but the subagent keeps running: not complete, so it stays held.
        let recorded = Recorded::default();
        let mut watch = Watch::from_scope(&scope).unwrap();
        assert!(!stop_tree(&mut watch, &recorded, &endpoint).await.unwrap());
        assert!(aborts.lock().unwrap().iter().all(|id| id == "ses_bg"));

        honored.store(true, Ordering::SeqCst);
        aborts.lock().unwrap().clear();
        let mut watch = Watch::from_scope(&scope).unwrap();
        assert!(stop_tree(&mut watch, &recorded, &endpoint).await.unwrap());
        server.abort();
        assert_eq!(*aborts.lock().unwrap(), ["ses_bg"]);
        assert!(recorded
            .samples
            .into_inner()
            .unwrap()
            .contains_key("prt_start"));
    }

    pub(super) fn live(
        endpoint: AgentEndpoint,
    ) -> impl Fn(Option<String>) -> std::future::Ready<Result<Option<AgentEndpoint>>> {
        move |_| std::future::ready(Ok(Some(endpoint.clone())))
    }

    fn v1_endpoint(base: &str) -> AgentEndpoint {
        AgentEndpoint {
            base_url: base.to_string(),
            client: reqwest::Client::new(),
            protocol: crate::local::opencode::Protocol::V1,
            legacy_v2_api: false,
        }
    }

    /// The same live watcher rides out an unreachable server, a failed status read and a failed
    /// history read while the subagent still runs, then captures its finished step; history gone
    /// for good instead leaves the partial step beside an explicit unresolved request.
    #[tokio::test]
    async fn v1_background_watch_retries_native_failures_until_the_tree_settles() {
        use axum::{extract::Path, http::StatusCode, routing::get, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let polls = std::sync::Arc::new(AtomicUsize::new(0));
        let reads = std::sync::Arc::new(AtomicUsize::new(0));
        let restart = (polls.clone(), reads.clone());
        let app = Router::new()
            .route(
                "/session/status",
                get(move || {
                    let poll = polls.fetch_add(1, Ordering::SeqCst);
                    async move {
                        match poll {
                            1 => Err(StatusCode::SERVICE_UNAVAILABLE),
                            0 | 2 => Ok(Json(json!({"ses_bg":{"type":"busy"}}))),
                            _ => Ok(Json(json!({}))),
                        }
                    }
                }),
            )
            .route(
                "/session/{id}/message",
                get(move |Path(id): Path<String>| {
                    let read = (id == "ses_bg").then(|| reads.fetch_add(1, Ordering::SeqCst));
                    async move {
                        let delivery = json!([{"info":{"id":"msg_delivery","sessionID":"ses_main","role":"user","time":{"created":20}},
                            "parts":[{"id":"prt_d","type":"text","synthetic":true,"text":"<task id=\"ses_bg\" state=\"completed\">"}]}]);
                        let mut parts = vec![json!({"id":"prt_start","messageID":"msg_bg","sessionID":"ses_bg","type":"step-start"})];
                        match read {
                            None => return Ok(Json(delivery)),
                            Some(1) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
                            Some(0) => {}
                            Some(_) => parts.push(json!({"id":"prt_finish","messageID":"msg_bg","sessionID":"ses_bg","type":"step-finish",
                                "tokens":{"input":3,"output":1,"reasoning":0,"cache":{"read":0,"write":0}}})),
                        }
                        Ok(Json(json!([{"info":{"id":"msg_bg","sessionID":"ses_bg","role":"assistant","modelID":"bg-model",
                            "providerID":"p","time":{"created":10}},"parts":parts}])))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let scope = json!({"native":"ses_main","startedAt":10,"roots":[["ses_bg","prt_task"]]});
        let endpoint = v1_endpoint(&base);
        let resolved = AtomicUsize::new(0);

        // Unreachable first, then live: status busy, status 503, history 500, then idle and done.
        let recorded = Recorded::default();
        settle_watch(&recorded, &scope, |_| {
            let first = resolved.fetch_add(1, Ordering::SeqCst) == 0;
            std::future::ready(Ok((!first).then(|| endpoint.clone())))
        })
        .await;
        let samples = recorded.samples.into_inner().unwrap();
        assert_eq!(samples["prt_start"].1.input_tokens, Some(3));
        assert!(
            samples["prt_start"].2,
            "the step finished after the failures"
        );
        assert!(!samples.contains_key("opencode-background:unrecoverable"));

        // Read while running, then gone for good.
        resolved.store(0, Ordering::SeqCst);
        restart.0.store(0, Ordering::SeqCst);
        restart.1.store(0, Ordering::SeqCst);
        let recorded = Recorded::default();
        settle_watch(&recorded, &scope, |_| {
            std::future::ready(match resolved.fetch_add(1, Ordering::SeqCst) {
                0 => Ok(Some(endpoint.clone())),
                _ => Err(anyhow!("OpenCode session no longer exists")),
            })
        })
        .await;
        server.abort();
        let samples = recorded.samples.into_inner().unwrap();
        assert!(
            matches!(&samples["prt_start"].0, crate::store::Attribution::Exact { model, .. } if model == "bg-model")
        );
        assert_eq!(
            samples["opencode-background:unrecoverable"],
            (
                crate::store::Attribution::Unresolved {
                    reason: crate::store::Missing::ChildModelUnknown
                },
                crate::store::TokenUsage::default(),
                false
            )
        );
    }

    #[test]
    fn reconciliation_reads_only_this_turns_assistant_messages() {
        let listing = json!([
            {"info":{"id":"msg_earlier","role":"assistant","time":{"created":99}},"parts":[]},
            {"info":{"id":"msg_user","role":"user","time":{"created":100}},"parts":[]},
            {"info":{"id":"msg_step","role":"assistant","time":{"created":100}},"parts":[]},
        ]);
        let ids: Vec<_> = v1_turn_messages(&listing, 100)
            .map(|message| message["info"]["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["msg_step"]);
    }

    /// A `#!/bin/sh` stand-in for the opencode CLI, as in `detect.rs`.
    #[cfg(unix)]
    fn sh_script(name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let dir =
            std::env::temp_dir().join(format!("orx-opencode-capture-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let body = body.replacen(
            "#!/bin/sh\n",
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 1.18.31; exit 0; fi\n",
            1,
        );
        std::fs::write(&path, body).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// Capture uses a private regular file because resolved config can contain API keys.
    #[cfg(unix)]
    #[tokio::test]
    async fn child_stdout_is_a_private_file() {
        let script = sh_script(
            "stdout-kind",
            "#!/bin/sh\n[ -f /dev/stdout ] && ls -lL /dev/stdout\n",
        );

        let binary = crate::local::opencode::resolve_binary_at(script.clone())
            .await
            .unwrap();
        let out = run_models(binary.path.clone(), &[])
            .await
            .expect("child output");
        // macOS reports the write-only descriptor mode through /dev/stdout.
        assert!(
            out.starts_with("-rw-------") || out.starts_with("--w-------"),
            "stdout permissions: {out}"
        );

        std::fs::remove_dir_all(script.parent().unwrap()).ok();
    }

    /// Read the whole capture file after the child exits, including output over 64 KiB.
    #[cfg(unix)]
    #[tokio::test]
    async fn output_past_the_pipe_capacity_arrives_whole() {
        let script = sh_script(
            "big-output",
            "#!/bin/sh\nawk 'BEGIN { for (i = 0; i < 7000; i++) printf \"%010d\", i }'\n",
        );

        let binary = crate::local::opencode::resolve_binary_at(script.clone())
            .await
            .unwrap();
        let out = run_models(binary.path.clone(), &[])
            .await
            .expect("child output");
        assert_eq!(out.len(), 70_000);
        assert!(
            out.ends_with("0000006999"),
            "tail: {}",
            &out[out.len() - 20..]
        );

        std::fs::remove_dir_all(script.parent().unwrap()).ok();
    }
}
