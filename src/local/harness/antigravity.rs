//! Google Antigravity harness.
//!
//! Chat: one `agy --output-format stream-json` child per turn. Multi-turn continues
//! via `--conversation <conversation_id>` from the init/result `conversation_id`. Isolated
//! ORX worktrees are the child's current working directory.
//!
//! The playbook is pointed at on the first turn (the file is already in the
//! worktree via [`ensure_playbook`]); session skills land in `.agents/skills`.
//!
//! Headless tools use the OpenResearch approval hook; explicit bypass skips its cards.
//!
//! Detection: `agy` on PATH or in `~/.local/bin`, `~/.gemini/bin`, or `~/.gemini/antigravity-cli/bin`;
//! `agy models` for catalog and authentication verification.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

use super::detect::{resolve_symlinks, HarnessAuthState, HarnessInfo, ModelInfo};
use super::options::{
    HarnessOptions, OptionChoice, PermissionMode, PlanActivation, REASONING_DEFAULT_ID,
};
use super::{Harness, ResumeAction, TurnFailure, TurnOutcome, TurnResult, TURN_WATCHDOG};
use crate::error::{anyhow, Result};
use crate::local::chat::{
    find_part_mut, harness_log, prepare_env, set_chat_session_env, DeliveryState, PromptAnswer,
    ResumeCtx, TurnCtx, WirePart, WirePrompt, WireToolState,
};
use crate::local::opencode::{ensure_playbook, PLAYBOOK_REL};
use crate::local::shell_env::{find_in_dir, find_on_path};

const AGY_REINSTALL: &str =
    "Reinstall Antigravity CLI via curl -fsSL https://antigravity.google/cli/install.sh | bash";
const MODELS_TIMEOUT: Duration = Duration::from_secs(15);

pub struct Antigravity;

impl Antigravity {
    /// `snapshot` reports discovery only: `agy`'s auth *is* the model list,
    /// so a child-free pass marks the install pending and leaves readiness to
    /// the background full pass.
    async fn detect_at(&self, snapshot: bool) -> Option<HarnessInfo> {
        let mut info = HarnessInfo::new(self.id(), self.name());
        super::detect::record_selected(
            &mut info,
            snapshot,
            "antigravity",
            find_agy,
            find_agy_working(),
        )
        .await;
        // Nothing else about an installed agy is cheap to verify — `detect_one`
        // marks the snapshot answer pending. A missing install still falls
        // through to pick up its note.
        if snapshot && info.installed {
            return Some(info);
        }
        if info.installed && !info.install_broken {
            if let Some(bin) = info.bin_path.as_deref().map(Path::new) {
                match super::detect::timed_probe("antigravity", "models", agy_model_list(bin)).await
                {
                    Ok(models) => {
                        info.authenticated = true;
                        info.auth_state = HarnessAuthState::Ready;
                        info.auth_method = Some("oauth");
                        info = info.with_models(models);
                    }
                    Err(error) => {
                        let message = error.to_string();
                        info.auth_state = if message.to_lowercase().contains("sign in") {
                            HarnessAuthState::NeedsLogin
                        } else {
                            HarnessAuthState::Unknown
                        };
                        info.agent_note = Some(message);
                    }
                }
            }
        }
        info.agent_ready = info.ready();
        if info.agent_ready {
            return Some(info);
        } else if info.install_broken {
            info.agent_note = Some(info.broken_note(AGY_REINSTALL));
        } else if info.installed && info.agent_note.is_none() {
            info.agent_note = Some(
                "Sign in by running `agy` in your terminal, then re-check this harness."
                    .to_string(),
            );
        } else if !info.installed {
            info.agent_note = Some(
                "Install Antigravity CLI with `curl -fsSL https://antigravity.google/cli/install.sh | bash`, then sign in with `agy`."
                    .to_string(),
            );
        }
        Some(info)
    }
}

#[async_trait]
impl Harness for Antigravity {
    fn id(&self) -> &'static str {
        "antigravity"
    }

    fn name(&self) -> &'static str {
        "Google Antigravity"
    }

    fn supports_chat(&self) -> bool {
        true
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
        HarnessOptions::none().with_permission_choices(
            vec![
                OptionChoice::described(
                    "default",
                    "Ask for approval",
                    "Ask before changes; allow read-only planning",
                ),
                OptionChoice::described(
                    "bypass",
                    "Bypass permissions",
                    "Allow commands and skip tool confirmation prompts",
                ),
            ],
            "bypass",
            PlanActivation::Command,
        )
    }

    async fn resume_from_prompt(
        &self,
        ctx: &ResumeCtx,
        prompt: &WirePrompt,
        answer: &PromptAnswer,
    ) -> Result<ResumeAction> {
        if prompt.kind == "permission" {
            if let Some(native_id) = &prompt.native_id {
                if !ctx.is_busy().await {
                    ctx.host
                        .resolve_zombie_prompt(&ctx.session_id, &answer.prompt_id);
                    return Err(anyhow!("this approval is no longer pending"));
                }
                let decision = if answer.approve {
                    crate::local::chat::PermissionDecision::Allow {
                        updated_input: prompt.tool_input.clone(),
                    }
                } else {
                    crate::local::chat::PermissionDecision::Deny {
                        message: format!(
                            "The user denied this action. Do not retry it. {}",
                            answer.note.as_deref().unwrap_or("")
                        ),
                    }
                };
                ctx.host.settle_permission(native_id, decision)?;
                return Ok(ResumeAction::Handled { plan_mode: None });
            }
        }
        if prompt.kind != "plan" {
            return Ok(ResumeAction::Nothing);
        }
        if !answer.approve && answer.note.as_deref().is_none_or(|s| s.trim().is_empty()) {
            return Ok(ResumeAction::Nothing);
        }
        Ok(ResumeAction::SendMessage {
            text: super::synthesize_resume("plan", answer).0,
            mode: None,
            plan_mode: Some(!answer.approve),
        })
    }

    fn config_home(&self) -> Option<PathBuf> {
        Some(dirs::home_dir()?.join(".gemini").join("antigravity-cli"))
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
        Some(super::CLAUDE_SKILL)
    }

    fn session_skills_dir(&self) -> Option<&'static str> {
        Some(".agents/skills")
    }
}

/// `agy` on PATH, then common install locations under `~/.local/bin`,
/// `~/.gemini/bin`, or `~/.gemini/antigravity-cli/bin`, in preference order.
fn agy_candidates() -> Vec<PathBuf> {
    let home_dirs = dirs::home_dir().into_iter().flat_map(|home| {
        let gemini = home.join(".gemini");
        [
            home.join(".local").join("bin"),
            gemini.join("bin"),
            gemini.join("antigravity-cli").join("bin"),
        ]
    });
    let drops = home_dirs
        .chain(dirs::data_local_dir().map(|dir| dir.join("agy").join("bin")))
        .filter_map(|dir| find_in_dir(&dir, "agy"));
    find_on_path("agy")
        .into_iter()
        .chain(drops)
        .map(resolve_symlinks)
        .collect()
}

/// The executable detection selected, else the first candidate — sync callers
/// cannot probe, and must not spawn a launcher detection already skipped.
pub(crate) fn find_agy() -> Option<PathBuf> {
    super::detect::selected_bin("antigravity", agy_candidates())
}

/// The first candidate that actually runs, with its version probe.
pub(super) async fn find_agy_working() -> Option<(PathBuf, super::detect::BinProbe)> {
    super::detect::select_working("antigravity", agy_candidates(), None).await
}

async fn agy_model_list(bin: &Path) -> Result<Vec<ModelInfo>> {
    let mut cmd = Command::new(bin);
    cmd.arg("models")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    prepare_env(&mut cmd);
    cmd.env("NO_COLOR", "1");
    let out = super::detect::detect_spawn_output_timed(cmd, MODELS_TIMEOUT)
        .await
        .ok_or_else(|| {
            anyhow!("Antigravity model discovery timed out. Re-check when connected.")
        })??;
    if !out.status.success() {
        let error = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow!(
            "Antigravity model discovery failed: {}",
            error.trim()
        ));
    }
    let models = parse_agy_model_list(&String::from_utf8_lossy(&out.stdout));
    if models.is_empty() {
        return Err(anyhow!(
            "Antigravity returned no available models. Re-check your account."
        ));
    }
    Ok(models)
}

/// Parse the whitespace-separated model catalog, ignoring
/// informational banner lines such as `Fetching available models...`.
fn parse_agy_model_list(text: &str) -> Vec<ModelInfo> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.starts_with("Fetching")
                || line.starts_with("Available")
                || line.starts_with("Listing")
            {
                return None;
            }
            let (id, label) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
            let label = label.trim();
            if id.is_empty()
                || id == REASONING_DEFAULT_ID
                || !id.chars().all(|c| {
                    c.is_ascii_alphanumeric()
                        || matches!(c, '-' | '_' | '.' | '[' | ']' | '=' | ',')
                })
            {
                return None;
            }
            Some(ModelInfo::new(id).with_label((!label.is_empty()).then_some(label), None))
        })
        .collect()
}

fn first_turn_prompt(text: &str) -> String {
    format!(
        "Read and follow `{PLAYBOOK_REL}` before acting. It is the OpenResearch session playbook for this worktree.\n\n{text}"
    )
}

async fn run_turn(ctx: &mut TurnCtx) -> Result<()> {
    let bin = find_agy().ok_or_else(|| {
        anyhow!("agy not found on PATH — install Antigravity CLI and sign in first")
    })?;
    let project = ctx.project.clone();
    let session_id = ctx.session_id.clone();
    let skills_dir = Antigravity.session_skills_dir();
    let (repo, _playbook) =
        tokio::task::spawn_blocking(move || ensure_playbook(&project, &session_id, skills_dir))
            .await
            .map_err(|e| anyhow!("playbook task failed: {e}"))??;

    let up_port = ctx
        .host
        .up_port()
        .ok_or_else(|| anyhow!("Antigravity requires the OpenResearch approval bridge"))?;
    let bypass = ctx.permission_mode.unwrap_or(PermissionMode::Bypass) == PermissionMode::Bypass;
    let hook_enabled = !bypass || ctx.plan_mode;
    write_approval_hook(&repo, hook_enabled)?;
    let resume = ctx.native_session_id.clone();
    let mut prompt = ctx.text.clone();
    if resume.is_none() {
        prompt = first_turn_prompt(&prompt);
    }

    let mut cmd = Command::new(&bin);
    cmd.args([
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
    ]);

    if let Some(model) = ctx.model.as_deref().filter(|model| !model.is_empty()) {
        cmd.args(["--model", model]);
    }

    // Native headless permission checks deny even after a hook allows the action.
    cmd.arg("--dangerously-skip-permissions");
    if ctx.plan_mode {
        cmd.arg("--mode=plan");
    }
    cmd.args(["--print-timeout", "60m"]);

    if let Some(native_id) = &resume {
        cmd.args(["--conversation", native_id]);
    }

    cmd.current_dir(&repo);
    cmd.arg("--add-dir").arg(&repo);

    let log_name = format!("antigravity-{}", uuid::Uuid::new_v4());
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(harness_log(&log_name)?))
        .kill_on_drop(true);

    prepare_env(&mut cmd);
    cmd.env("NO_COLOR", "1");
    set_chat_session_env(&mut cmd, &ctx.session_id, "antigravity", Some(up_port));
    cmd.env("ORX_SESSION_ID", &ctx.session_id);
    // Hooks (this conversation's and every sub-agent's) record into this turn's execution.
    if let Some(execution) = ctx.usage_execution_id() {
        cmd.env("ORX_USAGE_EXECUTION_ID", execution);
    }
    cmd.env(
        "ORX_GATE_TOKEN",
        ctx.host
            .mint_gate_token(&ctx.session_id, ctx.plan_mode, bypass),
    );
    cmd.env(
        "ORX_AGY_GATE",
        if bypass && !ctx.plan_mode {
            "bypass"
        } else {
            "ask"
        },
    );

    ctx.persist_delivery(DeliveryState::Unknown)?;
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(error) => {
            ctx.mark_delivery(DeliveryState::NotSent);
            return Err(anyhow!("Could not spawn {}: {}", bin.display(), error));
        }
    };
    let mut turn = FinalPass {
        session_id: ctx.session_id.clone(),
        message_id: ctx.assistant.id.clone(),
        execution: ctx.usage_execution_id().map(str::to_string),
        state: TurnState::default(),
    };
    // This turn's transcript rows begin after everything a resumed conversation already holds.
    if let (Some(conversation), Some(root)) = (&resume, brain_root()) {
        turn.state.from_step = transcript_rows(&transcript_path(&root, conversation))
            .iter()
            .filter_map(row_step)
            .max()
            .map_or(0, |step| step + 1);
        turn.state.conversation_id = Some(conversation.clone());
        persist_turn_scope(ctx, &mut turn.state);
    }
    let mut cancellation = TurnProcesses(child.id());
    let mut stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
    let message =
        serde_json::json!({"event":"user","message":{"content":prompt}}).to_string() + "\n";
    tokio::time::timeout(TURN_WATCHDOG, stdin.write_all(message.as_bytes()))
        .await
        .map_err(|_| anyhow!("Antigravity did not read the prompt"))??;
    drop(stdin);
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let mut lines = BufReader::new(stdout).lines();
    let state = &mut turn.state;

    loop {
        match tokio::time::timeout(TURN_WATCHDOG, lines.next_line()).await {
            Ok(Ok(Some(line))) => {
                let Ok(event) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                if matches!(
                    event.get("event").and_then(Value::as_str),
                    Some("step_update")
                ) {
                    ctx.mark_delivery(DeliveryState::Accepted);
                }
                let terminal = apply_event(ctx, state, &event);
                if let Some(sid) = state.conversation_id.as_deref() {
                    ctx.set_native_session_id(sid);
                }
                persist_turn_scope(ctx, state);
                // Retry on tool updates rather than per text delta; the post-exit pass catches the rest.
                if event
                    .pointer("/step_update/step_type")
                    .and_then(Value::as_str)
                    == Some("tool")
                {
                    record_tool_invokers(&ctx.session_id, state);
                }
                ctx.maybe_flush();
                if terminal {
                    break;
                }
            }
            Ok(Ok(None)) => break,
            Ok(Err(error)) => {
                attach_spawns(ctx, state);
                return Err(anyhow!("antigravity stdout: {error}"));
            }
            Err(_) if ctx.host.has_pending_permission(&ctx.session_id) => continue,
            Err(_) => {
                attach_spawns(ctx, state);
                return Err(anyhow!(
                    "Antigravity went silent for {} minutes and was interrupted.",
                    TURN_WATCHDOG.as_secs() / 60
                ));
            }
        }
    }

    let status = tokio::time::timeout(TURN_WATCHDOG, child.wait())
        .await
        .map_err(|_| anyhow!("Antigravity did not exit after its response"))??;
    cancellation.0 = None;
    // The process exited, so its (and its sub-agents') transcripts are final.
    attach_spawns(ctx, state);
    let log_path = crate::store::data_dir().join(format!("agent-{log_name}.log"));
    if !state.saw_result {
        return Err(anyhow!(
            "Antigravity ended without a result ({status}); see {}",
            log_path.display()
        ));
    }
    if !status.success() && !state.turn_errored {
        return Err(anyhow!("Antigravity ended with error ({status})"));
    }
    if ctx.plan_mode && !state.turn_errored {
        if let Some(card) = plan_card(&ctx.assistant.parts, &ctx.assistant.id) {
            ctx.upsert_part(card);
        }
    }
    if !state.turn_errored {
        let _ = std::fs::remove_file(log_path);
    }
    Ok(())
}

/// Resolves pending tool invokers on every exit, including cancellation (the aborted future
/// drops, leaving no `ctx`: the interrupt path persists the stored message, so parts go there).
struct FinalPass {
    session_id: String,
    message_id: String,
    execution: Option<String>,
    state: TurnState,
}

impl Drop for FinalPass {
    fn drop(&mut self) {
        record_tool_invokers(&self.session_id, &mut self.state);
        if self.state.spawns_attached {
            return;
        }
        let parts = turn_parts(&self.state, &self.session_id, self.execution.as_deref());
        if let Err(error) = crate::store::Store::open()
            .and_then(|store| persist_parts(&store, &self.message_id, parts))
        {
            eprintln!("orx up: could not persist interrupted Antigravity sub-agents: {error}");
        }
    }
}

/// Reconciles this turn's final transcript: returns its spawn parts after accounting planners.
fn turn_parts(state: &TurnState, session: &str, execution: Option<&str>) -> Vec<WirePart> {
    let (Some(conversation), Some(root)) = (state.conversation_id.as_deref(), brain_root()) else {
        return Vec::new();
    };
    match crate::store::Store::open() {
        Ok(store) => reconcile_turn(
            &store,
            execution,
            session,
            &root,
            conversation,
            state.from_step,
            false,
        ),
        Err(error) => {
            eprintln!("orx up: could not reconcile Antigravity transcript: {error}");
            Vec::new()
        }
    }
}

fn attach_spawns(ctx: &mut TurnCtx, state: &mut TurnState) {
    for part in turn_parts(state, &ctx.session_id, ctx.usage_execution_id()) {
        ctx.upsert_part_preserving_children(part);
    }
    state.spawns_attached = true;
}

/// Durable, so startup recovery can reconcile this turn's transcript if this process dies.
fn persist_turn_scope(ctx: &TurnCtx, state: &mut TurnState) {
    let (false, Some(execution), Some(conversation)) = (
        state.scoped,
        ctx.usage_execution_id(),
        state.conversation_id.as_deref(),
    ) else {
        return;
    };
    state.scoped = true;
    let scope = serde_json::json!({
        "session": ctx.session_id,
        "message": ctx.assistant.id,
        "conversation": conversation,
        "from": state.from_step,
    });
    if let Err(error) = crate::store::Store::open()
        .and_then(|store| store.set_native_scope(execution, TURN_SCOPE, &scope))
    {
        eprintln!("orx up: could not persist Antigravity turn scope: {error}");
    }
}

struct TurnProcesses(Option<u32>);

impl Drop for TurnProcesses {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            stop_turn_process(pid);
        }
    }
}

#[cfg(unix)]
fn stop_turn_process(pid: u32) {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return;
    };
    // Freeze each parent before discovering children: agy commands create their own sessions.
    if unsafe { libc::kill(pid, libc::SIGSTOP) } != 0 {
        return;
    }
    let snapshot = std::process::Command::new("ps")
        .args(["-ww", "-axo", "pid=,ppid=,args="])
        .output();
    let supervisor_exe = crate::paths::spawnable_exe().ok();
    match snapshot {
        Ok(output) if output.status.success() => {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some(child) = turn_child(line, pid, supervisor_exe.as_deref()) {
                    stop_turn_process(child);
                }
            }
        }
        _ => eprintln!("Could not inspect Antigravity descendants during cancellation"),
    }
    // SAFETY: this is the owned child or a descendant observed while its parent was stopped.
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
}

#[cfg(unix)]
fn turn_child(line: &str, parent: libc::pid_t, supervisor_exe: Option<&Path>) -> Option<u32> {
    let mut fields = line.trim().splitn(2, char::is_whitespace);
    let pid = fields.next()?.parse().ok()?;
    let mut fields = fields.next()?.trim_start().splitn(2, char::is_whitespace);
    if fields.next()?.parse::<libc::pid_t>().ok()? != parent {
        return None;
    }
    let command = fields.next()?.trim_start();
    if let Some((exe, _)) = command.split_once(" supervise ") {
        if supervisor_exe.is_some_and(|expected| Path::new(exe) == expected)
            || (Path::new(exe).file_name().is_some_and(|name| name == "orx")
                && Path::new(exe).is_file())
        {
            return None;
        }
    }
    Some(pid)
}

#[cfg(not(unix))]
fn stop_turn_process(pid: u32) {
    let script = r#"
$ErrorActionPreference = 'Stop'
$processes = @(Get-CimInstance Win32_Process)
function Stop-TurnProcess([uint32] $processId) {
    $children = @($processes | Where-Object { $_.ParentProcessId -eq $processId })
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        if ($child.Name -eq 'orx.exe' -and $child.CommandLine -match '^(?:"[^"\r\n]+"|\S+)\s+supervise\s') { continue }
        Stop-TurnProcess $child.ProcessId
    }
}
Stop-TurnProcess ([uint32] $env:ORX_STOP_PID)
"#;
    let result = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .env("ORX_STOP_PID", pid.to_string())
        .stdout(Stdio::null())
        .status();
    if !result.is_ok_and(|status| status.success()) {
        eprintln!("Could not inspect Antigravity descendants during cancellation");
    }
}

fn write_approval_hook(repo: &Path, enabled: bool) -> Result<()> {
    std::fs::create_dir_all(repo)?;
    let tracked = std::process::Command::new("git")
        .args(["ls-files", "--error-unmatch", ".agents/hooks.json"])
        .current_dir(repo)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if tracked.success() {
        return Err(anyhow!("This project tracks .agents/hooks.json. Antigravity approval setup requires an untracked hook file and will not modify the tracked file."));
    }
    let path = repo.join(".agents/hooks.json");
    let mut hooks = match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str::<Value>(&text)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(error.into()),
    };
    let object = hooks
        .as_object_mut()
        .ok_or_else(|| anyhow!("Antigravity hooks must be an object"))?;
    let exe = crate::paths::spawnable_exe()?;
    #[cfg(not(windows))]
    let command = format!(
        "{} antigravity-gate",
        crate::jobs::ssh::sh_quote(&exe.to_string_lossy())
    );
    #[cfg(windows)]
    let command = {
        anyhow::ensure!(
            cfg!(test)
                || exe
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().eq_ignore_ascii_case("orx.exe")),
            "The Antigravity approval bridge requires the executable name orx.exe"
        );
        // prepare_env puts this executable's directory first on PATH.
        "orx antigravity-gate".to_string()
    };
    object.insert(
        "openresearch-approval".into(),
        serde_json::json!({
            "enabled": enabled,
            "PreToolUse": [{"matcher":"*","hooks":[{"type":"command","command":command,"timeout":3600}]}]
        }),
    );
    object.insert(
        "openresearch-accounting".into(),
        serde_json::json!({
            "enabled": true, "PostInvocation": [{"type":"command","command":command}]
        }),
    );
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(path, serde_json::to_vec_pretty(&hooks)?)?;
    Ok(())
}

pub(crate) fn normalize_tool<'a>(
    name: &'a str,
    params: Option<&Value>,
) -> (&'a str, Option<Value>) {
    let (tool, aliases): (&str, &[(&str, &str)]) = match name {
        "run_command" => ("Bash", &[("CommandLine", "command"), ("Cwd", "cwd")]),
        "view_file" => ("Read", &[("AbsolutePath", "file_path")]),
        "write_to_file" => (
            "Write",
            &[("TargetFile", "file_path"), ("CodeContent", "content")],
        ),
        "replace_file_content" | "multi_replace_file_content" => {
            ("Edit", &[("TargetFile", "file_path")])
        }
        "list_dir" => ("Glob", &[("DirectoryPath", "path")]),
        "grep_search" | "code_search" => ("Grep", &[("Query", "pattern"), ("SearchPath", "path")]),
        "find_by_name" => (
            "Glob",
            &[("Pattern", "pattern"), ("SearchDirectory", "path")],
        ),
        "read_url_content" => ("WebFetch", &[("Url", "url")]),
        "search_web" => ("WebSearch", &[]),
        _ => (name, &[]),
    };
    let mut input = params.cloned();
    if let Some(object) = input.as_mut().and_then(Value::as_object_mut) {
        for &(native, normalized) in aliases {
            if let Some(value) = object.get(native).cloned() {
                object.insert(normalized.into(), value);
            }
        }
    }
    (tool, input)
}

fn error_text(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| value.get("message")?.as_str())
}

fn step_is_terminal(state: &str) -> bool {
    matches!(state, "DONE" | "ERROR" | "CANCELED")
}

fn denied_actions_error(result: &Value) -> Option<String> {
    if result
        .get("response")
        .and_then(Value::as_str)
        .is_some_and(|response| !response.trim().is_empty())
    {
        return None;
    }
    let actions = result.get("denied_actions")?.as_array()?;
    if actions.is_empty() {
        return None;
    }
    let names = actions
        .iter()
        .filter_map(|action| {
            action
                .get("display_name")
                .or_else(|| action.get("action"))
                .and_then(Value::as_str)
        })
        .collect::<Vec<_>>();
    Some(if names.is_empty() {
        "Antigravity denied one or more required actions".into()
    } else {
        format!(
            "Antigravity denied required action(s): {}",
            names.join(", ")
        )
    })
}

#[derive(Default)]
struct TurnState {
    /// The parent conversation (init/result); forwarded steps may carry a sub-agent's own id.
    conversation_id: Option<String>,
    text_part_id: Option<String>,
    text_seq: usize,
    saw_result: bool,
    turn_errored: bool,
    /// Observed (conversation, step index) → whether it is a planner (`agent_response`) step.
    steps: std::collections::HashMap<(String, i64), bool>,
    /// Tool parts (conversation, step index, part id) whose invoking model is not recorded yet.
    unattributed_tools: Vec<(String, i64, String)>,
    spawns_attached: bool,
    /// This turn's first transcript step on the parent conversation.
    from_step: i64,
    scoped: bool,
}

pub(crate) fn invocation_sample_id(conversation: &str, step: i64) -> String {
    format!("antigravity:{conversation}:step:{step}")
}

/// One child an `invoke_subagent` call at planner `step` created (`child` is its native
/// conversation id, or `#i` when the output named fewer children than requested). Reported as
/// `child_model_unknown` until that child's own hook identity exists; a sibling never covers it.
pub(crate) fn spawn_sample_id(conversation: &str, step: i64, child: &str) -> String {
    format!("antigravity:{conversation}:step:{step}:subagent:{child}")
}

/// A native invocation that produced no planner row by the time its hook ran.
pub(crate) fn unmatched_invocation_id(conversation: &str, initial_steps: i64) -> String {
    format!("antigravity:{conversation}:invocation:{initial_steps}")
}

fn brain_root() -> Option<PathBuf> {
    Some(Antigravity.config_home()?.join("brain"))
}

fn transcript_path(root: &Path, conversation: &str) -> PathBuf {
    root.join(conversation)
        .join(".system_generated/logs/transcript.jsonl")
}

/// Native transcript rows, from the sibling `transcript_full.jsonl` when present (real JSON args,
/// never truncated); the compact file JSON-encodes each tool argument as a string.
pub(crate) fn transcript_rows(path: &Path) -> Vec<Value> {
    let (text, compact) =
        match std::fs::read_to_string(path.with_file_name("transcript_full.jsonl")) {
            Ok(text) => (text, false),
            Err(_) => (std::fs::read_to_string(path).unwrap_or_default(), true),
        };
    text.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .map(|mut row| {
            if compact {
                for call in row
                    .get_mut("tool_calls")
                    .and_then(Value::as_array_mut)
                    .into_iter()
                    .flatten()
                {
                    for value in call
                        .get_mut("args")
                        .and_then(Value::as_object_mut)
                        .into_iter()
                        .flat_map(|args| args.values_mut())
                    {
                        if let Some(decoded) = value
                            .as_str()
                            .and_then(|text| serde_json::from_str(text).ok())
                        {
                            *value = decoded;
                        }
                    }
                }
            }
            row
        })
        .collect()
}

fn row_step(row: &Value) -> Option<i64> {
    row.get("step_index").and_then(Value::as_i64)
}

fn row_is(row: &Value, kind: &str) -> bool {
    row.get("type").and_then(Value::as_str) == Some(kind)
}

/// The invocation's own model output: native code may insert system/user steps at or after
/// `initialNumSteps` before it, so it is the first planner row from there. Read at hook time,
/// when no later invocation exists yet, so it never claims another invocation's planner.
pub(crate) fn invocation_planner(rows: &[Value], initial_steps: i64) -> Option<&Value> {
    rows.iter().find(|row| {
        row_is(row, "PLANNER_RESPONSE") && row_step(row).is_some_and(|step| step >= initial_steps)
    })
}

/// A planner's tool calls paired with their outputs: only when the rows right after it are
/// exactly that many GENERIC outputs (the order of multiple calls is never guessed otherwise).
pub(crate) fn planner_tool_steps(rows: &[Value], planner: &Value) -> Vec<(Value, Value)> {
    let (Some(step), Some(calls)) = (
        row_step(planner),
        planner.get("tool_calls").and_then(Value::as_array),
    ) else {
        return Vec::new();
    };
    let outputs: Vec<&Value> = (1..=calls.len() as i64)
        .filter_map(|offset| rows.iter().find(|row| row_step(row) == Some(step + offset)))
        .filter(|row| row_is(row, "GENERIC"))
        .collect();
    if outputs.len() != calls.len() {
        return Vec::new();
    }
    calls
        .iter()
        .cloned()
        .zip(outputs.into_iter().cloned())
        .collect()
}

/// Child conversation ids from native `invoke_subagent` output ("Created the following subagents:"
/// then one JSON object per child).
pub(crate) fn spawned_children(output: &Value) -> Vec<String> {
    let Some((_, rest)) = output
        .get("content")
        .and_then(Value::as_str)
        .and_then(|content| content.split_once("Created the following subagents:"))
    else {
        return Vec::new();
    };
    serde_json::Deserializer::from_str(rest)
        .into_iter::<Value>()
        .map_while(|child| child.ok())
        .filter_map(|child| Some(child.get("conversationId")?.as_str()?.to_string()))
        .collect()
}

pub(crate) fn requested_subagents(call: &Value) -> usize {
    call.pointer("/args/Subagents")
        .and_then(Value::as_array)
        .map_or(1, Vec::len)
}

/// One conversation's final native transcript from `from_step`: accounts every planner (its hook
/// identity, else an explicit unknown), binds tool ids the hook could not pair to their own
/// planner's identity, replaces spawn placeholders with the children the output names, and rebuilds
/// each spawn as a `subagent` part with its children's tool calls (sub-agents never reach the
/// stream), recursively. A planner never inherits an unmatched hook or another invocation's model.
fn reconcile_transcript(
    ledger: &Ledger,
    root: &Path,
    conversation: &str,
    from_step: i64,
    child: bool,
) -> (Vec<WirePart>, Vec<WirePart>) {
    let rows = transcript_rows(&transcript_path(root, conversation));
    let (mut tools, mut spawns) = (Vec::new(), Vec::new());
    for planner in rows.iter().filter(|row| {
        row_is(row, "PLANNER_RESPONSE") && row_step(row).is_some_and(|step| step >= from_step)
    }) {
        let step = row_step(planner).unwrap_or_default();
        let identity = ledger.planner(conversation, step, child);
        let calls = planner_tool_steps(&rows, planner);
        for (index, (call, output)) in calls.iter().enumerate() {
            let tool_step = row_step(output).unwrap_or_default();
            if let Some(identity) = &identity {
                ledger.invoker(&tool_part_id(conversation, tool_step), identity);
            }
            let name = call.get("name").and_then(Value::as_str).unwrap_or("tool");
            if name != "invoke_subagent" {
                let (tool, input) = normalize_tool(name, call.get("args"));
                let mut part = WirePart::tool(
                    tool_part_id(conversation, tool_step),
                    tool,
                    "completed",
                    None,
                );
                if let Some(state) = part.state.as_mut() {
                    state.input = input;
                    state.output = output
                        .get("content")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
                tools.push(part);
                continue;
            }
            let children = spawned_children(output);
            ledger.spawns(
                conversation,
                step,
                index,
                &children,
                requested_subagents(call),
            );
            let mut spawn = WirePart::tool(
                format!("subagent-{conversation}-{tool_step}"),
                "subagent",
                "completed",
                None,
            );
            if let Some(state) = spawn.state.as_mut() {
                state.input = Some(serde_json::json!({ "conversationIds": children }));
            }
            for child in &children {
                let (child_tools, child_spawns) =
                    reconcile_transcript(ledger, root, child, 0, true);
                spawn.children.extend(child_tools);
                spawn.children.extend(child_spawns);
            }
            spawns.push(spawn);
        }
    }
    (tools, spawns)
}

/// Where transcript reconciliation writes: identities the hooks recorded are read back; samples go
/// to the turn's execution when one is open.
struct Ledger<'a> {
    store: &'a crate::store::Store,
    execution: Option<&'a str>,
    session: &'a str,
}

impl Ledger<'_> {
    fn planner(
        &self,
        conversation: &str,
        step: i64,
        child: bool,
    ) -> Option<crate::store::InvocationIdentity> {
        let key = invocation_sample_id(conversation, step);
        let identity = self
            .store
            .native_invocation_identity("antigravity", &key)
            .ok()
            .flatten();
        let attribution = match &identity {
            Some(identity) => crate::store::Attribution::native(
                "antigravity",
                Some(&identity.model),
                None,
                crate::store::Missing::IdentityNotReported,
            ),
            None if child => crate::store::Attribution::Unresolved {
                reason: crate::store::Missing::ChildModelUnknown,
            },
            None => crate::store::Attribution::Unresolved {
                reason: crate::store::Missing::IdentityNotReported,
            },
        };
        self.sample(&key, &attribution);
        identity
    }

    fn invoker(&self, part_id: &str, identity: &crate::store::InvocationIdentity) {
        if let Err(error) =
            self.store
                .record_native_invocation(part_id, identity, Some(self.session))
        {
            eprintln!("orx up: could not bind Antigravity tool identity: {error}");
        }
    }

    /// Named children replace the hook-time `#` placeholders for a spawn whose output flushed late.
    fn spawns(
        &self,
        conversation: &str,
        step: i64,
        index: usize,
        children: &[String],
        requested: usize,
    ) {
        let unknown = crate::store::Attribution::Unresolved {
            reason: crate::store::Missing::ChildModelUnknown,
        };
        for child in children {
            self.sample(&spawn_sample_id(conversation, step, child), &unknown);
        }
        let Some(execution) = self.execution else {
            return;
        };
        for missing in 0..requested {
            let placeholder = spawn_sample_id(conversation, step, &format!("#{index}.{missing}"));
            let result = if missing < requested.saturating_sub(children.len()) {
                self.store.record_attributed_sample(
                    execution,
                    &placeholder,
                    "antigravity",
                    &unknown,
                    &Default::default(),
                    false,
                )
            } else {
                self.store.delete_usage_sample(execution, &placeholder)
            };
            if let Err(error) = result {
                eprintln!("orx up: could not reconcile Antigravity spawn: {error}");
            }
        }
    }

    fn sample(&self, id: &str, attribution: &crate::store::Attribution) {
        let Some(execution) = self.execution else {
            return;
        };
        if let Err(error) = self.store.record_attributed_sample(
            execution,
            id,
            "antigravity",
            attribution,
            &Default::default(),
            false,
        ) {
            eprintln!("orx up: could not record Antigravity planner: {error}");
        }
    }
}

/// Final reconciliation of a turn's parent conversation: the parts the transcript adds (spawns;
/// with `all_tools`, also the parent's own tool calls a crash kept out of the stored message).
fn reconcile_turn(
    store: &crate::store::Store,
    execution: Option<&str>,
    session: &str,
    root: &Path,
    conversation: &str,
    from_step: i64,
    all_tools: bool,
) -> Vec<WirePart> {
    let ledger = Ledger {
        store,
        execution,
        session,
    };
    let (tools, mut spawns) = reconcile_transcript(&ledger, root, conversation, from_step, false);
    if all_tools {
        spawns.extend(tools);
    }
    spawns
}

const TURN_SCOPE: &str = "antigravity-turn";

/// Startup: turns a dead process left open. Their transcripts and hook identities survived, so
/// account them and restore the parts (child tool calls included) before recovery closes the
/// executions and settles runs they launched.
pub(crate) fn recover_orphaned_turns(store: &crate::store::Store) -> Result<()> {
    let Some(root) = brain_root() else {
        return Ok(());
    };
    for (execution, _, scope, orphaned) in store.native_scopes(TURN_SCOPE)? {
        if orphaned {
            recover_turn(store, &root, &execution, &scope)?;
        }
    }
    Ok(())
}

fn recover_turn(
    store: &crate::store::Store,
    root: &Path,
    execution: &str,
    scope: &Value,
) -> Result<()> {
    let (Some(session), Some(message), Some(conversation), Some(from)) = (
        scope["session"].as_str(),
        scope["message"].as_str(),
        scope["conversation"].as_str(),
        scope["from"].as_i64(),
    ) else {
        return Ok(());
    };
    let parts = reconcile_turn(
        store,
        Some(execution),
        session,
        root,
        conversation,
        from,
        true,
    );
    persist_parts(store, message, parts)
}

fn persist_parts(store: &crate::store::Store, message: &str, parts: Vec<WirePart>) -> Result<()> {
    if parts.is_empty() {
        return Ok(());
    }
    let Some(stored) = store.get_chat_message(message)? else {
        return Ok(());
    };
    let mut wire = crate::local::chat::stored_to_wire(&stored);
    for part in parts {
        if !wire.parts.iter().any(|existing| existing.id == part.id)
            || part.tool.as_deref() == Some("subagent")
        {
            crate::local::chat::upsert_preserving_children(&mut wire.parts, part);
        }
    }
    store.upsert_chat_message(&crate::store::StoredChatMessage {
        parts_json: serde_json::to_string(&wire.parts)?,
        ..stored
    })
}

/// Tool part id, unique across conversations because it keys durable invoker identities.
pub(crate) fn tool_part_id(conversation: &str, step: i64) -> String {
    format!("tool-{conversation}-{step}")
}

/// A native invocation's steps start at its planner step (`initialNumSteps`) followed only by its
/// tool steps, so any other or unobserved step in between means the invoker is unknown.
fn invoking_step(
    steps: &std::collections::HashMap<(String, i64), bool>,
    conversation: &str,
    tool_step: i64,
) -> Option<i64> {
    (0..tool_step)
        .rev()
        .map_while(|step| Some((step, *steps.get(&(conversation.to_string(), step))?)))
        .find_map(|(step, planner)| planner.then_some(step))
}

/// Tool parts whose invocation identity the PostInvocation hook has recorded, removed from `state`.
fn attributed_tools(
    state: &mut TurnState,
    identity: impl Fn(&str) -> Option<crate::store::InvocationIdentity>,
) -> Vec<(String, crate::store::InvocationIdentity)> {
    let mut found = Vec::new();
    state
        .unattributed_tools
        .retain(|(conversation, step, part_id)| {
            match invoking_step(&state.steps, conversation, *step)
                .and_then(|invocation| identity(&invocation_sample_id(conversation, invocation)))
            {
                Some(identity) => {
                    found.push((part_id.clone(), identity));
                    false
                }
                None => true,
            }
        });
    found
}

fn record_tool_invokers(session_id: &str, state: &mut TurnState) {
    if state.unattributed_tools.is_empty() {
        return;
    }
    let store = match crate::store::Store::open() {
        Ok(store) => store,
        Err(error) => {
            eprintln!("orx up: could not read native tool identity: {error}");
            return;
        }
    };
    for (part_id, identity) in attributed_tools(state, |key| {
        store
            .native_invocation_identity("antigravity", key)
            .ok()
            .flatten()
    }) {
        if let Err(error) = store.record_native_invocation(&part_id, &identity, Some(session_id)) {
            eprintln!("orx up: could not capture native tool identity: {error}");
        }
    }
}

fn antigravity_step_usage(step: &Value) -> Option<crate::store::TokenUsage> {
    let usage = step.get("usage")?;
    let field = |key| usage.get(key).and_then(Value::as_u64);
    Some(crate::store::TokenUsage {
        input_tokens: field("input_tokens"),
        output_tokens: field("output_tokens"),
        cache_read_tokens: field("cache_read_tokens"),
        cache_write_tokens: None,
        reasoning_tokens: field("thinking_tokens"),
    })
}

fn apply_event(ctx: &mut TurnCtx, state: &mut TurnState, event: &Value) -> bool {
    let event_type = event.get("event").and_then(Value::as_str).unwrap_or("");
    match event_type {
        "init" => {
            if let Some(cid) = event
                .get("conversation_id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                state.conversation_id = Some(cid.to_string());
            }
            false
        }
        "step_update" => {
            if let Some(step) = event.get("step_update") {
                // Never adopt a forwarded sub-agent step's conversation: the next turn would resume it.
                let conversation = step
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
                    .or_else(|| state.conversation_id.clone());
                if state.conversation_id.is_none() {
                    state.conversation_id.clone_from(&conversation);
                }
                let step_type = step.get("step_type").and_then(Value::as_str).unwrap_or("");
                let step_state = step.get("state").and_then(Value::as_str).unwrap_or("");
                if let Some(index) = step.get("step_index").and_then(Value::as_i64) {
                    if let ("agent_response" | "tool", Some(conversation)) =
                        (step_type, &conversation)
                    {
                        state
                            .steps
                            .insert((conversation.clone(), index), step_type == "agent_response");
                    }
                    if let Some(usage) = antigravity_step_usage(step) {
                        let sample_id = conversation
                            .as_deref()
                            .map(|conversation| invocation_sample_id(conversation, index))
                            .unwrap_or_else(|| {
                                format!("antigravity-{}:{index}", ctx.attempt_count_for_usage())
                            });
                        // The stream has no model; the PostInvocation hook's identity joins on `sample_id`.
                        ctx.record_attributed_usage(
                            &sample_id,
                            ctx.native_attribution(
                                None,
                                None,
                                crate::store::Missing::IdentityNotReported,
                            ),
                            usage,
                            step_state == "DONE",
                        );
                    }
                }

                match step_type {
                    "agent_response" => {
                        if let Some(delta) = step.get("text_delta").and_then(Value::as_str) {
                            if !delta.is_empty() {
                                let id = match &state.text_part_id {
                                    Some(id) => id.clone(),
                                    None => {
                                        state.text_seq += 1;
                                        let id = format!("text-{}", state.text_seq);
                                        ctx.upsert_part(WirePart::text(id.clone(), ""));
                                        state.text_part_id = Some(id.clone());
                                        id
                                    }
                                };
                                ctx.append_part_text(&id, delta);
                            }
                        }
                        if step_is_terminal(step_state) {
                            state.text_part_id = None;
                        }
                    }
                    "tool" => {
                        state.text_part_id = None;
                        let tool_name = step
                            .get("tool_name")
                            .and_then(Value::as_str)
                            .unwrap_or("tool");
                        let tool_info = step.get("tool_info").unwrap_or(&Value::Null);
                        let step_index =
                            step.get("step_index").and_then(Value::as_i64).unwrap_or(0);
                        let call_id = match &conversation {
                            Some(conversation) => tool_part_id(conversation, step_index),
                            None => format!("tool-{step_index}-{tool_name}"),
                        };

                        let (tool, params) = normalize_tool(tool_name, tool_info.get("parameters"));
                        let output = tool_info.get("output").and_then(Value::as_str);
                        let error =
                            tool_info
                                .get("error")
                                .and_then(error_text)
                                .or(match step_state {
                                    "ERROR" => Some("Antigravity tool failed"),
                                    "CANCELED" => Some("Antigravity tool was canceled"),
                                    _ => None,
                                });
                        let is_done = step_is_terminal(step_state) || error.is_some();

                        let status = if !is_done {
                            "running"
                        } else if error.is_some() {
                            "error"
                        } else {
                            "completed"
                        };
                        if let Some(part) = find_part_mut(&mut ctx.assistant.parts, &call_id) {
                            if let Some(part_state) = part.state.as_mut() {
                                if params.is_some() {
                                    part_state.input = params;
                                }
                                if is_done {
                                    part_state.status = status.into();
                                    if let Some(out) = output {
                                        part_state.output = Some(out.to_string());
                                    }
                                    if let Some(err) = error {
                                        part_state.error = Some(err.to_string());
                                    }
                                }
                            }
                        } else {
                            if let Some(conversation) = &conversation {
                                state.unattributed_tools.push((
                                    conversation.clone(),
                                    step_index,
                                    call_id.clone(),
                                ));
                            }
                            ctx.upsert_part(WirePart {
                                id: call_id,
                                kind: "tool".into(),
                                text: None,
                                tool: Some(tool.to_string()),
                                state: Some(WireToolState {
                                    status: status.into(),
                                    input: params,
                                    output: output.map(str::to_string),
                                    error: error.map(str::to_string),
                                    title: None,
                                }),
                                prompt: None,
                                phase: None,
                                children: Vec::new(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            false
        }
        "result" => {
            state.saw_result = true;
            let res = event.get("result").unwrap_or(&Value::Null);
            if let Some(cid) = res
                .get("conversation_id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                state.conversation_id = Some(cid.to_string());
            }
            let status = res
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("INVALID");
            if status != "SUCCESS" {
                state.turn_errored = true;
                let error = res.get("error").and_then(error_text).unwrap_or(status);
                ctx.mark_terminal_failure("antigravity_terminal", format!("Antigravity: {error}"));
            } else if let Some(error) = denied_actions_error(res) {
                ctx.mark_delivery(DeliveryState::Accepted);
                state.turn_errored = true;
                ctx.mark_terminal_failure("antigravity_permission_denied", error);
            } else {
                ctx.mark_delivery(DeliveryState::Accepted);
                ctx.mark_final_text_tail();
            }
            true
        }
        _ => false,
    }
}

fn plan_card(parts: &[WirePart], assistant_id: &str) -> Option<WirePart> {
    let last_text = parts.iter().rev().find_map(|part| {
        (part.kind == "text")
            .then_some(part.text.as_deref())
            .flatten()
            .filter(|text| !text.trim().is_empty())
    })?;
    if !super::should_synthesize_plan(true, false, false, last_text) {
        return None;
    }
    Some(WirePart::prompt(
        format!("plan-synth-{assistant_id}"),
        WirePrompt {
            kind: "plan".into(),
            plan: Some(last_text.to_string()),
            synthesized: true,
            ..Default::default()
        },
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_step_usage_includes_reasoning_without_double_counting() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/antigravity-usage.json")).unwrap();
        let usage = super::antigravity_step_usage(&fixture[0]["step_update"]).unwrap();
        usage.validate().unwrap();
        assert_eq!(usage.total(), Some(13155));
        assert_eq!(usage.reasoning_tokens, Some(138));
        assert_eq!(usage.cache_write_tokens, None);
        assert!(super::antigravity_step_usage(&serde_json::json!({})).is_none());
    }

    use super::*;
    use serde_json::json;

    fn fold(events: &[Value]) -> (TurnCtx, TurnState) {
        let mut ctx = TurnCtx::test_stub();
        let mut state = TurnState::default();
        for event in events {
            apply_event(&mut ctx, &mut state, event);
        }
        (ctx, state)
    }

    #[test]
    fn stream_folds_init_text_tools_and_result() {
        let (mut ctx, mut state) = fold(&[
            json!({
                "event": "init",
                "conversation_id": "test-conv-123",
                "init": {"tools": ["run_command", "view_file"]}
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 0,
                    "state": "DONE",
                    "step_type": "user_input"
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "state": "ACTIVE",
                    "step_type": "agent_response",
                    "text_delta": "Checking directory "
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "state": "DONE",
                    "step_type": "agent_response",
                    "text_delta": "contents..."
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 2,
                    "state": "ACTIVE",
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "tool_info": {
                        "name": "run_command",
                        "parameters": {"CommandLine": "ls -la"}
                    }
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 2,
                    "state": "DONE",
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "tool_info": {
                        "name": "run_command",
                        "parameters": {"CommandLine": "ls -la"},
                        "output": "file.txt\n"
                    }
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 3,
                    "state": "ACTIVE",
                    "step_type": "agent_response",
                    "text_delta": "Found file.txt"
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 3,
                    "state": "DONE",
                    "step_type": "agent_response",
                    "text_delta": "."
                }
            }),
        ]);

        assert_eq!(state.conversation_id.as_deref(), Some("test-conv-123"));
        assert_eq!(ctx.assistant.parts.len(), 3);
        assert_eq!(ctx.assistant.parts[0].kind, "text");
        assert_eq!(
            ctx.assistant.parts[0].text.as_deref(),
            Some("Checking directory contents...")
        );

        assert_eq!(ctx.assistant.parts[1].kind, "tool");
        let tool_state = ctx.assistant.parts[1].state.as_ref().unwrap();
        assert_eq!(tool_state.status, "completed");
        assert_eq!(ctx.assistant.parts[1].tool.as_deref(), Some("Bash"));
        assert_eq!(tool_state.input.as_ref().unwrap()["command"], "ls -la");
        assert_eq!(tool_state.output.as_deref(), Some("file.txt\n"));

        assert_eq!(ctx.assistant.parts[2].kind, "text");
        assert_eq!(
            ctx.assistant.parts[2].text.as_deref(),
            Some("Found file.txt.")
        );

        let done = apply_event(
            &mut ctx,
            &mut state,
            &json!({
                "event": "result",
                "result": {
                    "conversation_id": "test-conv-123",
                    "status": "SUCCESS",
                    "response": "Done"
                }
            }),
        );
        assert!(done);
        assert!(state.saw_result);
        assert!(!state.turn_errored);
    }

    #[test]
    fn tool_invokers_follow_native_invocation_step_ranges_only() {
        let step = |index: i64, step_type: &str, tool: Option<&str>| json!({"event":"step_update","step_update":{"step_index":index,"state":"ACTIVE","step_type":step_type,"tool_name":tool,"tool_info":{"parameters":{"CommandLine":"orx exp run e"}}}});
        // Native gen_metadata ranges: invocations produce [1,2], [3,4,5], [10,11]; 6-9 are not model steps.
        let (ctx, mut state) = fold(&[
            json!({"event":"init","conversation_id":"conv"}),
            step(0, "user_input", None),
            step(1, "agent_response", None),
            step(2, "tool", Some("run_command")),
            step(3, "agent_response", None),
            step(4, "tool", Some("run_command")),
            step(5, "tool", Some("view_file")),
            step(6, "user_input", None),
            step(8, "tool", Some("run_command")),
            step(11, "tool", Some("run_command")),
            step(10, "agent_response", None),
        ]);
        assert_eq!(ctx.assistant.parts[0].id, "tool-conv-2");
        let identity = |model: &str| crate::store::InvocationIdentity {
            harness: "antigravity".into(),
            model: model.into(),
            provider: None,
        };
        let hooked = |key: &str| match key {
            "antigravity:conv:step:1" => Some(identity("gemini-3.8-flash-high")),
            "antigravity:conv:step:3" => Some(identity("claude-sonnet-4-6")),
            _ => None,
        };
        let found: Vec<_> = attributed_tools(&mut state, hooked)
            .into_iter()
            .map(|(part, identity)| (part, identity.model))
            .collect();
        assert_eq!(
            found,
            [
                ("tool-conv-2".into(), "gemini-3.8-flash-high".into()),
                ("tool-conv-4".into(), "claude-sonnet-4-6".to_string()),
                ("tool-conv-5".into(), "claude-sonnet-4-6".into()),
            ]
        );
        // Step 8 follows a gap, and invocation 10 has no hook yet: neither borrows invocation 3's model.
        assert_eq!(state.unattributed_tools.len(), 2);
        let late = attributed_tools(&mut state, |key| {
            (key == "antigravity:conv:step:10").then(|| identity("gemini-3.1-pro-high"))
        });
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].0, "tool-conv-11");
        assert_eq!(
            state.unattributed_tools,
            [("conv".to_string(), 8, "tool-conv-8".to_string())]
        );
    }

    #[test]
    fn forwarded_subagent_steps_keep_their_own_conversation() {
        let step = |conversation: &str, index: i64, step_type: &str| json!({"event":"step_update","step_update":{"conversation_id":conversation,"step_index":index,"state":"ACTIVE","step_type":step_type,"tool_name":"run_command","tool_info":{"parameters":{"CommandLine":"orx exp run e"}}}});
        let (ctx, mut state) = fold(&[
            json!({"event":"init","conversation_id":"parent"}),
            step("parent", 1, "agent_response"),
            step("parent", 2, "tool"),
            step("child", 1, "agent_response"),
            step("child", 2, "tool"),
        ]);
        // Resuming the child instead of the parent would silently continue the wrong conversation.
        assert_eq!(state.conversation_id.as_deref(), Some("parent"));
        assert_eq!(ctx.assistant.parts[1].id, "tool-child-2");
        let found = attributed_tools(&mut state, |key| {
            (key == "antigravity:child:step:1").then(|| crate::store::InvocationIdentity {
                harness: "antigravity".into(),
                model: "gemini-3.1-pro-high".into(),
                provider: None,
            })
        });
        // The child's tool takes the child's hooked model; the parent's never borrows it.
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, "tool-child-2");
        assert_eq!(state.unattributed_tools[0].2, "tool-parent-2");
    }

    #[test]
    fn non_success_results_never_finish_the_answer() {
        for status in [
            "ERROR",
            "CANCELED",
            "INTERRUPTED",
            "INVALID",
            "WAITING",
            "RUNNING",
        ] {
            let (ctx, state) = fold(&[json!({
                "event": "result",
                "result": {"conversation_id": "", "status": status, "error": "Quota limit exceeded"}
            })]);
            assert!(state.saw_result);
            assert!(state.turn_errored, "{status}");
            assert!(state.conversation_id.is_none());
            assert!(ctx.assistant.parts.is_empty());
        }
    }

    #[test]
    fn tool_updates_preserve_inputs_and_surface_object_errors() {
        let (ctx, _) = fold(&[
            json!({"event":"step_update","step_update":{"step_index":1,"state":"ACTIVE","step_type":"tool","tool_name":"view_file","tool_info":{"parameters":{"AbsolutePath":"/repo/file.rs"}}}}),
            json!({"event":"step_update","step_update":{"step_index":1,"state":"DONE","step_type":"tool","tool_name":"view_file","tool_info":{"error":{"type":"permission","message":"Denied"}}}}),
        ]);
        let part = &ctx.assistant.parts[0];
        assert_eq!(part.tool.as_deref(), Some("Read"));
        let state = part.state.as_ref().unwrap();
        assert_eq!(state.status, "error");
        assert_eq!(state.error.as_deref(), Some("Denied"));
        assert_eq!(state.input.as_ref().unwrap()["file_path"], "/repo/file.rs");
        for name in [
            "write_to_file",
            "replace_file_content",
            "multi_replace_file_content",
        ] {
            let (_, input) = normalize_tool(name, Some(&json!({"TargetFile":"/repo/file.rs"})));
            assert_eq!(input.unwrap()["file_path"], "/repo/file.rs");
        }
    }

    #[test]
    fn permission_denial_is_not_a_successful_turn() {
        let (ctx, state) = fold(&[
            json!({"event":"step_update","step_update":{"step_index":2,"state":"ERROR","step_type":"tool","tool_name":"view_file","tool_info":{"error":{"type":"TOOL_ERROR","message":"Permission denied"}}}}),
            json!({"event":"result","result":{"status":"SUCCESS","response":"","denied_actions":[{"action":"read_file","display_name":"ViewFile"}]}}),
        ]);
        assert!(state.turn_errored);
        let tool = ctx.assistant.parts[0].state.as_ref().unwrap();
        assert_eq!(tool.status, "error");
        assert_eq!(tool.error.as_deref(), Some("Permission denied"));
    }

    #[cfg(unix)]
    #[test]
    fn cancellation_preserves_detached_experiment_supervisors() {
        let supervisor = Some(Path::new("/path with spaces/orx"));
        assert_eq!(
            turn_child(" 12 10 /bin/sh -c sleep 60", 10, supervisor),
            Some(12)
        );
        assert_eq!(turn_child(" 13 12 sleep 60", 10, supervisor), None);
        assert_eq!(
            turn_child(
                " 14 10 /path with spaces/orx supervise run-id",
                10,
                supervisor
            ),
            None
        );
        assert_eq!(
            turn_child(
                " 17 10 /bin/sh -c /path with spaces/orx supervise run-id",
                10,
                supervisor
            ),
            Some(17)
        );
        assert_eq!(
            turn_child(" 15 10 /path/orx exp run experiment-id", 10, supervisor),
            Some(15)
        );
        assert_eq!(
            turn_child(" 16 10 python -c print('supervise')", 10, supervisor),
            Some(16)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_kills_command_descendants() {
        let mut child = Command::new("sh")
            .args(["-c", "sleep 60 & echo $!; wait"])
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let cancellation = TurnProcesses(child.id());
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let pid: libc::pid_t = lines.next_line().await.unwrap().unwrap().parse().unwrap();
        drop(cancellation);
        child.wait().await.unwrap();
        for _ in 0..100 {
            if unsafe { libc::kill(pid, 0) } != 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("command descendant survived cancellation: {pid}");
    }

    #[test]
    fn approval_setup_leaves_tracked_hooks_untouched() {
        let repo = std::env::temp_dir().join(format!("orx-agy-hooks-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(repo.join(".agents")).unwrap();
        let path = repo.join(".agents/hooks.json");
        std::fs::write(&path, "{}").unwrap();
        assert!(std::process::Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(&repo)
            .status()
            .unwrap()
            .success());
        assert!(std::process::Command::new("git")
            .args(["add", ".agents/hooks.json"])
            .current_dir(&repo)
            .status()
            .unwrap()
            .success());
        assert!(write_approval_hook(&repo, true)
            .unwrap_err()
            .to_string()
            .contains("tracks .agents/hooks.json"));
        assert_eq!(std::fs::read_to_string(path).unwrap(), "{}");
        std::fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn approval_hook_respects_enabled_flag() {
        let repo =
            std::env::temp_dir().join(format!("orx-agy-hooks-flag-{}", uuid::Uuid::new_v4()));
        write_approval_hook(&repo, false).unwrap();
        let path = repo.join(".agents/hooks.json");
        let content: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(content["openresearch-approval"]["enabled"], false);
        assert_eq!(content["openresearch-accounting"]["enabled"], true);
        assert_eq!(
            content["openresearch-accounting"]["PostInvocation"]
                .as_array()
                .unwrap()
                .len(),
            1
        );

        write_approval_hook(&repo, true).unwrap();
        let content: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(content["openresearch-approval"]["enabled"], true);
        std::fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn failed_tool_states_are_terminal_even_without_error_payloads() {
        for (step_state, expected_error) in [
            ("ERROR", "Antigravity tool failed"),
            ("CANCELED", "Antigravity tool was canceled"),
        ] {
            let (ctx, _) = fold(&[json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "state": step_state,
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "tool_info": {"parameters": {"CommandLine": "false"}}
                }
            })]);
            let tool = ctx.assistant.parts[0].state.as_ref().unwrap();
            assert_eq!(tool.status, "error", "{step_state}");
            assert_eq!(tool.error.as_deref(), Some(expected_error), "{step_state}");
        }
    }

    #[test]
    fn an_error_payload_terminates_an_active_tool() {
        let (ctx, _) = fold(&[json!({
            "event": "step_update",
            "step_update": {
                "step_index": 1,
                "state": "ACTIVE",
                "step_type": "tool",
                "tool_name": "view_file",
                "tool_info": {"error": {"message": "Denied"}}
            }
        })]);
        let tool = ctx.assistant.parts[0].state.as_ref().unwrap();
        assert_eq!(tool.status, "error");
        assert_eq!(tool.error.as_deref(), Some("Denied"));
    }

    #[test]
    fn terminal_agent_response_clears_the_streamed_text_part() {
        for step_state in ["ERROR", "CANCELED"] {
            let (_, state) = fold(&[
                json!({"event":"step_update","step_update":{"step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"partial"}}),
                json!({"event":"step_update","step_update":{"step_index":1,"state":step_state,"step_type":"agent_response"}}),
            ]);
            assert!(state.text_part_id.is_none(), "{step_state}");
        }
    }

    #[test]
    fn empty_success_with_denied_actions_is_a_failed_turn() {
        let (ctx, state) = fold(&[json!({
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "response": "",
                "denied_actions": [{"action": "command", "display_name": "RunCommand"}]
            }
        })]);
        assert!(state.saw_result);
        assert!(state.turn_errored);
        assert_eq!(ctx.delivery_state(), DeliveryState::Accepted);
        assert_eq!(
            denied_actions_error(&json!({
                "response": "",
                "denied_actions": [{"display_name": "RunCommand"}]
            }))
            .as_deref(),
            Some("Antigravity denied required action(s): RunCommand")
        );
    }

    #[test]
    fn denied_actions_do_not_discard_a_nonempty_response() {
        assert!(denied_actions_error(&json!({
            "response": "I could not run it, but here is an explanation.",
            "denied_actions": [{"display_name": "RunCommand"}]
        }))
        .is_none());
    }

    #[test]
    fn parses_agy_model_list_output() {
        let sample = "Fetching available models...\n\
                      gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n\
                      gemini-3.1-pro-high    Gemini 3.1 Pro (High)\n\
                      claude-sonnet-4-6\tClaude Sonnet 4.6 (Thinking)\n";
        let models = parse_agy_model_list(sample);
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "gemini-3.8-flash-high");
        assert_eq!(
            models[0].display_name.as_deref(),
            Some("Gemini 3.8 Flash (High)")
        );
        assert_eq!(models[1].id, "gemini-3.1-pro-high");
        assert_eq!(models[2].id, "claude-sonnet-4-6");
    }

    /// Native sub-agents never reach the parent stream. A run a child launched, with a sibling
    /// running the identical command on another model, binds to the child that printed its run id.
    #[test]
    fn a_run_launched_by_a_native_child_binds_to_that_childs_model() {
        let dir = std::env::temp_dir().join(format!("orx-agy-child-run-{}", uuid::Uuid::new_v4()));
        let root = dir.join("brain");
        let write = |conversation: &str, rows: &[Value]| {
            let path = transcript_path(&root, conversation).with_file_name("transcript_full.jsonl");
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(
                path,
                rows.iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .unwrap();
        };
        let run = uuid::Uuid::new_v4().to_string();
        let other = uuid::Uuid::new_v4().to_string();
        write(
            "parent",
            &[
                serde_json::json!({"step_index":11,"type":"PLANNER_RESPONSE","tool_calls":[{"name":"invoke_subagent",
                "args":{"Subagents":[{"Model":"inherit"},{"Model":"inherit"}]}}]}),
                serde_json::json!({"step_index":12,"type":"GENERIC","content":
                "Created the following subagents:\n{\"conversationId\": \"child-a\"}\n{\"conversationId\": \"child-b\"}\nThe subagents will send you a message."}),
            ],
        );
        for (child, printed) in [("child-a", &run), ("child-b", &other)] {
            write(
                child,
                &[
                    serde_json::json!({"step_index":0,"type":"SYSTEM_MESSAGE"}),
                    serde_json::json!({"step_index":1,"type":"PLANNER_RESPONSE","tool_calls":[{"name":"run_command",
                    "args":{"CommandLine":"orx exp run exp"}}]}),
                    serde_json::json!({"step_index":2,"type":"GENERIC","content":format!("  run  {printed}\n")}),
                ],
            );
        }
        let parts = reconcile_turn(
            &crate::store::Store::open_at(dir.join("data")).unwrap(),
            None,
            "session",
            &root,
            "parent",
            11,
            false,
        );
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0]
                .children
                .iter()
                .map(|part| part.id.as_str())
                .collect::<Vec<_>>(),
            ["tool-child-a-2", "tool-child-b-2"]
        );

        let store = crate::store::Store::open_at(dir.join("data")).unwrap();
        let db = rusqlite::Connection::open(dir.join("data/orx.db")).unwrap();
        db.execute_batch(&format!(
            "INSERT INTO chat_sessions (id, project_id, harness, created_at, updated_at) VALUES ('session', 'p', 'antigravity', 1, 1);
             INSERT INTO chat_turns (id, session_id, assistant_message_id, client_turn_id, request_hash, prepared_input, settings_json, state, delivery_state, created_at, updated_at) VALUES ('turn', 'session', 'message', 'c', 'h', '', '{{}}', 'running', 'accepted', 1, 1);
             INSERT INTO chat_turn_leases (chat_session_id, claim_token, heartbeat_at) VALUES ('session', 'claim', {});",
            crate::store::now_ms()
        ))
        .unwrap();
        store
            .upsert_chat_message(&crate::store::StoredChatMessage {
                id: "message".into(),
                session_id: "session".into(),
                role: "assistant".into(),
                parts_json: serde_json::to_string(&parts).unwrap(),
                created_at: 1,
                completed_at: None,
                parent_id: None,
                base_native_session_id: None,
                result_native_session_id: None,
            })
            .unwrap();
        // What each child's PostInvocation hook records for its tool step.
        for (child, model) in [
            ("child-a", "gemini-3.1-pro-high"),
            ("child-b", "claude-sonnet-4-6"),
        ] {
            store
                .record_native_invocation(
                    &tool_part_id(child, 2),
                    &crate::store::InvocationIdentity {
                        harness: "antigravity".into(),
                        model: model.into(),
                        provider: None,
                    },
                    Some("session"),
                )
                .unwrap();
        }
        let report = (
            run.clone(),
            serde_json::json!({"events":[{"eventId":run,"properties":{"status":"failed"}}]}),
        );
        store
            .reserve_run_telemetry(
                &run,
                None,
                Some("session"),
                Some("orx exp run exp"),
                None,
                Some(&report),
            )
            .unwrap();
        store
            .upsert_run(&crate::store::StoredRun {
                id: run.clone(),
                experiment_id: "exp".into(),
                project_id: "p".into(),
                status: "starting".into(),
                backend_json: "{}".into(),
                command: String::new(),
                created_at: 1,
                updated_at: 1,
                ended_at: None,
                exit_code: None,
                commit_sha: None,
                result_markdown: None,
                cancel_requested: false,
                chat_session_id: Some("session".into()),
            })
            .unwrap();
        store
            .update_status(&run, crate::store::RunStatus::Done, Some(2), Some(0))
            .unwrap();
        let staged = store.pending_telemetry().unwrap();
        assert_eq!(staged.len(), 1);
        let properties = &staged[0].1["events"][0]["properties"];
        assert_eq!(properties["attribution"], "exact");
        assert_eq!(properties["model"], "gemini-3.1-pro-high");
        drop((db, store));
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// The hook saw its planner and the `invoke_subagent` call but no output yet; the child's hook saw
    /// its planner but not its tool output. Final (or post-crash) transcript reconciliation links the
    /// child, binds the child's tool to its own planner's model, keeps an unhooked child planner
    /// explicit, and the run the child launched is attributed exactly once.
    fn late_flush_fixture(crash: bool) {
        let dir = std::env::temp_dir().join(format!("orx-agy-late-{}", uuid::Uuid::new_v4()));
        let root = dir.join("brain");
        let write = |conversation: &str, rows: &[Value]| {
            let path = transcript_path(&root, conversation).with_file_name("transcript_full.jsonl");
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(
                &path,
                rows.iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .unwrap();
            serde_json::json!({ "transcriptPath": path })
        };
        let run = uuid::Uuid::new_v4().to_string();
        let store = crate::store::Store::open_at(dir.join("data")).unwrap();
        let db = rusqlite::Connection::open(dir.join("data/orx.db")).unwrap();
        db.execute_batch(&format!(
            "INSERT INTO chat_sessions (id, project_id, harness, created_at, updated_at) VALUES ('session', 'p', 'antigravity', 1, 1);
             INSERT INTO chat_turns (id, session_id, assistant_message_id, client_turn_id, request_hash, prepared_input, settings_json, state, delivery_state, created_at, updated_at) VALUES ('turn', 'session', 'message', 'c', 'h', '', '{{}}', 'running', 'accepted', 1, 1);
             INSERT INTO chat_turn_leases (chat_session_id, claim_token, heartbeat_at) VALUES ('session', 'claim', {});
             INSERT INTO chat_messages (id, session_id, role, parts_json, created_at) VALUES ('message', 'session', 'assistant', '[]', 1);",
            crate::store::now_ms()
        ))
        .unwrap();
        store
            .begin_usage_execution("exec", "turn", "antigravity")
            .unwrap();
        let identity = |model: &str| crate::store::InvocationIdentity {
            harness: "antigravity".into(),
            model: model.into(),
            provider: None,
        };
        let spawn = serde_json::json!({"step_index":11,"type":"PLANNER_RESPONSE","tool_calls":[
            {"name":"invoke_subagent","args":{"Subagents":[{"Model":"inherit"}]}}]});
        let parent_hook = write(
            "parent",
            &[
                serde_json::json!({"step_index":10,"type":"SYSTEM_MESSAGE"}),
                spawn.clone(),
            ],
        );
        crate::commands::mcp_gate::record_invocation(
            &store,
            &parent_hook,
            "parent",
            10,
            &identity("gemini-3.8-flash-high"),
            "session",
            Some("exec"),
        )
        .unwrap();
        let launch = serde_json::json!({"step_index":1,"type":"PLANNER_RESPONSE","tool_calls":[
            {"name":"run_command","args":{"CommandLine":"orx exp run exp"}}]});
        let child_hook = write(
            "child-a",
            &[
                serde_json::json!({"step_index":0,"type":"SYSTEM_MESSAGE"}),
                launch.clone(),
            ],
        );
        crate::commands::mcp_gate::record_invocation(
            &store,
            &child_hook,
            "child-a",
            0,
            &identity("gemini-3.1-pro-high"),
            "session",
            Some("exec"),
        )
        .unwrap();
        assert_eq!(
            store
                .native_invocation_identity("antigravity", "tool-child-a-2")
                .unwrap(),
            None
        );

        // The child launched a run; it finishes before any part names it.
        let report = (
            run.clone(),
            serde_json::json!({"events":[{"eventId":run,"properties":{"status":"failed"}}]}),
        );
        store
            .reserve_run_telemetry(
                &run,
                None,
                Some("session"),
                Some("orx exp run exp"),
                None,
                Some(&report),
            )
            .unwrap();
        store
            .upsert_run(&crate::store::StoredRun {
                id: run.clone(),
                experiment_id: "exp".into(),
                project_id: "p".into(),
                status: "starting".into(),
                backend_json: "{}".into(),
                command: String::new(),
                created_at: 1,
                updated_at: 1,
                ended_at: None,
                exit_code: None,
                commit_sha: None,
                result_markdown: None,
                cancel_requested: false,
                chat_session_id: Some("session".into()),
            })
            .unwrap();
        store
            .update_status(&run, crate::store::RunStatus::Done, Some(2), Some(0))
            .unwrap();
        assert!(store.pending_telemetry().unwrap().is_empty());

        // Final native transcripts: the spawn output names the child; the child's tool output prints
        // the run id; a later child invocation never hooked.
        write(
            "parent",
            &[
                serde_json::json!({"step_index":10,"type":"SYSTEM_MESSAGE"}),
                spawn,
                serde_json::json!({"step_index":12,"type":"GENERIC","content":
                "Created the following subagents:\n{\"conversationId\": \"child-a\"}\nThe subagents will send you a message."}),
            ],
        );
        write(
            "child-a",
            &[
                serde_json::json!({"step_index":0,"type":"SYSTEM_MESSAGE"}),
                launch,
                serde_json::json!({"step_index":2,"type":"GENERIC","content":format!("  run  {run}\n")}),
                serde_json::json!({"step_index":3,"type":"SYSTEM_MESSAGE"}),
                serde_json::json!({"step_index":4,"type":"PLANNER_RESPONSE","content":"done"}),
            ],
        );
        let scope = serde_json::json!({"session":"session","message":"message","conversation":"parent","from":10});
        if crash {
            // The process died before exit: startup reconciles from the persisted scope, then
            // recovery closes the turn and settles the run.
            db.execute("UPDATE chat_turn_leases SET heartbeat_at = 0", [])
                .unwrap();
            recover_turn(&store, &root, "exec", &scope).unwrap();
            store.reconcile_expired_unfinished_chat_turns().unwrap();
        } else {
            let parts = reconcile_turn(&store, Some("exec"), "session", &root, "parent", 10, false);
            persist_parts(&store, "message", parts).unwrap();
            db.execute_batch("UPDATE chat_turns SET state = 'completed'; UPDATE chat_messages SET completed_at = 2;").unwrap();
            store.finalize_turn_usage("turn", "done").unwrap();
            store.reconcile_run_attribution().unwrap();
        }
        store.reconcile_run_attribution().unwrap();
        let staged = store.pending_telemetry().unwrap();
        assert_eq!(staged.len(), 1, "exactly once");
        let properties = &staged[0].1["events"][0]["properties"];
        assert_eq!(properties["attribution"], "exact");
        assert_eq!(properties["model"], "gemini-3.1-pro-high");

        let samples: Vec<(String, String)> = db
            .prepare("SELECT sample_id, attribution_json FROM chat_usage_samples WHERE execution_id = 'exec' ORDER BY sample_id")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        let reason = |id: &str| {
            samples
                .iter()
                .find(|(sample, _)| sample == id)
                .map(|(_, json)| json.clone())
                .unwrap_or_default()
        };
        assert!(reason("antigravity:child-a:step:1").contains("gemini-3.1-pro-high"));
        assert!(reason("antigravity:child-a:step:4").contains("child_model_unknown"));
        assert!(reason("antigravity:parent:step:11").contains("gemini-3.8-flash-high"));
        assert!(
            reason("antigravity:parent:step:11:subagent:child-a").contains("child_model_unknown")
        );
        assert!(
            !samples.iter().any(|(id, _)| id.contains(":subagent:#")),
            "the late output replaced the hook-time placeholder"
        );

        // The next turn resumes `parent` from step 13 and is cancelled before any planner: its
        // final pass (and startup recovery) accounts nothing from earlier turns.
        use std::io::Write;
        let path = transcript_path(&root, "parent").with_file_name("transcript_full.jsonl");
        let mut transcript = std::fs::OpenOptions::new().append(true).open(path).unwrap();
        write!(
            transcript,
            "\n{}",
            serde_json::json!({"step_index":13,"type":"USER_INPUT"})
        )
        .unwrap();
        store
            .begin_usage_execution("cancelled", "turn-2", "antigravity")
            .unwrap();
        assert!(reconcile_turn(
            &store,
            Some("cancelled"),
            "session",
            &root,
            "parent",
            13,
            false
        )
        .is_empty());
        let cancelled: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM chat_usage_samples WHERE execution_id = 'cancelled'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(cancelled, 0);
        drop((db, store));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn late_flushed_child_output_reconciles_at_exit() {
        late_flush_fixture(false);
    }

    #[test]
    fn late_flushed_child_output_reconciles_after_a_crash() {
        late_flush_fixture(true);
    }
}
