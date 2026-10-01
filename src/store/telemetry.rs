use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InvocationIdentity {
    pub harness: String,
    pub model: String,
    pub provider: Option<String>,
}

fn valid_model_label(label: &str) -> bool {
    let trimmed = label.trim();
    let lower = trimmed.to_ascii_lowercase();
    !trimmed.is_empty()
        && trimmed.encode_utf16().count() <= 256
        && !label.chars().any(|c| c.is_control() || matches!(c, '\u{200b}'..='\u{200f}' | '\u{2028}'..='\u{202e}' | '\u{2060}'..='\u{206f}' | '\u{feff}'))
        && !trimmed.contains('\\')
        && !lower.starts_with("arn:")
        && !lower.contains("//")
        && !lower.starts_with("file:")
        && ![":file:", "=file:", "@file:"]
            .iter()
            .any(|prefix| lower.contains(prefix))
        && !trimmed.split(|c: char| c.is_whitespace() || [':', '=', '@', '('].contains(&c)).any(|part| {
            part.starts_with(['/', '\\', '~'])
                || part.starts_with("./")
                || part.starts_with("../")
                || part.starts_with(".\\")
                || part.starts_with("..\\")
        })
}

impl InvocationIdentity {
    pub fn validate(&self) -> Result<()> {
        if !["claude-code", "codex", "opencode", "cursor", "antigravity"]
            .contains(&self.harness.as_str())
        {
            return Err(anyhow!("Invalid invoking harness"));
        }
        for label in std::iter::once(self.model.as_str()).chain(self.provider.as_deref()) {
            if !valid_model_label(label) {
                return Err(anyhow!("Invalid invoking model/provider"));
            }
        }
        Ok(())
    }
}

/// Why a report has no exact model. The report's harness plus this reason names the missing capture path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Missing {
    /// The native stream reported usage or a tool call without a model.
    IdentityNotReported,
    /// A sub-agent or child the harness never identified; never inherits the parent's model.
    ChildModelUnknown,
    /// The native label is a path, URL, or ARN rather than a model identifier.
    InvalidLabel,
    /// Claude's locally generated `<synthetic>` error message.
    SyntheticModel,
    /// A delivered execution produced no native samples.
    NoUsageReported,
    /// A chat-launched run whose launching tool part was never found.
    InvokerNotLinked,
    /// The tool parts that could have launched a run disagree on the model.
    InvokerAmbiguous,
    /// A native agent CLI outside OpenResearch chat launched the run; it reports no model.
    ExternalAgent,
}

impl Missing {
    /// An unidentified sub-agent never inherits its parent's model.
    pub(crate) fn unidentified(child: bool) -> Self {
        if child {
            Self::ChildModelUnknown
        } else {
            Self::IdentityNotReported
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "attribution", rename_all = "snake_case")]
pub(crate) enum Attribution {
    Exact {
        model: String,
        provider: Option<String>,
    },
    AutoRouting,
    NotExecuted,
    Manual,
    Unresolved {
        reason: Missing,
    },
}

impl Attribution {
    /// Exact only from a label the harness reported natively; never pass a selected or configured model.
    pub(crate) fn native(
        harness: &str,
        model: Option<&str>,
        provider: Option<&str>,
        missing: Missing,
    ) -> Self {
        match model {
            None => Self::Unresolved { reason: missing },
            Some("<synthetic>") if harness == "claude-code" => Self::Unresolved {
                reason: Missing::SyntheticModel,
            },
            Some(model) if harness == "cursor" && model.eq_ignore_ascii_case("auto") => {
                Self::AutoRouting
            }
            Some(model) if !valid_model_label(model) => Self::Unresolved {
                reason: Missing::InvalidLabel,
            },
            Some(model) => Self::Exact {
                model: model.to_string(),
                provider: provider
                    .filter(|label| valid_model_label(label))
                    .map(str::to_string),
            },
        }
    }

    fn of(identity: &InvocationIdentity) -> Self {
        Self::native(
            &identity.harness,
            Some(&identity.model),
            identity.provider.as_deref(),
            Missing::IdentityNotReported,
        )
    }

    fn model(&self) -> (Option<&str>, Option<&str>) {
        match self {
            Self::Exact { model, provider } => (Some(model), provider.as_deref()),
            _ => (None, None),
        }
    }

    /// Writes model, provider, attribution, and attributionReason into report properties.
    fn apply(&self, properties: &mut serde_json::Value) -> Result<()> {
        let (model, provider) = self.model();
        properties["model"] = serde_json::json!(model);
        properties["provider"] = serde_json::json!(provider);
        properties["attribution"] = serde_json::to_value(self)?["attribution"].take();
        properties["attributionReason"] = match self {
            Self::Unresolved { reason } => serde_json::to_value(reason)?,
            _ => serde_json::Value::Null,
        };
        Ok(())
    }
}

/// Identifies this process as the holder of open executions; a restart makes its holds stale.
fn usage_holder() -> &'static str {
    static HOLDER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    HOLDER.get_or_init(|| uuid::Uuid::new_v4().to_string())
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunLink {
    session_id: String,
    message_id: Option<String>,
    command: String,
}

struct LaunchCandidate {
    part_id: String,
    nested: bool,
    /// The command matches the launching shell's command (or is `orx exp run` without one).
    hinted: bool,
    /// The native tool output names this run's unique ID.
    printed_run: bool,
}

fn normalized_command(command: &str) -> String {
    command
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

/// Tool parts that may have launched `run_id`. Nested parts belong to sub-agents.
fn run_launch_candidates(
    parts: &[serde_json::Value],
    hint: &str,
    run_id: &str,
    nested: bool,
    out: &mut Vec<LaunchCandidate>,
) {
    for part in parts {
        if let Some(children) = part.get("children").and_then(serde_json::Value::as_array) {
            run_launch_candidates(children, hint, run_id, true, out);
        }
        let (Some(id), Some(input)) = (
            part.get("id").and_then(serde_json::Value::as_str),
            part.pointer("/state/input")
                .and_then(serde_json::Value::as_object),
        ) else {
            continue;
        };
        let command = normalized_command(crate::local::chat::tool_command(input));
        let launches = command.contains("orx exp run");
        let hinted = if hint.is_empty() {
            launches
        } else {
            !command.is_empty() && (command.contains(hint) || hint.contains(&command))
        };
        let printed_run = (launches || hinted)
            && ["/state/output", "/state/error"].iter().any(|pointer| {
                part.pointer(pointer)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|text| text.to_ascii_lowercase().contains(run_id))
            });
        if hinted || printed_run {
            out.push(LaunchCandidate {
                part_id: id.to_string(),
                nested,
                hinted,
                printed_run,
            });
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

#[derive(Serialize, Deserialize)]
struct CumulativeBaseline {
    usage: TokenUsage,
    generation: u64,
}

impl TokenUsage {
    pub fn validate(&self) -> Result<()> {
        const MAX: u64 = 9_007_199_254_740_991;
        if self
            .counters()
            .into_iter()
            .flatten()
            .any(|value| value > MAX)
            || self.total().is_some_and(|total| total > MAX)
            || self.input_tokens.is_some_and(|input| {
                self.cache_read_tokens.unwrap_or(0) + self.cache_write_tokens.unwrap_or(0) > input
            })
            || self
                .output_tokens
                .is_some_and(|output| self.reasoning_tokens.unwrap_or(0) > output)
        {
            return Err(anyhow!("Invalid native token counters"));
        }
        Ok(())
    }

    pub fn total(&self) -> Option<u64> {
        self.input_tokens?.checked_add(self.output_tokens?)
    }

    fn counters(&self) -> [Option<u64>; 5] {
        [
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.reasoning_tokens,
        ]
    }
}

impl Store {
    pub(crate) fn native_invocation_identity(
        &self,
        harness: &str,
        call_id: &str,
    ) -> Result<Option<InvocationIdentity>> {
        let json: Option<String> = self.conn.query_row("SELECT identity_json FROM native_invocation_identities WHERE harness = ?1 AND call_id = ?2", params![harness, call_id], |row| row.get(0)).optional()?;
        json.map(|json| serde_json::from_str(&json).map_err(Into::into))
            .transpose()
    }

    pub(crate) fn record_native_invocation(
        &self,
        call_id: &str,
        identity: &InvocationIdentity,
        session_id: Option<&str>,
    ) -> Result<()> {
        identity.validate()?;
        let tx = self.begin_immediate()?;
        let owner: Option<String> = self.conn.query_row("SELECT id FROM chat_sessions WHERE harness = ?1 AND (id = ?2 OR native_session_id = ?2)", params![identity.harness, session_id], |row| row.get(0)).optional()?;
        self.conn.execute("INSERT INTO native_invocation_identities (harness, call_id, identity_json, session_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(harness, call_id) DO NOTHING", params![identity.harness, call_id, serde_json::to_string(identity)?, owner, now_ms()])?;
        if self
            .native_invocation_identity(&identity.harness, call_id)?
            .as_ref()
            != Some(identity)
        {
            return Err(anyhow!("Native tool identity changed after capture"));
        }
        tx.commit()?;
        Ok(())
    }

    /// Call inside the transaction that inserts the run. A chat-launched run without a native
    /// invoker stays pending until its launching tool part resolves; it is never reported as manual.
    pub(crate) fn reserve_run_telemetry(
        &self,
        run_id: &str,
        identity: Option<&InvocationIdentity>,
        launching_session: Option<&str>,
        tool_command: Option<&str>,
        agent_origin: Option<&str>,
        report: Option<&(String, serde_json::Value)>,
    ) -> Result<()> {
        if let Some(identity) = identity {
            identity.validate()?;
        }
        let mut report = report.map(|(_, payload)| payload.clone());
        let mut link = None;
        if let Some(payload) = report.as_mut() {
            let (harness, attribution) = match (identity, launching_session) {
                (Some(identity), _) => (Some(identity.harness.clone()), Attribution::of(identity)),
                (None, Some(session_id)) => {
                    let (harness, message_id) = self
                        .conn
                        .query_row(
                            // With no turn running, a natively woken run (no app turn) launched it:
                            // it belongs to the session's latest turn still held for descendants.
                            "SELECT s.harness, COALESCE((SELECT t.assistant_message_id FROM chat_turns t WHERE t.session_id = s.id AND t.state IN ('preparing', 'retrying', 'running') ORDER BY t.created_at DESC LIMIT 1), (SELECT t.assistant_message_id FROM chat_turns t JOIN chat_usage_executions u ON u.turn_id = t.id WHERE t.session_id = s.id AND u.outcome IS NULL AND u.held_by IS NOT NULL ORDER BY t.created_at DESC LIMIT 1)) FROM chat_sessions s WHERE s.id = ?1",
                            [session_id],
                            |row| Ok((Some(row.get(0)?), row.get(1)?)),
                        )
                        .optional()?
                        .unwrap_or((None, None));
                    link = Some(RunLink {
                        session_id: session_id.to_string(),
                        message_id,
                        command: tool_command.unwrap_or_default().to_string(),
                    });
                    (
                        harness,
                        Attribution::Unresolved {
                            reason: Missing::InvokerNotLinked,
                        },
                    )
                }
                (None, None) => match agent_origin {
                    Some(origin) => (
                        ["claude-code", "codex", "opencode", "cursor", "antigravity"]
                            .contains(&origin)
                            .then(|| origin.to_string()),
                        Attribution::Unresolved {
                            reason: Missing::ExternalAgent,
                        },
                    ),
                    None => (None, Attribution::Manual),
                },
            };
            let properties = &mut payload["events"][0]["properties"];
            properties["harness"] = serde_json::json!(harness);
            attribution.apply(properties)?;
        }
        self.conn.execute("INSERT INTO run_telemetry (run_id, identity_json, report_json, link_json) VALUES (?1, ?2, ?3, ?4) ON CONFLICT(run_id) DO NOTHING", params![run_id, identity.map(serde_json::to_string).transpose()?, report.as_ref().map(serde_json::to_string).transpose()?, link.as_ref().map(serde_json::to_string).transpose()?])?;
        self.settle_run_report(run_id, false)
    }

    pub(super) fn stage_run_terminal(&self, run_id: &str, status: RunStatus) -> Result<()> {
        let report: Option<String> = self
            .conn
            .query_row(
                "SELECT report_json FROM run_telemetry WHERE run_id = ?1",
                [run_id],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        if let Some(report) = report {
            let mut payload: serde_json::Value = serde_json::from_str(&report)?;
            payload["events"][0]["properties"]["status"] = serde_json::json!(status.as_str());
            payload["events"][0]["occurredAt"] =
                serde_json::json!(crate::telemetry::iso8601_utc(now_ms()));
            self.conn.execute(
                "UPDATE run_telemetry SET report_json = ?2, terminal = 1 WHERE run_id = ?1",
                params![run_id, serde_json::to_string(&payload)?],
            )?;
            self.settle_run_report(run_id, true)?;
        }
        Ok(())
    }

    /// Resolves pending run invokers whose evidence may have changed. Safe to repeat and to race.
    pub(crate) fn reconcile_run_attribution(&self) -> Result<()> {
        let pending = self
            .conn
            .prepare("SELECT run_id FROM run_telemetry WHERE link_json IS NOT NULL AND report_json IS NOT NULL")?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for run_id in pending {
            let tx = self.begin_immediate()?;
            self.settle_run_report(&run_id, false)?;
            tx.commit()?;
        }
        Ok(())
    }

    /// Caller holds a write transaction. Stages the terminal report exactly once: either at the
    /// terminal transition (`terminal_now`) with an already resolved invoker, or when a pending
    /// invoker resolves after the run was already terminal.
    fn settle_run_report(&self, run_id: &str, terminal_now: bool) -> Result<()> {
        let Some((report, link, terminal)) = self
            .conn
            .query_row(
                "SELECT report_json, link_json, terminal FROM run_telemetry WHERE run_id = ?1 AND report_json IS NOT NULL",
                [run_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?, row.get::<_, bool>(2)?)),
            )
            .optional()?
        else {
            return Ok(());
        };
        let mut payload: serde_json::Value = serde_json::from_str(&report)?;
        let resolved_now = if let Some(link) = link {
            let Some(attribution) =
                self.resolve_run_invoker(run_id, &serde_json::from_str(&link)?)?
            else {
                return Ok(());
            };
            attribution.apply(&mut payload["events"][0]["properties"])?;
            self.conn.execute(
                "UPDATE run_telemetry SET report_json = ?2, link_json = NULL WHERE run_id = ?1",
                params![run_id, serde_json::to_string(&payload)?],
            )?;
            true
        } else {
            false
        };
        if terminal && (terminal_now || resolved_now) {
            let id = payload["events"][0]["eventId"]
                .as_str()
                .ok_or_else(|| anyhow!("Missing run telemetry event ID"))?
                .to_string();
            self.stage_telemetry(&id, &payload)?;
        }
        Ok(())
    }

    /// The part whose native output prints the run ID is definitive at any time. Without one, only the
    /// settled turn's full set of command matches may decide, and only when they all agree.
    /// `None` while the launching turn can still persist that evidence.
    fn resolve_run_invoker(&self, run_id: &str, link: &RunLink) -> Result<Option<Attribution>> {
        let (settled, parts) = match &link.message_id {
            None => (true, None),
            Some(message_id) => self
                .conn
                .query_row(
                    "SELECT t.state IN ('completed', 'failed', 'interrupted') AND (m.completed_at IS NOT NULL OR NOT EXISTS (SELECT 1 FROM chat_turn_leases l WHERE l.chat_session_id = t.session_id)) AND NOT EXISTS (SELECT 1 FROM chat_usage_executions u WHERE u.turn_id = t.id AND u.outcome IS NULL), m.parts_json FROM chat_turns t LEFT JOIN chat_messages m ON m.id = t.assistant_message_id WHERE t.assistant_message_id = ?1",
                    [message_id],
                    |row| Ok((row.get::<_, bool>(0)?, row.get::<_, Option<String>>(1)?)),
                )
                .optional()?
                .unwrap_or((true, None)),
        };
        let harness: Option<String> = self
            .conn
            .query_row(
                "SELECT harness FROM chat_sessions WHERE id = ?1",
                [&link.session_id],
                |row| row.get(0),
            )
            .optional()?;
        let mut parts: Vec<serde_json::Value> = parts
            .map(|json| serde_json::from_str(&json))
            .transpose()?
            .unwrap_or_default();
        // Tool parts no transcript holds (natively woken runs), kept as native scopes.
        if let Some(message_id) = &link.message_id {
            let detached = self
                .conn
                .prepare("SELECT b.totals_json FROM native_usage_baselines b JOIN chat_usage_executions u ON u.execution_id = b.execution_id JOIN chat_turns t ON t.id = u.turn_id WHERE t.assistant_message_id = ?1 AND substr(b.prefix, 1, 10) = 'tool-part:'")?
                .query_map([message_id], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for json in detached {
                parts.push(serde_json::from_str(&json)?);
            }
        }
        let hint = normalized_command(&link.command);
        let mut candidates = Vec::new();
        run_launch_candidates(
            &parts,
            &hint,
            &run_id.to_ascii_lowercase(),
            false,
            &mut candidates,
        );
        let printed = candidates.iter().any(|candidate| candidate.printed_run);
        if !printed && !settled {
            return Ok(None);
        }
        let mut identities = std::collections::BTreeSet::new();
        let mut unidentified = None;
        for candidate in candidates.into_iter().filter(|candidate| {
            if printed {
                candidate.printed_run
            } else {
                candidate.hinted
            }
        }) {
            match harness
                .as_deref()
                .map(|harness| self.native_invocation_identity(harness, &candidate.part_id))
                .transpose()?
                .flatten()
            {
                Some(identity) => {
                    identities.insert(Attribution::of(&identity));
                }
                None => unidentified = Some(unidentified.unwrap_or(false) || candidate.nested),
            }
        }
        if identities.len() == 1 && unidentified.is_none() {
            return Ok(identities.pop_first());
        }
        if !settled {
            return Ok(None);
        }
        let reason = match (identities.len(), unidentified) {
            (0, None) => Missing::InvokerNotLinked,
            (0, Some(true)) => Missing::ChildModelUnknown,
            (0, Some(false)) => Missing::IdentityNotReported,
            _ => Missing::InvokerAmbiguous,
        };
        Ok(Some(Attribution::Unresolved { reason }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_cumulative_usage(
        &self,
        execution_id: &str,
        harness: &str,
        native_scope: &str,
        native_turn: &str,
        attribution: &Attribution,
        total: &TokenUsage,
        last: &TokenUsage,
    ) -> Result<()> {
        total.validate()?;
        last.validate()?;
        let prefix = format!("cumulative:{native_scope}");
        let tx = self.begin()?;
        let previous: Option<String> = tx.query_row("SELECT totals_json FROM native_usage_baselines WHERE execution_id = ?1 AND prefix = ?2", params![execution_id, prefix], |row| row.get(0)).optional()?.flatten();
        let previous: Option<CumulativeBaseline> = previous
            .map(|json| serde_json::from_str(&json))
            .transpose()?;
        if previous
            .as_ref()
            .is_some_and(|previous| &previous.usage == total)
        {
            return Ok(());
        }
        let delta = if let Some(previous) = &previous {
            TokenUsage {
                input_tokens: total
                    .input_tokens
                    .and_then(|n| n.checked_sub(previous.usage.input_tokens?)),
                output_tokens: total
                    .output_tokens
                    .and_then(|n| n.checked_sub(previous.usage.output_tokens?)),
                cache_read_tokens: total
                    .cache_read_tokens
                    .and_then(|n| n.checked_sub(previous.usage.cache_read_tokens?)),
                cache_write_tokens: total
                    .cache_write_tokens
                    .and_then(|n| n.checked_sub(previous.usage.cache_write_tokens?)),
                reasoning_tokens: total
                    .reasoning_tokens
                    .and_then(|n| n.checked_sub(previous.usage.reasoning_tokens?)),
            }
        } else {
            last.clone()
        };
        let reset =
            previous.is_some() && (delta.input_tokens.is_none() || delta.output_tokens.is_none());
        let generation =
            previous.as_ref().map_or(0, |previous| previous.generation) + u64::from(reset);
        if !reset {
            let sample_id = format!(
                "{native_scope}:{native_turn}:{generation}:{}",
                serde_json::to_string(total)?
            );
            self.record_attributed_sample(
                execution_id,
                &sample_id,
                harness,
                attribution,
                &delta,
                true,
            )?;
        }
        let baseline = CumulativeBaseline {
            usage: total.clone(),
            generation,
        };
        tx.execute("INSERT INTO native_usage_baselines (execution_id, prefix, totals_json) VALUES (?1, ?2, ?3) ON CONFLICT(execution_id, prefix) DO UPDATE SET totals_json = excluded.totals_json", params![execution_id, prefix, serde_json::to_string(&baseline)?])?;
        tx.commit()?;
        Ok(())
    }

    pub(crate) fn begin_native_usage_attempt(
        &self,
        execution_id: &str,
        prefix: &str,
        harness: &str,
        native_scope: Option<&str>,
    ) -> Result<()> {
        let baseline: Option<String> = if let Some(scope) = native_scope {
            self.conn.query_row("SELECT totals_json FROM native_usage_totals WHERE harness = ?1 AND native_scope = ?2", params![harness, scope], |row| row.get(0)).optional()?
        } else {
            Some("{}".into())
        };
        self.conn.execute("INSERT INTO native_usage_baselines (execution_id, prefix, totals_json) VALUES (?1, ?2, ?3) ON CONFLICT(execution_id, prefix) DO NOTHING", params![execution_id, prefix, baseline])?;
        Ok(())
    }

    pub(crate) fn replace_native_usage_aggregate(
        &self,
        execution_id: &str,
        prefix: &str,
        harness: &str,
        native_scope: &str,
        samples: &[(String, Option<String>, TokenUsage)],
    ) -> Result<()> {
        for (_, _, usage) in samples {
            usage.validate()?;
        }
        let tx = self.begin()?;
        let active: bool = tx.query_row("SELECT EXISTS (SELECT 1 FROM chat_usage_executions WHERE execution_id = ?1 AND outcome IS NULL)", [execution_id], |row| row.get(0))?;
        let baseline: Option<String> = tx.query_row("SELECT totals_json FROM native_usage_baselines WHERE execution_id = ?1 AND prefix = ?2", params![execution_id, prefix], |row| row.get(0)).optional()?.flatten();
        let totals: std::collections::BTreeMap<_, _> = samples
            .iter()
            .map(|(model, _, usage)| (model, usage))
            .collect();
        if active && !samples.is_empty() {
            if let Some(baseline) = baseline {
                let baseline: std::collections::BTreeMap<String, TokenUsage> =
                    serde_json::from_str(&baseline)?;
                let reset = samples.iter().any(|(model, _, usage)| {
                    baseline.get(model).is_some_and(|previous| {
                        matches!((usage.input_tokens, previous.input_tokens), (Some(now), Some(before)) if now < before)
                            || matches!((usage.output_tokens, previous.output_tokens), (Some(now), Some(before)) if now < before)
                    })
                });
                if reset {
                    tx.execute("UPDATE native_usage_baselines SET totals_json = NULL WHERE execution_id = ?1 AND prefix = ?2", params![execution_id, prefix])?;
                    tx.execute("UPDATE chat_usage_samples SET complete = 0 WHERE execution_id = ?1 AND substr(sample_id, 1, length(?2)) = ?2", params![execution_id, prefix])?;
                } else {
                    // The aggregate counts every request's tokens, so listed identities are replaced
                    // and any other identity keeps only its model (its label may differ from the key).
                    let models: Vec<_> = samples.iter().map(|(model, _, _)| model).collect();
                    tx.execute("DELETE FROM chat_usage_samples WHERE execution_id = ?1 AND substr(sample_id, 1, length(?2)) = ?2 AND (substr(sample_id, 1, length(?3)) = ?3 OR model IN (SELECT value FROM json_each(?4)))", params![execution_id, prefix, format!("{prefix}aggregate:"), serde_json::to_string(&models)?])?;
                    tx.execute("UPDATE chat_usage_samples SET usage_json = ?3, complete = 0 WHERE execution_id = ?1 AND substr(sample_id, 1, length(?2)) = ?2", params![execution_id, prefix, serde_json::to_string(&TokenUsage::default())?])?;
                    for (model, provider, usage) in samples {
                        let zero = TokenUsage {
                            input_tokens: Some(0),
                            output_tokens: Some(0),
                            cache_read_tokens: Some(0),
                            cache_write_tokens: Some(0),
                            reasoning_tokens: Some(0),
                        };
                        let previous = baseline.get(model).unwrap_or(&zero);
                        if usage == previous {
                            continue;
                        }
                        let delta = TokenUsage {
                            input_tokens: usage
                                .input_tokens
                                .and_then(|n| n.checked_sub(previous.input_tokens?)),
                            output_tokens: usage
                                .output_tokens
                                .and_then(|n| n.checked_sub(previous.output_tokens?)),
                            cache_read_tokens: usage
                                .cache_read_tokens
                                .and_then(|n| n.checked_sub(previous.cache_read_tokens?)),
                            cache_write_tokens: usage
                                .cache_write_tokens
                                .and_then(|n| n.checked_sub(previous.cache_write_tokens?)),
                            reasoning_tokens: usage
                                .reasoning_tokens
                                .and_then(|n| n.checked_sub(previous.reasoning_tokens?)),
                        };
                        let sample_id = format!("{prefix}aggregate:{model}");
                        self.record_usage_sample(
                            execution_id,
                            &sample_id,
                            harness,
                            Some(model),
                            provider.as_deref(),
                            &delta,
                        )?;
                        tx.execute("UPDATE chat_usage_samples SET complete = ?3 WHERE execution_id = ?1 AND sample_id = ?2", params![execution_id, sample_id, delta.input_tokens.is_some() && delta.output_tokens.is_some()])?;
                    }
                }
            }
            tx.execute("INSERT INTO native_usage_totals (harness, native_scope, totals_json) VALUES (?1, ?2, ?3) ON CONFLICT(harness, native_scope) DO UPDATE SET totals_json = excluded.totals_json", params![harness, native_scope, serde_json::to_string(&totals)?])?;
        }
        tx.commit()?;
        Ok(())
    }

    pub(crate) fn begin_usage_execution(
        &self,
        execution_id: &str,
        turn_id: &str,
        harness: &str,
    ) -> Result<()> {
        self.conn.execute("INSERT INTO chat_usage_executions (execution_id, turn_id, harness, report_id, suppressed) VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(execution_id) DO NOTHING", params![execution_id, turn_id, harness, uuid::Uuid::new_v4().to_string(), !crate::telemetry::accounting_reports_enabled()])?;
        Ok(())
    }

    pub(crate) fn recover_terminal_usage(&self) -> Result<()> {
        let mut stmt = self.conn.prepare("SELECT DISTINCT t.id, t.state FROM chat_usage_executions u JOIN chat_turns t ON t.id = u.turn_id WHERE u.outcome IS NULL AND t.state IN ('completed', 'failed', 'interrupted') AND NOT EXISTS (SELECT 1 FROM chat_turn_leases l WHERE l.chat_session_id = t.session_id)")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (turn_id, state) in rows {
            self.finalize_turn_usage(
                &turn_id,
                match state.as_str() {
                    "completed" => "done",
                    "interrupted" => "cancelled",
                    _ => "failed",
                },
            )?;
        }
        Ok(())
    }

    /// Closes the turn's executions. One this process holds (native descendants still running)
    /// only records the outcome and closes on release; a hold left by a dead process closes now.
    pub(crate) fn finalize_turn_usage(&self, turn_id: &str, outcome: &str) -> Result<()> {
        let executions = self
            .conn
            .prepare("SELECT execution_id FROM chat_usage_executions WHERE turn_id = ?1 AND outcome IS NULL")?
            .query_map([turn_id], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for execution_id in executions {
            let held = self.conn.execute("UPDATE chat_usage_executions SET pending_outcome = COALESCE(pending_outcome, ?2) WHERE execution_id = ?1 AND outcome IS NULL AND held_by = ?3", params![execution_id, outcome, usage_holder()])?;
            if held == 0 {
                self.finalize_usage_execution(&execution_id, outcome)?;
            }
        }
        Ok(())
    }

    /// Keeps an open execution accepting native samples after its turn ends, until
    /// [`Self::release_usage_execution`] or this process exits. False if it already closed.
    pub(crate) fn hold_usage_execution(&self, execution_id: &str) -> Result<bool> {
        Ok(self.conn.execute(
            "UPDATE chat_usage_executions SET held_by = ?2 WHERE execution_id = ?1 AND outcome IS NULL",
            params![execution_id, usage_holder()],
        )? == 1)
    }

    /// Ends a hold; closes the execution now if its turn already ended.
    pub(crate) fn release_usage_execution(&self, execution_id: &str) -> Result<()> {
        let pending: Option<Option<String>> = self.conn.query_row("UPDATE chat_usage_executions SET held_by = NULL WHERE execution_id = ?1 AND outcome IS NULL RETURNING pending_outcome", [execution_id], |row| row.get(0)).optional()?;
        if let Some(Some(outcome)) = pending {
            self.finalize_usage_execution(execution_id, &outcome)?;
            // Native descendants settled, so run invokers waiting on them can resolve now.
            self.reconcile_run_attribution()?;
        }
        Ok(())
    }

    /// Drops an open execution's placeholder sample that later native evidence superseded.
    pub(crate) fn delete_usage_sample(&self, execution_id: &str, sample_id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM chat_usage_samples WHERE execution_id = ?1 AND sample_id = ?2 AND EXISTS (SELECT 1 FROM chat_usage_executions WHERE execution_id = ?1 AND outcome IS NULL)", params![execution_id, sample_id])?;
        Ok(())
    }

    /// Durable per-execution native capture state (e.g. which rollout segment belongs to it),
    /// so a restart can finish reading evidence a dead process left behind.
    pub(crate) fn set_native_scope(
        &self,
        execution_id: &str,
        prefix: &str,
        state: &serde_json::Value,
    ) -> Result<()> {
        self.conn.execute("INSERT INTO native_usage_baselines (execution_id, prefix, totals_json) VALUES (?1, ?2, ?3) ON CONFLICT(execution_id, prefix) DO UPDATE SET totals_json = excluded.totals_json", params![execution_id, prefix, state.to_string()])?;
        Ok(())
    }

    pub(crate) fn clear_native_scope(&self, execution_id: &str, prefix: &str) -> Result<()> {
        self.conn.execute("UPDATE native_usage_baselines SET totals_json = NULL WHERE execution_id = ?1 AND prefix = ?2", params![execution_id, prefix])?;
        Ok(())
    }

    /// Native scopes under `prefix`, flagged when their execution is open and no live process
    /// holds it (a dead process left it). Call before recovery closes them.
    pub(crate) fn native_scopes(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, String, serde_json::Value, bool)>> {
        self.conn
            .prepare("SELECT b.execution_id, b.prefix, b.totals_json, u.outcome IS NULL AND (u.held_by IS NULL OR u.held_by != ?2) FROM native_usage_baselines b JOIN chat_usage_executions u ON u.execution_id = b.execution_id WHERE substr(b.prefix, 1, length(?1)) = ?1 AND b.totals_json IS NOT NULL")?
            .query_map(params![prefix, usage_holder()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, bool>(3)?,
                ))
            })?
            .map(|row| {
                let (execution, prefix, json, orphaned) = row?;
                Ok((execution, prefix, serde_json::from_str(&json)?, orphaned))
            })
            .collect()
    }

    /// The last cumulative total recorded for `native_scope` in this execution.
    pub(crate) fn cumulative_usage_total(
        &self,
        execution_id: &str,
        native_scope: &str,
    ) -> Result<Option<TokenUsage>> {
        let json: Option<String> = self.conn.query_row("SELECT totals_json FROM native_usage_baselines WHERE execution_id = ?1 AND prefix = ?2", params![execution_id, format!("cumulative:{native_scope}")], |row| row.get(0)).optional()?.flatten();
        Ok(json
            .map(|json| serde_json::from_str::<CumulativeBaseline>(&json))
            .transpose()?
            .map(|baseline| baseline.usage))
    }

    /// The open execution of a turn, for native evidence captured outside its `TurnCtx`.
    pub(crate) fn open_usage_execution(&self, turn_id: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT execution_id FROM chat_usage_executions WHERE turn_id = ?1 AND outcome IS NULL",
                [turn_id],
                |row| row.get(0),
            )
            .optional()?)
    }

    /// One report per (attribution, measured counters) group of an execution's samples.
    fn usage_report_properties(
        &self,
        execution_id: &str,
        harness: &str,
        report_id: &str,
        outcome: &str,
        delivery: Option<&str>,
    ) -> Result<Vec<serde_json::Value>> {
        let mut samples = self.conn.prepare("SELECT s.sample_id, s.model, s.provider, s.usage_json, s.complete, n.identity_json, s.attribution_json FROM chat_usage_samples s LEFT JOIN native_invocation_identities n ON n.harness = s.harness AND n.call_id = s.sample_id WHERE s.execution_id = ?1 ORDER BY s.sample_id")?;
        // A `None` attribution is a sample recorded before attribution existed; it reports as legacy.
        type Key = (
            Option<Attribution>,
            Option<String>,
            Option<String>,
            [bool; 5],
        );
        let mut rows = Vec::<(String, Key, TokenUsage, bool)>::new();
        for row in samples.query_map([&execution_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, bool>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        })? {
            let (sample_id, model, provider, json, complete, identity, attribution) = row?;
            let identity: Option<InvocationIdentity> = identity
                .map(|json| serde_json::from_str(&json))
                .transpose()?;
            let attribution: Option<Attribution> = attribution
                .map(|json| serde_json::from_str(&json))
                .transpose()?;
            // A native identity recorded under the same sample ID resolves it, child or not.
            let (attribution, model, provider) = match (attribution, identity) {
                (
                    Some(Attribution::Unresolved {
                        reason: Missing::IdentityNotReported | Missing::ChildModelUnknown,
                    }),
                    Some(identity),
                ) => (Some(Attribution::of(&identity)), None, None),
                (Some(attribution), _) => (Some(attribution), None, None),
                (None, identity) => (
                    None,
                    model.or_else(|| identity.as_ref().map(|identity| identity.model.clone())),
                    provider.or_else(|| identity.and_then(|identity| identity.provider)),
                ),
            };
            let usage: TokenUsage = serde_json::from_str(&json)?;
            let measured = usage.counters().map(|counter| counter.is_some());
            rows.push((
                sample_id,
                (attribution, model, provider, measured),
                usage,
                complete,
            ));
        }
        // Every identity-only request stays visible, except two explicit markers: a
        // `{scope}:identity[:…]` marker whose own scope has measured samples of the same
        // attribution, and a `…:subagent:{child}` spawn marker once that child's own exact
        // samples exist. A sibling's identity never covers another child's spawn.
        let covered: Vec<bool> = rows
            .iter()
            .map(|(id, key, _, _)| {
                if key.3.contains(&true) {
                    return false;
                }
                if let Some((_, child)) = id.split_once(":subagent:") {
                    let own = format!(":{child}:");
                    return rows.iter().any(|(other, other_key, _, _)| {
                        (other.contains(&own) || other.starts_with(&own[1..]))
                            && matches!(other_key.0, Some(Attribution::Exact { .. }))
                    });
                }
                id.find(":identity").is_some_and(|at| {
                    let scope = &id[..=at];
                    rows.iter().any(|(other, other_key, _, _)| {
                        other.starts_with(scope)
                            && other_key.3.contains(&true)
                            && other_key.0 == key.0
                    })
                })
            })
            .collect();
        let mut grouped = std::collections::BTreeMap::<Key, Vec<(TokenUsage, bool)>>::new();
        for ((_, key, usage, complete), covered) in rows.into_iter().zip(covered) {
            if !covered {
                grouped.entry(key).or_default().push((usage, complete));
            }
        }
        // A zero-usage synthetic error only reports when nothing else ran.
        let only_group = grouped.len() == 1;
        grouped.retain(|(attribution, _, _, _), samples| match attribution {
            Some(Attribution::Unresolved {
                reason: Missing::SyntheticModel,
            }) => {
                only_group
                    || samples
                        .iter()
                        .any(|(usage, _)| usage.counters().iter().any(|n| n.unwrap_or(0) > 0))
            }
            _ => true,
        });
        if grouped.is_empty() {
            let attribution = if matches!(delivery, Some("not_sent" | "rejected")) {
                Attribution::NotExecuted
            } else {
                Attribution::Unresolved {
                    reason: Missing::NoUsageReported,
                }
            };
            grouped.insert((Some(attribution), None, None, [false; 5]), Vec::new());
        }
        let mut reports = Vec::new();
        for ((attribution, model, provider, _), samples) in grouped {
            let sum = |field: fn(&TokenUsage) -> Option<u64>| -> Option<u64> {
                if samples.is_empty() {
                    return None;
                }
                let measured: Vec<_> = samples
                    .iter()
                    .filter_map(|(usage, _)| field(usage))
                    .collect();
                if measured.is_empty() {
                    return None;
                }
                measured
                    .into_iter()
                    .try_fold(0u64, |total, value| total.checked_add(value))
            };
            let usage = TokenUsage {
                input_tokens: sum(|u| u.input_tokens),
                output_tokens: sum(|u| u.output_tokens),
                cache_read_tokens: sum(|u| u.cache_read_tokens),
                cache_write_tokens: sum(|u| u.cache_write_tokens),
                reasoning_tokens: sum(|u| u.reasoning_tokens),
            };
            usage.validate()?;
            let mut properties = serde_json::to_value(&usage)?;
            properties["reportId"] = serde_json::json!(report_id);
            properties["harness"] = serde_json::json!(harness);
            properties["model"] = serde_json::json!(model);
            properties["provider"] = serde_json::json!(provider);
            // Legacy samples keep the old rule that complete coverage also needs a model.
            let identity_complete = match &attribution {
                Some(attribution) => {
                    attribution.apply(&mut properties)?;
                    true
                }
                None => model.is_some(),
            };
            properties["totalTokens"] = serde_json::json!(usage.total());
            properties["outcome"] = serde_json::json!(outcome);
            properties["coverage"] =
                serde_json::json!(if usage.counters().iter().all(Option::is_none) {
                    "missing"
                } else if identity_complete
                    && samples.iter().all(|(usage, complete)| *complete
                        && usage.input_tokens.is_some()
                        && usage.output_tokens.is_some())
                {
                    "complete"
                } else {
                    "partial"
                });
            reports.push(properties);
        }
        Ok(reports)
    }

    /// Snapshots, closes, and stages under one write lock, so a sample committed by another
    /// process is either in the report or rejected as late; never accepted and unreported.
    fn finalize_usage_execution(&self, execution_id: &str, outcome: &str) -> Result<()> {
        let tx = self.begin_immediate()?;
        let Some((turn_id, harness, report_id, suppressed, pending)) = tx
            .query_row(
                "SELECT turn_id, harness, report_id, suppressed, pending_outcome FROM chat_usage_executions WHERE execution_id = ?1 AND outcome IS NULL",
                [execution_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, bool>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                },
            )
            .optional()?
        else {
            return Ok(());
        };
        let outcome = pending.as_deref().unwrap_or(outcome);
        tx.execute(
            "UPDATE chat_usage_executions SET outcome = ?2, held_by = NULL WHERE execution_id = ?1",
            params![execution_id, outcome],
        )?;
        if !suppressed {
            let delivery: Option<String> = tx
                .query_row(
                    "SELECT delivery_state FROM chat_turns WHERE id = ?1",
                    [&turn_id],
                    |row| row.get(0),
                )
                .optional()?;
            for properties in self.usage_report_properties(
                execution_id,
                &harness,
                &report_id,
                outcome,
                delivery.as_deref(),
            )? {
                if let Some((id, payload)) =
                    crate::telemetry::pending_event_payload("chat_model_usage", properties)
                {
                    self.stage_telemetry(&id, &payload)?;
                }
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub(crate) fn stage_telemetry(
        &self,
        event_id: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        self.conn.execute("INSERT INTO telemetry_pending_events (event_id, payload_json) VALUES (?1, ?2) ON CONFLICT(event_id) DO NOTHING", params![event_id, serde_json::to_string(payload)?])?;
        Ok(())
    }

    pub(crate) fn pending_telemetry(&self) -> Result<Vec<(String, serde_json::Value)>> {
        let mut stmt = self.conn.prepare(
            "SELECT event_id, payload_json FROM telemetry_pending_events ORDER BY rowid LIMIT 100",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.map(|row| {
            let (id, json) = row?;
            Ok((id, serde_json::from_str(&json)?))
        })
        .collect()
    }

    pub(crate) fn acknowledge_pending_telemetry(&self, event_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM telemetry_pending_events WHERE event_id = ?1",
            [event_id],
        )?;
        Ok(())
    }

    pub(crate) fn purge_pending_telemetry(&self) -> Result<()> {
        let tx = self.begin()?;
        tx.execute("DELETE FROM telemetry_pending_events", [])?;
        tx.execute(
            "UPDATE run_telemetry SET report_json = NULL, link_json = NULL",
            [],
        )?;
        tx.execute(
            "UPDATE chat_usage_executions SET suppressed = 1 WHERE outcome IS NULL",
            [],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub(crate) fn record_usage_sample(
        &self,
        execution_id: &str,
        sample_id: &str,
        harness: &str,
        model: Option<&str>,
        provider: Option<&str>,
        usage: &TokenUsage,
    ) -> Result<()> {
        self.record_attributed_sample(
            execution_id,
            sample_id,
            harness,
            &Attribution::native(harness, model, provider, Missing::IdentityNotReported),
            usage,
            false,
        )
    }

    /// Upserts one native sample; a later sample without a model never downgrades a recorded model.
    pub(crate) fn record_attributed_sample(
        &self,
        execution_id: &str,
        sample_id: &str,
        harness: &str,
        attribution: &Attribution,
        usage: &TokenUsage,
        complete: bool,
    ) -> Result<()> {
        usage.validate()?;
        let (model, provider) = attribution.model();
        let measured = usage != &TokenUsage::default();
        // An identity-only write keeps measured counters; a measured write replaces them (snapshots, not deltas).
        self.conn.execute("INSERT INTO chat_usage_samples (execution_id, sample_id, harness, model, provider, usage_json, complete, attribution_json) SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8 WHERE EXISTS (SELECT 1 FROM chat_usage_executions WHERE execution_id = ?1 AND outcome IS NULL) ON CONFLICT(execution_id, sample_id) DO UPDATE SET usage_json = CASE WHEN ?9 THEN excluded.usage_json ELSE usage_json END, complete = CASE WHEN ?9 THEN excluded.complete ELSE complete END, model = COALESCE(excluded.model, model), provider = CASE WHEN excluded.model IS NULL AND model IS NOT NULL THEN provider ELSE excluded.provider END, attribution_json = CASE WHEN excluded.model IS NULL AND model IS NOT NULL THEN attribution_json ELSE excluded.attribution_json END", params![execution_id, sample_id, harness, model, provider, serde_json::to_string(usage)?, complete, serde_json::to_string(attribution)?, measured])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_chat_report_cannot_escape_opt_out_or_duplicate_finalization() {
        let dir = std::env::temp_dir().join(format!("orx-opt-out-race-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "claude-code")
            .unwrap();
        store
            .conn
            .execute("UPDATE chat_usage_executions SET suppressed = 0", [])
            .unwrap();
        store.purge_pending_telemetry().unwrap();
        store.finalize_usage_execution("execution", "done").unwrap();
        assert!(store.pending_telemetry().unwrap().is_empty());
        store
            .conn
            .execute("UPDATE chat_usage_executions SET suppressed = 0", [])
            .unwrap();
        store
            .finalize_usage_execution("execution", "failed")
            .unwrap();
        assert!(store.pending_telemetry().unwrap().is_empty());
        let outcome: String = store
            .conn
            .query_row("SELECT outcome FROM chat_usage_executions", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(outcome, "done");
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn complementary_partial_samples_finalize_without_losing_measured_counters() {
        let dir = std::env::temp_dir().join(format!("orx-partial-usage-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "claude-code")
            .unwrap();
        store
            .conn
            .execute("UPDATE chat_usage_executions SET suppressed = 0", [])
            .unwrap();
        for (id, usage) in [
            (
                "input",
                TokenUsage {
                    input_tokens: Some(10),
                    cache_read_tokens: Some(8),
                    ..Default::default()
                },
            ),
            (
                "cache",
                TokenUsage {
                    cache_read_tokens: Some(9),
                    ..Default::default()
                },
            ),
            (
                "output",
                TokenUsage {
                    output_tokens: Some(3),
                    reasoning_tokens: Some(2),
                    ..Default::default()
                },
            ),
            (
                "reasoning",
                TokenUsage {
                    reasoning_tokens: Some(4),
                    ..Default::default()
                },
            ),
        ] {
            store
                .record_usage_sample(
                    "execution",
                    id,
                    "claude-code",
                    Some("claude-opus-5-5"),
                    None,
                    &usage,
                )
                .unwrap();
        }
        store.finalize_turn_usage("turn", "failed").unwrap();
        store.finalize_turn_usage("turn", "failed").unwrap();
        let outcome: String = store
            .conn
            .query_row("SELECT outcome FROM chat_usage_executions", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(outcome, "failed");
        let counters: Vec<String> = store
            .conn
            .prepare("SELECT usage_json FROM chat_usage_samples")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        let cache: u64 = counters
            .iter()
            .map(|json| {
                serde_json::from_str::<TokenUsage>(json)
                    .unwrap()
                    .cache_read_tokens
                    .unwrap_or(0)
            })
            .sum();
        assert_eq!(cache, 17);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invocation_identities_follow_session_deletion_and_unowned_retention() {
        let dir =
            std::env::temp_dir().join(format!("orx-identity-retention-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        for id in ["session", "project-session"] {
            store.conn.execute("INSERT INTO chat_sessions (id, project_id, harness, created_at, updated_at) VALUES (?1, 'project', 'claude-code', 1, 1)", [id]).unwrap();
        }
        let identity = InvocationIdentity {
            harness: "claude-code".into(),
            model: "claude-opus-5-5".into(),
            provider: None,
        };
        store
            .record_native_invocation("session-call", &identity, Some("session"))
            .unwrap();
        store
            .record_native_invocation("project-call", &identity, Some("project-session"))
            .unwrap();
        store
            .record_native_invocation("legacy-call", &identity, None)
            .unwrap();
        store
            .record_native_invocation("recent-unowned", &identity, None)
            .unwrap();
        store.conn.execute("UPDATE native_invocation_identities SET created_at = 0 WHERE call_id = 'legacy-call'", []).unwrap();
        store.delete_chat_session("session").unwrap();
        assert_eq!(
            store
                .native_invocation_identity("claude-code", "session-call")
                .unwrap(),
            None
        );
        store.delete_local_project("project").unwrap();
        assert_eq!(
            store
                .native_invocation_identity("claude-code", "project-call")
                .unwrap(),
            None
        );
        drop(store);
        let store = Store::open_at(dir.clone()).unwrap();
        assert_eq!(
            store
                .native_invocation_identity("claude-code", "legacy-call")
                .unwrap(),
            None
        );
        assert_eq!(
            store
                .native_invocation_identity("claude-code", "recent-unowned")
                .unwrap(),
            Some(identity)
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn pending_reports_commit_and_survive_restart() {
        let dir = std::env::temp_dir().join(format!("orx-telemetry-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        let payload = serde_json::json!({"measured": 0});
        {
            let tx = store.begin().unwrap();
            store.stage_telemetry("rolled-back", &payload).unwrap();
            drop(tx);
        }
        assert!(store.pending_telemetry().unwrap().is_empty());
        let tx = store.begin().unwrap();
        store.stage_telemetry("committed", &payload).unwrap();
        tx.commit().unwrap();
        drop(store);
        let store = Store::open_at(dir.clone()).unwrap();
        store.stage_telemetry("committed", &payload).unwrap();
        assert_eq!(
            store.pending_telemetry().unwrap(),
            vec![("committed".into(), payload)]
        );
        store.acknowledge_pending_telemetry("committed").unwrap();
        assert!(store.pending_telemetry().unwrap().is_empty());
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn terminal_report_is_guarded_transactional_and_independent_of_run_deletion() {
        let dir = std::env::temp_dir().join(format!("orx-terminal-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        let run = StoredRun {
            id: "run".into(),
            experiment_id: "exp".into(),
            project_id: "project".into(),
            status: "starting".into(),
            backend_json: "{}".into(),
            command: "".into(),
            created_at: 1,
            updated_at: 1,
            ended_at: None,
            exit_code: None,
            commit_sha: None,
            result_markdown: None,
            cancel_requested: false,
            chat_session_id: None,
        };
        let identity = InvocationIdentity {
            harness: "codex".into(),
            model: "gpt-6-sol".into(),
            provider: Some("openai".into()),
        };
        let event_id = uuid::Uuid::new_v4().to_string();
        let report = (
            event_id.clone(),
            serde_json::json!({"events":[{"eventId":event_id,"properties":{"harness":"codex","model":"gpt-6-sol","provider":"openai","status":"failed"}}]}),
        );
        let tx = store.begin().unwrap();
        store
            .reserve_run_telemetry(&run.id, Some(&identity), None, None, None, Some(&report))
            .unwrap();
        store.upsert_run(&run).unwrap();
        tx.commit().unwrap();
        {
            let tx = store.begin().unwrap();
            assert!(store
                .update_status("run", RunStatus::Failed, Some(2), Some(1))
                .unwrap());
            assert_eq!(store.pending_telemetry().unwrap().len(), 1);
            drop(tx);
        }
        assert!(store.pending_telemetry().unwrap().is_empty());
        assert!(store
            .update_status("run", RunStatus::Cancelled, Some(3), None)
            .unwrap());
        assert!(!store
            .update_status("run", RunStatus::Done, Some(4), Some(0))
            .unwrap());
        store.conn.execute("DELETE FROM runs", []).unwrap();
        let events = store.pending_telemetry().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].1["events"][0]["properties"]["status"],
            "cancelled"
        );
        assert_eq!(events[0].1["events"][0]["properties"]["model"], "gpt-6-sol");
        store.purge_pending_telemetry().unwrap();
        assert!(store.pending_telemetry().unwrap().is_empty());
        let report: Option<String> = store
            .conn
            .query_row("SELECT report_json FROM run_telemetry", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(report.is_none());
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cumulative_baselines_exclude_resume_history_and_replace_repeated_snapshots() {
        let dir = std::env::temp_dir().join(format!("orx-baselines-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        let usage = |input, output| TokenUsage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
            reasoning_tokens: Some(0),
        };
        let first = vec![
            ("parent".into(), None, usage(100, 20)),
            ("child".into(), None, usage(30, 4)),
        ];
        store
            .begin_usage_execution("first", "turn1", "claude-code")
            .unwrap();
        store
            .begin_native_usage_attempt("first", "attempt:", "claude-code", None)
            .unwrap();
        store
            .replace_native_usage_aggregate("first", "attempt:", "claude-code", "scope", &first)
            .unwrap();
        store.finalize_turn_usage("turn1", "done").unwrap();
        store
            .begin_usage_execution("second", "turn2", "claude-code")
            .unwrap();
        store
            .begin_native_usage_attempt("second", "attempt:", "claude-code", Some("scope"))
            .unwrap();
        let second = vec![("parent".into(), None, usage(140, 29)), first[1].clone()];
        for _ in 0..2 {
            store
                .replace_native_usage_aggregate(
                    "second",
                    "attempt:",
                    "claude-code",
                    "scope",
                    &second,
                )
                .unwrap();
        }
        let samples: Vec<String> = store
            .conn
            .prepare("SELECT usage_json FROM chat_usage_samples WHERE execution_id = 'second'")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(
            serde_json::from_str::<TokenUsage>(&samples[0]).unwrap(),
            usage(40, 9)
        );
        store
            .begin_usage_execution("imported", "turn3", "claude-code")
            .unwrap();
        store
            .begin_native_usage_attempt(
                "imported",
                "attempt:",
                "claude-code",
                Some("unknown-history"),
            )
            .unwrap();
        store
            .replace_native_usage_aggregate(
                "imported",
                "attempt:",
                "claude-code",
                "unknown-history",
                &second,
            )
            .unwrap();
        let count: i64 = store
            .conn
            .query_row(
                "SELECT count(*) FROM chat_usage_samples WHERE execution_id = 'imported'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
        store
            .begin_usage_execution("reset", "turn4", "claude-code")
            .unwrap();
        store
            .begin_native_usage_attempt("reset", "attempt:", "claude-code", Some("scope"))
            .unwrap();
        store
            .record_usage_sample(
                "reset",
                "attempt:request",
                "claude-code",
                Some("parent"),
                None,
                &usage(12, 3),
            )
            .unwrap();
        store
            .replace_native_usage_aggregate(
                "reset",
                "attempt:",
                "claude-code",
                "scope",
                &[("parent".into(), None, usage(20, 4))],
            )
            .unwrap();
        let (json, complete): (String, bool) = store
            .conn
            .query_row(
                "SELECT usage_json, complete FROM chat_usage_samples WHERE execution_id = 'reset'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<TokenUsage>(&json).unwrap(),
            usage(12, 3)
        );
        assert!(!complete);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cumulative_notifications_exclude_history_deduplicate_and_handle_resets() {
        let dir = std::env::temp_dir().join(format!("orx-cumulative-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        let usage = |input, output| TokenUsage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
            reasoning_tokens: Some(0),
        };
        store
            .begin_usage_execution("execution", "turn", "codex")
            .unwrap();
        let exact = Attribution::native(
            "codex",
            Some("gpt-6-sol"),
            None,
            Missing::IdentityNotReported,
        );
        for _ in 0..2 {
            store
                .record_cumulative_usage(
                    "execution",
                    "codex",
                    "thread",
                    "native-turn",
                    &exact,
                    &usage(1100, 110),
                    &usage(100, 10),
                )
                .unwrap();
        }
        store
            .record_cumulative_usage(
                "execution",
                "codex",
                "thread",
                "native-turn",
                &exact,
                &usage(1200, 120),
                &usage(100, 10),
            )
            .unwrap();
        store
            .record_cumulative_usage(
                "execution",
                "codex",
                "thread",
                "native-turn",
                &exact,
                &usage(1000, 100),
                &usage(0, 0),
            )
            .unwrap();
        store
            .record_cumulative_usage(
                "execution",
                "codex",
                "thread",
                "native-turn",
                &exact,
                &usage(1100, 110),
                &usage(100, 10),
            )
            .unwrap();
        let samples: Vec<String> = store
            .conn
            .prepare("SELECT usage_json FROM chat_usage_samples WHERE execution_id = 'execution'")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(
            samples
                .iter()
                .map(|json| serde_json::from_str::<TokenUsage>(json)
                    .unwrap()
                    .total()
                    .unwrap())
                .sum::<u64>(),
            330
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn repeated_native_samples_replace_and_finalized_samples_do_not_change() {
        let dir = std::env::temp_dir().join(format!("orx-samples-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "claude-code")
            .unwrap();
        for output in [2, 4, 4] {
            store
                .record_usage_sample(
                    "execution",
                    "request",
                    "claude-code",
                    Some("claude-opus-5-5"),
                    Some("anthropic"),
                    &TokenUsage {
                        input_tokens: Some(10),
                        output_tokens: Some(output),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let count: i64 = store
            .conn
            .query_row("SELECT count(*) FROM chat_usage_samples", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 1);
        store.finalize_turn_usage("turn", "cancelled").unwrap();
        store
            .record_usage_sample(
                "execution",
                "request",
                "claude-code",
                Some("wrong"),
                None,
                &TokenUsage::default(),
            )
            .unwrap();
        let model: String = store
            .conn
            .query_row("SELECT model FROM chat_usage_samples", [], |row| row.get(0))
            .unwrap();
        assert_eq!(model, "claude-opus-5-5");
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn native_invocations_are_immutable_and_distinct_by_call_id() {
        let dir =
            std::env::temp_dir().join(format!("orx-native-identities-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        let parent = InvocationIdentity {
            harness: "claude-code".into(),
            model: "claude-opus-5-5".into(),
            provider: None,
        };
        let child = InvocationIdentity {
            model: "claude-haiku-4-5-20251001".into(),
            ..parent.clone()
        };
        store
            .record_native_invocation("parent-call", &parent, None)
            .unwrap();
        store
            .record_native_invocation("child-call", &child, None)
            .unwrap();
        assert!(store
            .record_native_invocation("parent-call", &child, None)
            .is_err());
        assert_eq!(
            store
                .native_invocation_identity("claude-code", "parent-call")
                .unwrap(),
            Some(parent)
        );
        assert_eq!(
            store
                .native_invocation_identity("claude-code", "child-call")
                .unwrap(),
            Some(child)
        );
        assert!(!valid_model_label("/Users/person/models/weights.gguf"));
        assert!(!valid_model_label("https://user:secret@example.com/model"));
        assert!(!valid_model_label("C:\\models\\weights.gguf"));
        assert!(!valid_model_label("models\\weights.gguf"));
        assert!(!valid_model_label(
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/private"
        ));
        assert!(valid_model_label("org/model-v1"));
        store
            .begin_usage_execution("execution", "turn", "opencode")
            .unwrap();
        let usage = TokenUsage {
            input_tokens: Some(2),
            output_tokens: Some(3),
            ..Default::default()
        };
        store
            .record_usage_sample(
                "execution",
                "sample",
                "opencode",
                Some("/private/model.gguf"),
                None,
                &usage,
            )
            .unwrap();
        let (model, counters): (Option<String>, String) = store
            .conn
            .query_row(
                "SELECT model, usage_json FROM chat_usage_samples WHERE execution_id = 'execution'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(model, None);
        assert_eq!(
            serde_json::from_str::<TokenUsage>(&counters).unwrap(),
            usage
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn token_breakdowns_are_subsets_and_unknown_is_not_zero() {
        let usage = TokenUsage {
            input_tokens: Some(100),
            output_tokens: Some(20),
            cache_read_tokens: Some(80),
            reasoning_tokens: Some(15),
            ..Default::default()
        };
        usage.validate().unwrap();
        assert_eq!(usage.total(), Some(120));
        assert_eq!(TokenUsage::default().total(), None);
        assert!(TokenUsage {
            cache_write_tokens: Some(30),
            ..usage
        }
        .validate()
        .is_err());
    }

    fn chat_fixture(harness: &str) -> (std::path::PathBuf, Store) {
        let dir = std::env::temp_dir().join(format!("orx-run-invoker-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store.conn.execute("INSERT INTO chat_sessions (id, project_id, harness, created_at, updated_at) VALUES ('session', 'project', ?1, 1, 1)", [harness]).unwrap();
        store.conn.execute("INSERT INTO chat_turns (id, session_id, assistant_message_id, client_turn_id, request_hash, prepared_input, settings_json, state, delivery_state, created_at, updated_at) VALUES ('turn', 'session', 'message', 'client', 'hash', '', '{}', 'running', 'accepted', 1, 1)", []).unwrap();
        store.conn.execute("INSERT INTO chat_messages (id, session_id, role, parts_json, created_at) VALUES ('message', 'session', 'assistant', '[]', 1)", []).unwrap();
        store.conn.execute("INSERT INTO chat_turn_leases (chat_session_id, claim_token, heartbeat_at) VALUES ('session', 'claim', ?1)", [now_ms()]).unwrap();
        (dir, store)
    }

    fn tool(id: &str, command: &str) -> serde_json::Value {
        serde_json::json!({"id": id, "type": "tool", "state": {"input": {"command": command}}})
    }

    fn set_parts(store: &Store, parts: serde_json::Value) {
        store
            .conn
            .execute(
                "UPDATE chat_messages SET parts_json = ?1 WHERE id = 'message'",
                [parts.to_string()],
            )
            .unwrap();
    }

    fn launch(store: &Store, run_id: &str, session: Option<&str>, command: &str) {
        let event_id = uuid::Uuid::new_v4().to_string();
        let report = (
            event_id.clone(),
            serde_json::json!({"events":[{"eventId":event_id,"properties":{"status":"failed"}}]}),
        );
        let tx = store.begin().unwrap();
        store
            .reserve_run_telemetry(run_id, None, session, Some(command), None, Some(&report))
            .unwrap();
        store
            .upsert_run(&StoredRun {
                id: run_id.into(),
                experiment_id: "exp".into(),
                project_id: "project".into(),
                status: "starting".into(),
                backend_json: "{}".into(),
                command: "".into(),
                created_at: 1,
                updated_at: 1,
                ended_at: None,
                exit_code: None,
                commit_sha: None,
                result_markdown: None,
                cancel_requested: false,
                chat_session_id: session.map(str::to_string),
            })
            .unwrap();
        tx.commit().unwrap();
    }

    fn staged(store: &Store) -> Vec<serde_json::Value> {
        store
            .pending_telemetry()
            .unwrap()
            .into_iter()
            .map(|(_, payload)| payload["events"][0]["properties"].clone())
            .collect()
    }

    fn codex_identity() -> InvocationIdentity {
        InvocationIdentity {
            harness: "codex".into(),
            model: "gpt-6-sol".into(),
            provider: Some("openai".into()),
        }
    }

    fn printed(id: &str, command: &str, run_id: &str) -> serde_json::Value {
        let mut part = tool(id, command);
        part["state"]["output"] = serde_json::json!(format!(
            "\u{2713} local run submitted by orx up.\n  run  {run_id}\n"
        ));
        part
    }

    #[test]
    fn run_finishing_before_its_tool_binds_waits_across_restart_and_reports_once() {
        let (dir, store) = chat_fixture("codex");
        let run = uuid::Uuid::new_v4().to_string();
        launch(&store, &run, Some("session"), "orx exp run exp");
        assert!(store
            .update_status(&run, RunStatus::Failed, Some(2), Some(1))
            .unwrap());
        store.reconcile_run_attribution().unwrap();
        assert!(staged(&store).is_empty());
        drop(store);
        let store = Store::open_at(dir.clone()).unwrap();
        assert!(store
            .reconcile_expired_unfinished_chat_turns()
            .unwrap()
            .is_empty());
        set_parts(
            &store,
            serde_json::json!([tool("call-1", "cd /repo && orx exp run exp")]),
        );
        store
            .record_native_invocation("call-1", &codex_identity(), Some("session"))
            .unwrap();
        store.reconcile_run_attribution().unwrap();
        assert!(
            staged(&store).is_empty(),
            "a command match alone must wait for the output or the settled turn"
        );
        set_parts(
            &store,
            serde_json::json!([printed("call-1", "cd /repo && orx exp run exp", &run)]),
        );
        store.reconcile_run_attribution().unwrap();
        store.reconcile_run_attribution().unwrap();
        assert!(!store
            .update_status(&run, RunStatus::Cancelled, Some(3), None)
            .unwrap());
        let reports = staged(&store);
        assert_eq!(reports.len(), 1);
        assert_eq!(
            reports[0],
            serde_json::json!({"status":"failed","harness":"codex","model":"gpt-6-sol","provider":"openai","attribution":"exact","attributionReason":null})
        );
        let (id, _) = store.pending_telemetry().unwrap().remove(0);
        store.acknowledge_pending_telemetry(&id).unwrap();
        store.reconcile_run_attribution().unwrap();
        assert!(staged(&store).is_empty());
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn run_bound_before_it_finishes_reports_exact_at_cancellation() {
        let (dir, store) = chat_fixture("codex");
        let run = uuid::Uuid::new_v4().to_string();
        let command = "orx exp run exp --backend local";
        store
            .record_native_invocation("call-1", &codex_identity(), Some("session"))
            .unwrap();
        launch(&store, &run, Some("session"), command);
        set_parts(
            &store,
            serde_json::json!([printed("call-1", command, &run)]),
        );
        store.reconcile_run_attribution().unwrap();
        assert!(staged(&store).is_empty());
        for _ in 0..2 {
            store
                .update_status(&run, RunStatus::Cancelled, Some(2), None)
                .unwrap();
        }
        let reports = staged(&store);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0]["status"], "cancelled");
        assert_eq!(reports[0]["attribution"], "exact");
        assert_eq!(reports[0]["model"], "gpt-6-sol");
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn identical_commands_from_different_models_bind_by_their_printed_run_ids() {
        let (dir, store) = chat_fixture("codex");
        let command = "orx exp run exp";
        let child_model = InvocationIdentity {
            model: "gpt-6-mini".into(),
            ..codex_identity()
        };
        store
            .record_native_invocation("parent-call", &codex_identity(), Some("session"))
            .unwrap();
        store
            .record_native_invocation("child-call", &child_model, Some("session"))
            .unwrap();
        let [parent_run, child_run, silent_run] =
            std::array::from_fn(|_| uuid::Uuid::new_v4().to_string());
        for run in [&parent_run, &child_run, &silent_run] {
            launch(&store, run, Some("session"), command);
        }
        let parts = |parent_output: bool| {
            let mut spawn = tool("spawn", "");
            spawn["children"] = serde_json::json!([printed("child-call", command, &child_run)]);
            let parent = if parent_output {
                printed("parent-call", command, &parent_run)
            } else {
                tool("parent-call", command)
            };
            serde_json::json!([spawn, parent])
        };
        set_parts(&store, serde_json::json!([tool("parent-call", command)]));
        for run in [&child_run, &parent_run, &silent_run] {
            store
                .update_status(run, RunStatus::Done, Some(2), Some(0))
                .unwrap();
        }
        assert!(staged(&store).is_empty());
        set_parts(&store, parts(false));
        store.reconcile_run_attribution().unwrap();
        assert_eq!(
            staged(&store),
            [
                serde_json::json!({"status":"done","harness":"codex","model":"gpt-6-mini","provider":"openai","attribution":"exact","attributionReason":null})
            ]
        );
        set_parts(&store, parts(true));
        store.reconcile_run_attribution().unwrap();
        assert_eq!(staged(&store).len(), 2);
        assert_eq!(staged(&store)[1]["model"], "gpt-6-sol");
        store.interrupt_chat_turn("turn").unwrap();
        store
            .conn
            .execute("UPDATE chat_messages SET completed_at = 2", [])
            .unwrap();
        store.reconcile_run_attribution().unwrap();
        let reports = staged(&store);
        assert_eq!(reports.len(), 3);
        assert_eq!(reports[2]["attributionReason"], "invoker_ambiguous");
        assert_eq!(reports[2]["model"], serde_json::Value::Null);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn external_agent_launches_are_not_manual_and_forwarded_requests_ignore_server_env() {
        let (dir, store) = chat_fixture("codex");
        for (run, origin) in [
            ("codex-run", Some("codex")),
            ("nested-run", Some("unknown")),
            ("manual-run", None),
        ] {
            let report = (
                run.to_string(),
                serde_json::json!({"events":[{"eventId":run,"properties":{"status":"failed"}}]}),
            );
            store
                .reserve_run_telemetry(run, None, None, None, origin, Some(&report))
                .unwrap();
            store.stage_run_terminal(run, RunStatus::Done).unwrap();
        }
        let reports: std::collections::BTreeMap<_, _> = store
            .pending_telemetry()
            .unwrap()
            .into_iter()
            .map(|(id, payload)| {
                let properties = &payload["events"][0]["properties"];
                (
                    id,
                    (
                        properties["harness"].clone(),
                        properties["attribution"].clone(),
                        properties["attributionReason"].clone(),
                    ),
                )
            })
            .collect();
        let expect = |harness: serde_json::Value, attribution: &str, reason: serde_json::Value| {
            (harness, serde_json::json!(attribution), reason)
        };
        assert_eq!(
            reports["codex-run"],
            expect(
                serde_json::json!("codex"),
                "unresolved",
                serde_json::json!("external_agent")
            )
        );
        assert_eq!(
            reports["nested-run"],
            expect(
                serde_json::Value::Null,
                "unresolved",
                serde_json::json!("external_agent")
            )
        );
        assert_eq!(
            reports["manual-run"],
            expect(serde_json::Value::Null, "manual", serde_json::Value::Null)
        );
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |key: &str| {
                pairs
                    .iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, value)| value.to_string())
            }
        };
        assert_eq!(crate::agent_origin(env(&[])), None);
        assert_eq!(crate::agent_origin(env(&[("CLAUDECODE", "")])), None);
        assert_eq!(
            crate::agent_origin(env(&[("CODEX_THREAD_ID", "thread")])).as_deref(),
            Some("codex")
        );
        assert_eq!(
            crate::agent_origin(env(&[("CURSOR_AGENT", "1"), ("OPENCODE", "1")])).as_deref(),
            Some("unknown")
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn identity_only_writes_never_erase_measured_usage_and_snapshots_replace() {
        let dir = std::env::temp_dir().join(format!("orx-sample-upsert-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "opencode")
            .unwrap();
        let usage = |input, output| TokenUsage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            ..Default::default()
        };
        let exact = Attribution::native(
            "opencode",
            Some("gpt-6-sol"),
            Some("openai"),
            Missing::IdentityNotReported,
        );
        let unknown = Attribution::Unresolved {
            reason: Missing::IdentityNotReported,
        };
        let record = |id: &str, attribution: &Attribution, usage: &TokenUsage, complete| {
            store
                .record_attributed_sample("execution", id, "opencode", attribution, usage, complete)
                .unwrap()
        };
        let row = |id: &str| -> (Option<String>, TokenUsage, bool) {
            let (model, json, complete): (Option<String>, String, bool) = store
                .conn
                .query_row(
                    "SELECT model, usage_json, complete FROM chat_usage_samples WHERE sample_id = ?1",
                    [id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();
            (model, serde_json::from_str(&json).unwrap(), complete)
        };
        record("tokens-first", &unknown, &usage(10, 2), true);
        record("tokens-first", &exact, &TokenUsage::default(), false);
        assert_eq!(
            row("tokens-first"),
            (Some("gpt-6-sol".into()), usage(10, 2), true)
        );
        record("identity-first", &exact, &TokenUsage::default(), false);
        record("identity-first", &unknown, &usage(7, 1), true);
        assert_eq!(
            row("identity-first"),
            (Some("gpt-6-sol".into()), usage(7, 1), true)
        );
        for (input, output) in [(3, 1), (9, 4), (9, 4)] {
            record("polled", &exact, &usage(input, output), true);
            record("polled", &exact, &TokenUsage::default(), false);
        }
        assert_eq!(row("polled"), (Some("gpt-6-sol".into()), usage(9, 4), true));
        let reports = store
            .usage_report_properties("execution", "opencode", "report", "done", Some("accepted"))
            .unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0]["totalTokens"], 12 + 8 + 13);
        assert_eq!(reports[0]["coverage"], "complete");
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn settled_turns_give_unlinked_child_and_ambiguous_runs_explicit_reasons() {
        let (dir, store) = chat_fixture("codex");
        let mut spawn = tool("spawn", "");
        spawn["children"] = serde_json::json!([tool("child-call", "orx exp run child")]);
        set_parts(
            &store,
            serde_json::json!([
                spawn,
                tool("call-a", "orx exp run twice"),
                tool("call-b", "orx exp run twice")
            ]),
        );
        store
            .record_native_invocation("call-a", &codex_identity(), Some("session"))
            .unwrap();
        for (run, command) in [
            ("unlinked", "orx exp run other"),
            ("child", "orx exp run child"),
            ("ambiguous", "orx exp run twice"),
        ] {
            launch(&store, run, Some("session"), command);
            store
                .update_status(run, RunStatus::Done, Some(2), Some(0))
                .unwrap();
        }
        store.reconcile_run_attribution().unwrap();
        assert!(staged(&store).is_empty());
        store
            .conn
            .execute("UPDATE chat_turn_leases SET heartbeat_at = 0", [])
            .unwrap();
        store.reconcile_expired_unfinished_chat_turns().unwrap();
        let mut reasons: Vec<_> = staged(&store)
            .into_iter()
            .map(|report| {
                assert_eq!(report["harness"], "codex");
                assert_eq!(report["attribution"], "unresolved");
                assert_eq!(report["model"], serde_json::Value::Null);
                report["attributionReason"].as_str().unwrap().to_string()
            })
            .collect();
        reasons.sort();
        assert_eq!(
            reasons,
            [
                "child_model_unknown",
                "invoker_ambiguous",
                "invoker_not_linked"
            ]
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn interrupted_turn_settles_pending_runs_and_manual_runs_report_directly() {
        let (dir, store) = chat_fixture("cursor");
        launch(&store, "agent", Some("session"), "orx exp run exp");
        launch(&store, "manual", None, "");
        for run in ["agent", "manual"] {
            store
                .update_status(run, RunStatus::Failed, Some(2), Some(1))
                .unwrap();
        }
        let reports = staged(&store);
        assert_eq!(reports.len(), 1);
        assert_eq!(
            reports[0],
            serde_json::json!({"status":"failed","harness":null,"model":null,"provider":null,"attribution":"manual","attributionReason":null})
        );
        store.interrupt_chat_turn("turn").unwrap();
        store
            .conn
            .execute("UPDATE chat_messages SET completed_at = 2", [])
            .unwrap();
        store.reconcile_run_attribution().unwrap();
        let reports = staged(&store);
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().any(|report| report["harness"] == "cursor"
            && report["attributionReason"] == "invoker_not_linked"));
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn usage_reports_keep_identities_tokens_and_exceptions_separate() {
        let dir = std::env::temp_dir().join(format!("orx-attribution-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "claude-code")
            .unwrap();
        let tokens = TokenUsage {
            input_tokens: Some(10),
            output_tokens: Some(2),
            ..Default::default()
        };
        let exact = |model: &str| {
            Attribution::native(
                "claude-code",
                Some(model),
                None,
                Missing::IdentityNotReported,
            )
        };
        let record = |id: &str, attribution: &Attribution, usage: &TokenUsage| {
            store
                .record_attributed_sample("execution", id, "claude-code", attribution, usage, true)
                .unwrap()
        };
        record("thread:turn:0", &exact("claude-opus-5-5"), &tokens);
        record(
            "thread:turn:0",
            &Attribution::Unresolved {
                reason: Missing::IdentityNotReported,
            },
            &tokens,
        );
        // A scoped marker is covered by its own scope's measured samples.
        record(
            "thread:turn:identity",
            &exact("claude-opus-5-5"),
            &TokenUsage::default(),
        );
        // A distinct request of the same model without counters stays visible.
        record(
            "other-request",
            &exact("claude-opus-5-5"),
            &TokenUsage::default(),
        );
        record(
            "helper-identity",
            &exact("claude-sonnet-5-5"),
            &TokenUsage::default(),
        );
        record(
            "child",
            &Attribution::Unresolved {
                reason: Missing::ChildModelUnknown,
            },
            &tokens,
        );
        store
            .record_usage_sample(
                "execution",
                "synthetic",
                "claude-code",
                Some("<synthetic>"),
                None,
                &TokenUsage {
                    input_tokens: Some(0),
                    output_tokens: Some(0),
                    ..Default::default()
                },
            )
            .unwrap();
        store
            .record_usage_sample("execution", "joined", "claude-code", None, None, &tokens)
            .unwrap();
        store
            .record_native_invocation(
                "joined",
                &InvocationIdentity {
                    harness: "claude-code".into(),
                    model: "claude-haiku-4-5-20251001".into(),
                    provider: None,
                },
                None,
            )
            .unwrap();
        store.conn.execute("INSERT INTO chat_usage_samples (execution_id, sample_id, harness, model, usage_json) VALUES ('execution', 'legacy', 'claude-code', 'legacy-model', ?1)", [serde_json::to_string(&tokens).unwrap()]).unwrap();
        let reports = store
            .usage_report_properties(
                "execution",
                "claude-code",
                "report",
                "done",
                Some("accepted"),
            )
            .unwrap();
        let find = |model: &str| {
            reports
                .iter()
                .find(|report| report["model"] == model)
                .unwrap()
                .clone()
        };
        assert_eq!(reports.len(), 6);
        let opus: Vec<_> = reports
            .iter()
            .filter(|report| report["model"] == "claude-opus-5-5")
            .map(|report| (report["coverage"].as_str(), report["inputTokens"].clone()))
            .collect();
        assert_eq!(
            opus,
            [
                (Some("missing"), serde_json::Value::Null),
                (Some("complete"), serde_json::json!(10))
            ]
        );
        let helper = find("claude-sonnet-5-5");
        assert_eq!(helper["coverage"], "missing");
        assert_eq!(helper["totalTokens"], serde_json::Value::Null);
        let child = reports
            .iter()
            .find(|report| report["attributionReason"] == "child_model_unknown")
            .unwrap();
        assert_eq!(child["model"], serde_json::Value::Null);
        assert_eq!(child["coverage"], "complete");
        assert_eq!(find("claude-haiku-4-5-20251001")["attribution"], "exact");
        let legacy = find("legacy-model");
        assert!(legacy.get("attribution").is_none());
        assert_eq!(legacy["coverage"], "partial");
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn empty_and_exceptional_executions_are_explicit() {
        let dir = std::env::temp_dir().join(format!("orx-empty-usage-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "codex")
            .unwrap();
        let only = |delivery| {
            let reports = store
                .usage_report_properties("execution", "codex", "report", "failed", delivery)
                .unwrap();
            assert_eq!(reports.len(), 1);
            assert_eq!(reports[0]["coverage"], "missing");
            (
                reports[0]["attribution"].as_str().unwrap().to_string(),
                reports[0]["attributionReason"].clone(),
            )
        };
        assert_eq!(only(Some("rejected")).0, "not_executed");
        assert_eq!(
            only(Some("unknown")),
            ("unresolved".into(), serde_json::json!("no_usage_reported"))
        );
        store
            .record_usage_sample(
                "execution",
                "synthetic",
                "claude-code",
                Some("<synthetic>"),
                None,
                &TokenUsage::default(),
            )
            .unwrap();
        assert_eq!(
            only(Some("accepted")),
            ("unresolved".into(), serde_json::json!("synthetic_model"))
        );
        assert_eq!(
            Attribution::native("cursor", Some("Auto"), None, Missing::IdentityNotReported),
            Attribution::AutoRouting
        );
        assert_eq!(
            Attribution::native(
                "opencode",
                Some("/models/x.gguf"),
                None,
                Missing::ChildModelUnknown
            ),
            Attribution::Unresolved {
                reason: Missing::InvalidLabel
            }
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn sample_ids(store: &Store, execution: &str) -> Vec<String> {
        store
            .conn
            .prepare("SELECT sample_id FROM chat_usage_samples WHERE execution_id = ?1 ORDER BY sample_id")
            .unwrap()
            .query_map([execution], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap()
    }

    fn outcome(store: &Store, execution: &str) -> Option<String> {
        store
            .conn
            .query_row(
                "SELECT outcome FROM chat_usage_executions WHERE execution_id = ?1",
                [execution],
                |row| row.get(0),
            )
            .unwrap()
    }

    #[test]
    fn held_executions_accept_late_descendants_and_close_once_on_release_or_restart() {
        let (dir, store) = chat_fixture("codex");
        let tokens = TokenUsage {
            input_tokens: Some(5),
            output_tokens: Some(1),
            ..Default::default()
        };
        let child = Attribution::native(
            "codex",
            Some("gpt-6-mini"),
            None,
            Missing::ChildModelUnknown,
        );
        for execution in ["held", "orphaned"] {
            store
                .begin_usage_execution(execution, "turn", "codex")
                .unwrap();
        }
        assert!(store.hold_usage_execution("held").unwrap());
        assert!(store.hold_usage_execution("orphaned").unwrap());
        // A dead process's hold must not keep an execution open across restart.
        store
            .conn
            .execute(
                "UPDATE chat_usage_executions SET held_by = 'dead-process' WHERE execution_id = 'orphaned'",
                [],
            )
            .unwrap();
        store.interrupt_chat_turn("turn").unwrap();
        store.finalize_turn_usage("turn", "cancelled").unwrap();
        assert_eq!(outcome(&store, "held"), None);
        assert_eq!(outcome(&store, "orphaned").as_deref(), Some("cancelled"));
        // The spawning turn is over; its sub-agent's request still lands in its execution.
        store
            .record_attributed_sample("held", "child:c:request:6", "codex", &child, &tokens, true)
            .unwrap();
        store
            .conn
            .execute("DELETE FROM chat_turn_leases", [])
            .unwrap();
        store.recover_terminal_usage().unwrap();
        assert_eq!(outcome(&store, "held"), None, "a live holder keeps it open");
        store.release_usage_execution("held").unwrap();
        store.release_usage_execution("held").unwrap();
        assert_eq!(outcome(&store, "held").as_deref(), Some("cancelled"));
        assert_eq!(sample_ids(&store, "held"), ["child:c:request:6"]);
        store
            .record_attributed_sample("held", "child:c:request:9", "codex", &child, &tokens, true)
            .unwrap();
        assert_eq!(sample_ids(&store, "held"), ["child:c:request:6"]);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_sample_racing_finalization_is_reported_or_rejected_never_silently_kept() {
        let dir = std::env::temp_dir().join(format!("orx-final-race-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "antigravity")
            .unwrap();
        // Finalization's snapshot and close happen under one write lock; a hook process
        // writing meanwhile waits and then finds the execution closed.
        let tx = store.begin_immediate().unwrap();
        let hook = {
            let dir = dir.clone();
            std::thread::spawn(move || {
                let hook = Store::open_at(dir).unwrap();
                hook.record_attributed_sample(
                    "execution",
                    "antigravity:conv:step:3",
                    "antigravity",
                    &Attribution::native(
                        "antigravity",
                        Some("gemini-3.1-pro-high"),
                        None,
                        Missing::IdentityNotReported,
                    ),
                    &TokenUsage::default(),
                    false,
                )
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(200));
        store
            .conn
            .execute("UPDATE chat_usage_executions SET outcome = 'done'", [])
            .unwrap();
        tx.commit().unwrap();
        hook.join().unwrap().unwrap();
        assert!(sample_ids(&store, "execution").is_empty());
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Runs the real finalization while a hook's sample is in flight, then writes a late sample.
    fn finalize_during_an_in_flight_sample(dir: &std::path::Path) -> Store {
        let store = Store::open_at(dir.to_path_buf()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "antigravity")
            .unwrap();
        let sample = |store: &Store, id: &str| {
            store.record_attributed_sample(
                "execution",
                id,
                "antigravity",
                &Attribution::native(
                    "antigravity",
                    Some("gemini-3.1-pro-high"),
                    None,
                    Missing::IdentityNotReported,
                ),
                &TokenUsage {
                    input_tokens: Some(3),
                    output_tokens: Some(1),
                    ..Default::default()
                },
                true,
            )
        };
        // A hook process is mid-write when the real finalization starts in another process.
        let hook = Store::open_at(dir.to_path_buf()).unwrap();
        let finalizer = Store::open_at(dir.to_path_buf()).unwrap();
        let tx = hook.begin_immediate().unwrap();
        sample(&hook, "antigravity:conv:step:1").unwrap();
        let finalizer =
            std::thread::spawn(move || finalizer.finalize_usage_execution("execution", "done"));
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(
            !finalizer.is_finished(),
            "finalization waits for the writer"
        );
        tx.commit().unwrap();
        finalizer.join().unwrap().unwrap();
        sample(&hook, "antigravity:conv:step:2").unwrap();
        assert_eq!(sample_ids(&store, "execution"), ["antigravity:conv:step:1"]);
        assert_eq!(outcome(&store, "execution").as_deref(), Some("done"));
        store
    }

    #[test]
    fn finalization_snapshots_after_an_in_flight_sample_commits_and_rejects_later_ones() {
        let dir = std::env::temp_dir().join(format!("orx-final-order-{}", uuid::Uuid::new_v4()));
        let store = finalize_during_an_in_flight_sample(&dir);
        if crate::telemetry::build_channel() != "production" {
            assert!(store.pending_telemetry().unwrap().is_empty());
        }
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Official builds only; run alone:
    /// `ORX_OFFICIAL_RELEASE_BUILD=1 GITHUB_ACTIONS=true GITHUB_REPOSITORY=alphaXiv/OpenResearch cargo test --locked --bin orx store::telemetry::tests::official_finalization_reports_the_in_flight_sample_not_the_late_one -- --ignored --exact --test-threads=1`
    #[test]
    #[ignore = "official-build accounting gate; see the invocation above"]
    fn official_finalization_reports_the_in_flight_sample_not_the_late_one() {
        assert_eq!(crate::telemetry::build_channel(), "production");
        let _env = crate::telemetry::tests::EnvGuard::new(&[
            "XDG_CONFIG_HOME",
            "ORX_DATA_DIR",
            "ORX_TELEMETRY_ENV",
            "ORX_TELEMETRY_HOST",
        ]);
        let dir = std::env::temp_dir().join(format!("orx-official-final-{}", uuid::Uuid::new_v4()));
        std::env::set_var("XDG_CONFIG_HOME", dir.join("config"));
        // Any stray default-store open lands here, never in the user's data dir.
        std::env::set_var("ORX_DATA_DIR", dir.join("default-data"));
        // Any send lands on this loopback sink, never production; none is expected.
        let receiver = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        receiver.set_nonblocking(true).unwrap();
        std::env::set_var(
            "ORX_TELEMETRY_HOST",
            format!("http://{}", receiver.local_addr().unwrap()),
        );
        assert!(crate::telemetry::accounting_reports_enabled());

        let store = finalize_during_an_in_flight_sample(&dir.join("data"));
        let staged = store.pending_telemetry().unwrap();
        assert_eq!(staged.len(), 1, "one report for the one model");
        let event = &staged[0].1["events"][0];
        assert_eq!(event["name"], "cli_chat_model_usage");
        let properties = &event["properties"];
        assert_eq!(properties["model"], "gemini-3.1-pro-high");
        assert_eq!(properties["attribution"], "exact");
        assert_eq!(properties["outcome"], "done");
        // Exactly the in-flight sample's counters: the late sample was rejected, not added.
        assert_eq!(properties["inputTokens"], 3);
        assert_eq!(properties["outputTokens"], 1);
        assert_eq!(properties["coverage"], "complete");
        assert!(
            matches!(receiver.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "finalization only stages; nothing is sent"
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn claude_aggregates_replace_only_listed_identities_per_attempt() {
        let dir = std::env::temp_dir().join(format!("orx-claude-agg-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("e", "t", "claude-code")
            .unwrap();
        store
            .begin_native_usage_attempt("e", "claude-1:", "claude-code", None)
            .unwrap();
        let exact = |model: &str| {
            Attribution::native(
                "claude-code",
                Some(model),
                None,
                Missing::IdentityNotReported,
            )
        };
        let usage = |input, output| TokenUsage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            ..Default::default()
        };
        let record = |id: &str, attribution: &Attribution, usage: &TokenUsage| {
            store
                .record_attributed_sample("e", id, "claude-code", attribution, usage, false)
                .unwrap()
        };
        record("claude-1:msg_p", &exact("claude-opus-5-5"), &usage(10, 5));
        record(
            "claude-1:msg_c",
            &exact("claude-haiku-4-5-20251001"),
            &usage(3, 1),
        );
        record(
            "claude-1:msg_s",
            &exact("claude-sonnet-5-5"),
            &TokenUsage::default(),
        );
        record(
            "claude-1:msg_u",
            &Attribution::Unresolved {
                reason: Missing::ChildModelUnknown,
            },
            &usage(2, 1),
        );
        // Attempt 2 was interrupted: no modelUsage, so its messages keep their own counters.
        record("claude-2:msg_q", &exact("claude-opus-5-5"), &usage(4, 2));
        let full = |input, output| TokenUsage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
            reasoning_tokens: Some(0),
        };
        store
            .replace_native_usage_aggregate(
                "e",
                "claude-1:",
                "claude-code",
                "scope",
                &[("claude-opus-5-5".into(), None, full(15, 7))],
            )
            .unwrap();
        let reports = store
            .usage_report_properties("e", "claude-code", "report", "done", Some("accepted"))
            .unwrap();
        let mut summary: Vec<_> = reports
            .iter()
            .map(|report| {
                (
                    report["model"]
                        .as_str()
                        .unwrap_or(report["attributionReason"].as_str().unwrap_or_default()),
                    report["coverage"].as_str().unwrap(),
                    report["totalTokens"].to_string(),
                )
            })
            .collect();
        summary.sort();
        assert_eq!(
            summary,
            [
                ("child_model_unknown", "missing", "null".into()),
                ("claude-haiku-4-5-20251001", "missing", "null".into()),
                ("claude-opus-5-5", "complete", "22".into()),
                ("claude-opus-5-5", "partial", "6".into()),
                ("claude-sonnet-5-5", "missing", "null".into()),
            ]
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_native_identity_under_the_same_sample_resolves_an_unknown_child() {
        let dir = std::env::temp_dir().join(format!("orx-child-join-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("e", "t", "antigravity")
            .unwrap();
        let unknown = Attribution::Unresolved {
            reason: Missing::ChildModelUnknown,
        };
        for id in ["antigravity:child:step:1", "antigravity:other:step:1"] {
            store
                .record_attributed_sample(
                    "e",
                    id,
                    "antigravity",
                    &unknown,
                    &TokenUsage {
                        input_tokens: Some(3),
                        output_tokens: Some(1),
                        ..Default::default()
                    },
                    true,
                )
                .unwrap();
        }
        store
            .record_native_invocation(
                "antigravity:child:step:1",
                &InvocationIdentity {
                    harness: "antigravity".into(),
                    model: "gemini-3.8-flash-high".into(),
                    provider: None,
                },
                None,
            )
            .unwrap();
        let reports = store
            .usage_report_properties("e", "antigravity", "report", "done", Some("accepted"))
            .unwrap();
        let mut attributions: Vec<_> = reports
            .iter()
            .map(|report| (report["attribution"].clone(), report["model"].clone()))
            .collect();
        attributions.sort_by_key(|pair| pair.0.to_string());
        assert_eq!(
            attributions,
            [
                (
                    serde_json::json!("exact"),
                    serde_json::json!("gemini-3.8-flash-high")
                ),
                (serde_json::json!("unresolved"), serde_json::Value::Null),
            ]
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn report_summary(
        store: &Store,
        execution: &str,
        harness: &str,
    ) -> Vec<(String, String, String)> {
        let mut summary: Vec<_> = store
            .usage_report_properties(execution, harness, "report", "done", Some("accepted"))
            .unwrap()
            .iter()
            .map(|report| {
                (
                    report["model"]
                        .as_str()
                        .or(report["attributionReason"].as_str())
                        .unwrap_or_default()
                        .to_string(),
                    report["coverage"].as_str().unwrap().to_string(),
                    report["totalTokens"].to_string(),
                )
            })
            .collect();
        summary.sort();
        summary
    }

    #[test]
    fn spawn_markers_resolve_only_through_their_own_childs_identity() {
        let dir = std::env::temp_dir().join(format!("orx-spawns-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("e", "t", "antigravity")
            .unwrap();
        let unknown = Attribution::Unresolved {
            reason: Missing::ChildModelUnknown,
        };
        let record = |id: &str, attribution: &Attribution| {
            store
                .record_attributed_sample(
                    "e",
                    id,
                    "antigravity",
                    attribution,
                    &TokenUsage::default(),
                    false,
                )
                .unwrap()
        };
        for child in ["child-a", "child-b", "#0.0"] {
            record(
                &format!("antigravity:parent:step:11:subagent:{child}"),
                &unknown,
            );
        }
        record(
            "antigravity:child-a:step:1",
            &Attribution::native(
                "antigravity",
                Some("gemini-3.1-pro-high"),
                None,
                Missing::IdentityNotReported,
            ),
        );
        assert_eq!(
            report_summary(&store, "e", "antigravity"),
            [
                (
                    "child_model_unknown".into(),
                    "missing".into(),
                    "null".into()
                ),
                (
                    "gemini-3.1-pro-high".into(),
                    "missing".into(),
                    "null".into()
                ),
            ]
        );
        // Without child-b's and the unnamed child's markers, child-a's identity covers everything.
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_rerouted_away_model_keeps_its_identity_and_only_its_own_scope_covers_a_marker() {
        let dir = std::env::temp_dir().join(format!("orx-reroute-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store.begin_usage_execution("e", "t", "codex").unwrap();
        let exact = |model: &str| {
            Attribution::native("codex", Some(model), None, Missing::IdentityNotReported)
        };
        for model in ["gpt-6-sol", "gpt-6-astra"] {
            store
                .record_attributed_sample(
                    "e",
                    &format!("thread:turn:identity:{model}"),
                    "codex",
                    &exact(model),
                    &TokenUsage::default(),
                    false,
                )
                .unwrap();
        }
        let usage = |input, output| TokenUsage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
            reasoning_tokens: Some(0),
        };
        store
            .record_cumulative_usage(
                "e",
                "codex",
                "thread",
                "turn",
                &exact("gpt-6-astra"),
                &usage(40, 4),
                &usage(40, 4),
            )
            .unwrap();
        // Another turn's measured samples never cover this turn's marker.
        store
            .record_cumulative_usage(
                "e",
                "codex",
                "thread",
                "other",
                &exact("gpt-6-sol"),
                &usage(90, 9),
                &usage(50, 5),
            )
            .unwrap();
        assert_eq!(
            report_summary(&store, "e", "codex"),
            [
                ("gpt-6-astra".into(), "complete".into(), "44".into()),
                ("gpt-6-sol".into(), "complete".into(), "55".into()),
                ("gpt-6-sol".into(), "missing".into(), "null".into()),
            ]
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn held_descendants_defer_fallback_settlement_but_not_a_printed_run_id() {
        let (dir, store) = chat_fixture("opencode");
        store
            .begin_usage_execution("exec", "turn", "opencode")
            .unwrap();
        assert!(store.hold_usage_execution("exec").unwrap());
        let (printed, silent) = (
            uuid::Uuid::new_v4().to_string(),
            uuid::Uuid::new_v4().to_string(),
        );
        let mut part = tool("call-1", "orx exp run exp");
        part["state"]["output"] = serde_json::json!(format!("  run  {printed}\n"));
        set_parts(&store, serde_json::json!([part]));
        store
            .record_native_invocation(
                "call-1",
                &InvocationIdentity {
                    harness: "opencode".into(),
                    model: "big-pickle".into(),
                    provider: Some("opencode".into()),
                },
                Some("session"),
            )
            .unwrap();
        for run in [&printed, &silent] {
            launch(&store, run, Some("session"), "orx exp run other");
            store
                .update_status(run, RunStatus::Done, Some(2), Some(0))
                .unwrap();
        }
        // The turn ended; its execution stays held for background descendants.
        store.conn.execute_batch("UPDATE chat_turns SET state = 'completed'; UPDATE chat_messages SET completed_at = 2; DELETE FROM chat_turn_leases;").unwrap();
        store.finalize_turn_usage("turn", "done").unwrap();
        store.reconcile_run_attribution().unwrap();
        let staged = staged(&store);
        assert_eq!(staged.len(), 1, "only the definitive printed binding");
        assert_eq!(staged[0]["model"], "big-pickle");
        store.release_usage_execution("exec").unwrap();
        store.reconcile_run_attribution().unwrap();
        let staged = self::staged(&store);
        assert_eq!(staged.len(), 2);
        assert_eq!(staged[1]["attributionReason"], "invoker_not_linked");
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_run_a_natively_woken_continuation_launched_binds_through_its_detached_tool_evidence() {
        let (dir, store) = chat_fixture("opencode");
        store
            .begin_usage_execution("exec", "turn", "opencode")
            .unwrap();
        assert!(store.hold_usage_execution("exec").unwrap());
        store.conn.execute_batch("UPDATE chat_turns SET state = 'completed'; UPDATE chat_messages SET completed_at = 2; DELETE FROM chat_turn_leases;").unwrap();
        store.finalize_turn_usage("turn", "done").unwrap();
        let run = uuid::Uuid::new_v4().to_string();
        // What the held watcher records for the woken run's bash call (not in any transcript).
        let mut part = tool("prt_bash", "orx exp run exp");
        part["state"]["output"] = serde_json::json!(format!("  run  {run}\n"));
        store
            .set_native_scope("exec", "tool-part:prt_bash", &part)
            .unwrap();
        store
            .record_native_invocation(
                "prt_bash",
                &InvocationIdentity {
                    harness: "opencode".into(),
                    model: "big-pickle".into(),
                    provider: Some("opencode".into()),
                },
                Some("session"),
            )
            .unwrap();
        launch(&store, &run, Some("session"), "orx exp run exp");
        store
            .update_status(&run, RunStatus::Done, Some(2), Some(0))
            .unwrap();
        let staged = staged(&store);
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0]["attribution"], "exact");
        assert_eq!(staged[0]["model"], "big-pickle");
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
