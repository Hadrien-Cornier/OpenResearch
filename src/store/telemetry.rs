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
    IdentityNotReported,
    /// A sub-agent the harness never identified; never inherits the parent's model.
    ChildModelUnknown,
    /// The native label is a path, URL, or ARN rather than a model identifier.
    InvalidLabel,
    /// Claude's locally generated `<synthetic>` error message.
    SyntheticModel,
    NoUsageReported,
    InvokerNotLinked,
    InvokerAmbiguous,
    /// A native agent CLI outside OpenResearch chat launched the run; it reports no model.
    ExternalAgent,
}

impl Missing {
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

/// One launcher's attribution, `invoker_ambiguous` when launchers disagree, `None` without any.
fn agreed(mut found: std::collections::BTreeSet<Attribution>) -> Option<Attribution> {
    match found.len() {
        0 => None,
        1 => found.pop_first(),
        _ => Some(Attribution::Unresolved {
            reason: Missing::InvokerAmbiguous,
        }),
    }
}

/// Launch tool parts (`… exp run …`) whose native output names `run_id`; nested parts are sub-agents'.
fn printed_launches(
    parts: &[serde_json::Value],
    run_id: &str,
    nested: bool,
    out: &mut Vec<(String, bool)>,
) {
    for part in parts {
        if let Some(children) = part.get("children").and_then(serde_json::Value::as_array) {
            printed_launches(children, run_id, true, out);
        }
        let (Some(id), Some(input)) = (
            part.get("id").and_then(serde_json::Value::as_str),
            part.pointer("/state/input")
                .and_then(serde_json::Value::as_object),
        ) else {
            continue;
        };
        let printed = ["/state/output", "/state/error"].iter().any(|pointer| {
            part.pointer(pointer)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|text| text.to_ascii_lowercase().contains(run_id))
        });
        if printed && crate::local::chat::tool_command(input).contains("exp run") {
            out.push((id.to_string(), nested));
        }
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

    /// Call inside the transaction that inserts the run. A chat-launched run without a forwarded
    /// native identity binds at its terminal transition to the tool part whose output printed it.
    pub(crate) fn reserve_run_telemetry(
        &self,
        run_id: &str,
        identity: Option<&InvocationIdentity>,
        launching_session: Option<&str>,
        agent_origin: Option<&str>,
        report: Option<&(String, serde_json::Value)>,
    ) -> Result<()> {
        if let Some(identity) = identity {
            identity.validate()?;
        }
        // A context inherited from another agent (a host CLI around the app) never overrides the
        // launching chat's own harness.
        let chat_harness: Option<String> = launching_session
            .map(|session| {
                self.conn
                    .query_row(
                        "SELECT harness FROM chat_sessions WHERE id = ?1",
                        [session],
                        |row| row.get(0),
                    )
                    .optional()
            })
            .transpose()?
            .flatten();
        let identity = identity.filter(|identity| {
            chat_harness
                .as_deref()
                .is_none_or(|harness| harness == identity.harness)
        });
        let mut report = report.map(|(_, payload)| payload.clone());
        if let Some(payload) = report.as_mut() {
            let (harness, attribution) = match (identity, launching_session) {
                (Some(identity), _) => (Some(identity.harness.clone()), Attribution::of(identity)),
                (None, Some(_)) => (
                    chat_harness.clone(),
                    Attribution::Unresolved {
                        reason: Missing::InvokerNotLinked,
                    },
                ),
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
        self.conn.execute("INSERT INTO run_telemetry (run_id, identity_json, report_json) VALUES (?1, ?2, ?3) ON CONFLICT(run_id) DO NOTHING", params![run_id, identity.map(serde_json::to_string).transpose()?, report.as_ref().map(serde_json::to_string).transpose()?])?;
        Ok(())
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
                "UPDATE run_telemetry SET report_json = ?2, pending_since = ?3 WHERE run_id = ?1",
                params![run_id, serde_json::to_string(&payload)?, now_ms()],
            )?;
            self.settle_run(run_id)?;
        }
        Ok(())
    }

    /// Terminal runs whose launching chat may still persist the tool output that printed them.
    /// Called after each turn's final flush and at startup; safe to repeat.
    pub(crate) fn settle_pending_runs(&self) -> Result<()> {
        let pending = self
            .conn
            .prepare("SELECT run_id FROM run_telemetry WHERE pending_since IS NOT NULL")?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for run_id in pending {
            let tx = self.begin_immediate()?;
            self.settle_run(&run_id)?;
            tx.commit()?;
        }
        Ok(())
    }

    /// Caller holds a write transaction. Stages a terminal run's report exactly once: bound to the
    /// launch part that printed it, or as it stands once no turn of its chat is still running.
    fn settle_run(&self, run_id: &str) -> Result<()> {
        let Some(report) = self
            .conn
            .query_row(
                "SELECT report_json FROM run_telemetry WHERE run_id = ?1 AND pending_since IS NOT NULL AND report_json IS NOT NULL",
                [run_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
        else {
            return Ok(());
        };
        let mut payload: serde_json::Value = serde_json::from_str(&report)?;
        let properties = &mut payload["events"][0]["properties"];
        if properties["attributionReason"] == "invoker_not_linked" {
            let (attribution, active) = match self.printed_run_invoker(run_id)? {
                Some(attribution) => (Some(attribution), false),
                None => self.native_run_invoker(run_id)?,
            };
            match attribution {
                Some(attribution) => attribution.apply(properties)?,
                None if active || self.launching_turn_running(run_id)? => return Ok(()),
                None => {}
            }
        }
        let id = payload["events"][0]["eventId"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing run telemetry event ID"))?
            .to_string();
        self.stage_telemetry(&id, &payload)?;
        self.conn.execute(
            "UPDATE run_telemetry SET report_json = ?2, pending_since = NULL WHERE run_id = ?1",
            params![run_id, serde_json::to_string(&payload)?],
        )?;
        Ok(())
    }

    /// The run's launcher from its harness's native records, and whether that session still writes.
    fn native_run_invoker(&self, run_id: &str) -> Result<(Option<Attribution>, bool)> {
        let Some((harness, session, native_id)) = self
            .conn
            .query_row(
                "SELECT s.harness, s.id, s.native_session_id FROM runs r JOIN chat_sessions s ON s.id = r.chat_session_id WHERE r.id = ?1 AND s.native_session_id IS NOT NULL",
                [run_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?)),
            )
            .optional()?
        else {
            return Ok((None, false));
        };
        let (found, active) = crate::local::harness::native_run_invokers(
            self, &harness, &session, &native_id, run_id,
        );
        Ok((agreed(found.into_iter().collect()), active))
    }

    /// Records the native server process now serving `session` (since `now`): an earlier step's
    /// owner is the latest one started before it, and survives an ORX restart.
    pub(crate) fn record_native_owner(&self, session: &str, pid: u32, port: u16) -> Result<()> {
        let mut owners = self.native_owners(session)?;
        owners.push((pid, port, now_ms()));
        // ponytail: keeps the last 8 servers; older steps' owners read as gone.
        let owners = &owners[owners.len().saturating_sub(8)..];
        self.conn.execute(
            "UPDATE chat_sessions SET native_owners_json = ?2 WHERE id = ?1",
            params![session, serde_json::to_string(owners)?],
        )?;
        Ok(())
    }

    /// (pid, port, started) of the native servers recorded for `session`, oldest first.
    pub(crate) fn native_owners(&self, session: &str) -> Result<Vec<(u32, u16, i64)>> {
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT native_owners_json FROM chat_sessions WHERE id = ?1",
                [session],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        Ok(json
            .map(|json| serde_json::from_str(&json))
            .transpose()?
            .unwrap_or_default())
    }

    /// The chat session's native conversations its hooks recorded identities for.
    pub(crate) fn invocation_conversations(
        &self,
        harness: &str,
        session: &str,
    ) -> Result<std::collections::BTreeSet<String>> {
        let keys = self
            .conn
            .prepare("SELECT call_id FROM native_invocation_identities WHERE harness = ?1 AND session_id = ?2")?
            .query_map([harness, session], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(keys
            .iter()
            .filter_map(|key| {
                key.strip_prefix(&format!("{harness}:"))?
                    .split_once(":step:")
            })
            .map(|(conversation, _)| conversation.to_string())
            .collect())
    }

    fn launching_turn_running(&self, run_id: &str) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT EXISTS (SELECT 1 FROM runs r JOIN chat_turns t ON t.session_id = r.chat_session_id WHERE r.id = ?1 AND t.state IN ('preparing', 'retrying', 'running'))",
            [run_id],
            |row| row.get(0),
        )?)
    }

    /// The native invoker of the launch tool part whose output printed `run_id` in its chat.
    /// `None` when no part printed it (for example a run that ended before its tool output persisted).
    fn printed_run_invoker(&self, run_id: &str) -> Result<Option<Attribution>> {
        let Some((harness, created_at, session)) = self
            .conn
            .query_row(
                "SELECT s.harness, r.created_at, s.id FROM runs r JOIN chat_sessions s ON s.id = r.chat_session_id WHERE r.id = ?1",
                [run_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, String>(2)?)),
            )
            .optional()?
        else {
            return Ok(None);
        };
        // The launching turn started before the run; a few recent replies cover it.
        let messages = self
            .conn
            .prepare("SELECT parts_json FROM chat_messages WHERE session_id = ?1 AND role = 'assistant' AND created_at <= ?2 ORDER BY created_at DESC LIMIT 5")?
            .query_map(params![session, created_at], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut launches = Vec::new();
        for json in messages {
            let parts: Vec<serde_json::Value> = serde_json::from_str(&json)?;
            printed_launches(&parts, &run_id.to_ascii_lowercase(), false, &mut launches);
        }
        let mut found = std::collections::BTreeSet::new();
        for (part_id, nested) in &launches {
            // Harnesses whose part ids repeat across chats scope them by session.
            let identity = match self.native_invocation_identity(&harness, part_id)? {
                Some(identity) => Some(identity),
                None => {
                    self.native_invocation_identity(&harness, &format!("{session}:{part_id}"))?
                }
            };
            found.insert(match identity {
                Some(identity) => Attribution::of(&identity),
                None => Attribution::Unresolved {
                    reason: Missing::unidentified(*nested),
                },
            });
        }
        Ok(agreed(found))
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
                false,
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
                    // The aggregate counts every request's tokens, so listed models are replaced
                    // and any other identity keeps only its model.
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
                        self.record_attributed_sample(
                            execution_id,
                            &sample_id,
                            harness,
                            &Attribution::native(
                                harness,
                                Some(model),
                                provider.as_deref(),
                                Missing::IdentityNotReported,
                            ),
                            &delta,
                            delta.input_tokens.is_some() && delta.output_tokens.is_some(),
                            false,
                        )?;
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

    /// Opens an execution for native work outside any chat turn (a sub-agent that outlives its
    /// parent's turn), owned by the session's latest turn. Its id starts with `native:`, so the
    /// turn's own finalization leaves it open; it closes on its native end or startup recovery.
    pub(crate) fn begin_native_execution(
        &self,
        execution_id: &str,
        session_id: &str,
        harness: &str,
    ) -> Result<bool> {
        debug_assert!(execution_id.starts_with("native:"));
        let inserted = self.conn.execute("INSERT INTO chat_usage_executions (execution_id, turn_id, harness, report_id, suppressed) SELECT ?1, t.id, ?3, ?4, ?5 FROM chat_turns t WHERE t.session_id = ?2 ORDER BY t.created_at DESC LIMIT 1 ON CONFLICT(execution_id) DO NOTHING", params![execution_id, session_id, harness, uuid::Uuid::new_v4().to_string(), !crate::telemetry::accounting_reports_enabled()])?;
        Ok(inserted > 0
            || self.conn.query_row(
                "SELECT EXISTS (SELECT 1 FROM chat_usage_executions WHERE execution_id = ?1 AND outcome IS NULL)",
                [execution_id],
                |row| row.get(0),
            )?)
    }

    /// The open chat execution of `turn_id` on `harness`, with its native session and start time.
    pub(crate) fn open_turn_execution(
        &self,
        turn_id: &str,
        harness: &str,
    ) -> Result<Option<(String, String, i64)>> {
        Ok(self.conn.query_row("SELECT u.execution_id, s.native_session_id, t.created_at FROM chat_usage_executions u JOIN chat_turns t ON t.id = u.turn_id JOIN chat_sessions s ON s.id = t.session_id WHERE u.turn_id = ?1 AND u.harness = ?2 AND u.outcome IS NULL AND substr(u.execution_id, 1, 7) <> 'native:' AND s.native_session_id IS NOT NULL", params![turn_id, harness], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))).optional()?)
    }

    /// Samples under `prefix` that carry no native model and no hook identity.
    pub(crate) fn unidentified_samples(
        &self,
        execution_id: &str,
        prefix: &str,
    ) -> Result<Vec<String>> {
        Ok(self
            .conn
            .prepare("SELECT s.sample_id FROM chat_usage_samples s LEFT JOIN native_invocation_identities n ON n.harness = s.harness AND n.call_id = s.sample_id WHERE s.execution_id = ?1 AND s.model IS NULL AND n.call_id IS NULL AND substr(s.sample_id, 1, length(?2)) = ?2")?
            .query_map(params![execution_id, prefix], |row| row.get(0))?
            .collect::<std::result::Result<_, _>>()?)
    }

    /// Turns whose chat execution is still open.
    pub(crate) fn open_execution_turns(&self) -> Result<Vec<String>> {
        Ok(self
            .conn
            .prepare("SELECT turn_id FROM chat_usage_executions WHERE outcome IS NULL AND substr(execution_id, 1, 7) <> 'native:'")?
            .query_map([], |row| row.get(0))?
            .collect::<std::result::Result<_, _>>()?)
    }

    /// (session, native session, first accounted turn start) on `harness` with no turn still
    /// running. Native history before ORX's first turn (an imported session) is never accounted.
    pub(crate) fn idle_native_sessions(&self, harness: &str) -> Result<Vec<(String, String, i64)>> {
        Ok(self
            .conn
            .prepare("SELECT s.id, s.native_session_id, MIN(t.created_at) FROM chat_sessions s JOIN chat_turns t ON t.session_id = s.id JOIN chat_usage_executions u ON u.turn_id = t.id WHERE s.harness = ?1 AND s.native_session_id IS NOT NULL GROUP BY s.id HAVING SUM(u.outcome IS NULL AND substr(u.execution_id, 1, 7) <> 'native:') = 0")?
            .query_map([harness], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<std::result::Result<_, _>>()?)
    }

    /// Whether any execution holds native sample `sample_id` (measured, when `measured`).
    pub(crate) fn native_sample_exists(
        &self,
        harness: &str,
        sample_id: &str,
        measured: bool,
    ) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT EXISTS (SELECT 1 FROM chat_usage_samples WHERE harness = ?1 AND sample_id = ?2 AND (NOT ?3 OR usage_json <> ?4))",
            params![harness, sample_id, measured, serde_json::to_string(&TokenUsage::default())?],
            |row| row.get(0),
        )?)
    }

    pub(crate) fn recover_terminal_usage(&self) -> Result<()> {
        let rows = self.conn.prepare("SELECT u.execution_id, t.state FROM chat_usage_executions u JOIN chat_turns t ON t.id = u.turn_id WHERE u.outcome IS NULL AND t.state IN ('completed', 'failed', 'interrupted') AND NOT EXISTS (SELECT 1 FROM chat_turn_leases l WHERE l.chat_session_id = t.session_id)")?
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (execution_id, state) in rows {
            self.finalize_usage_execution(
                &execution_id,
                match state.as_str() {
                    "completed" => "done",
                    "interrupted" => "cancelled",
                    _ => "failed",
                },
            )?;
        }
        Ok(())
    }

    pub(crate) fn finalize_turn_usage(&self, turn_id: &str, outcome: &str) -> Result<()> {
        let executions = self
            .conn
            .prepare("SELECT execution_id FROM chat_usage_executions WHERE turn_id = ?1 AND outcome IS NULL AND substr(execution_id, 1, 7) <> 'native:'")?
            .query_map([turn_id], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for execution_id in executions {
            self.finalize_usage_execution(&execution_id, outcome)?;
        }
        Ok(())
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
        type Key = (Attribution, [bool; 5]);
        let mut grouped = std::collections::BTreeMap::<Key, Vec<(TokenUsage, bool)>>::new();
        let mut samples = self.conn.prepare("SELECT s.model, s.provider, s.usage_json, s.complete, n.identity_json, s.attribution_json FROM chat_usage_samples s LEFT JOIN native_invocation_identities n ON n.harness = s.harness AND n.call_id = s.sample_id WHERE s.execution_id = ?1 ORDER BY s.sample_id")?;
        for row in samples.query_map([execution_id], |row| {
            Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, bool>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })? {
            let (model, provider, json, complete, identity, attribution) = row?;
            let identity: Option<InvocationIdentity> = identity
                .map(|json| serde_json::from_str(&json))
                .transpose()?;
            // Rows from before attribution existed carry only the model.
            let attribution = match attribution {
                Some(json) => serde_json::from_str(&json)?,
                None => Attribution::native(
                    harness,
                    model.as_deref(),
                    provider.as_deref(),
                    Missing::IdentityNotReported,
                ),
            };
            // A native identity recorded under the same sample id (a hook) resolves it.
            let attribution = match (attribution, identity) {
                (
                    Attribution::Unresolved {
                        reason: Missing::IdentityNotReported | Missing::ChildModelUnknown,
                    },
                    Some(identity),
                ) => Attribution::of(&identity),
                (attribution, _) => attribution,
            };
            let usage: TokenUsage = serde_json::from_str(&json)?;
            let measured = usage.counters().map(|counter| counter.is_some());
            grouped
                .entry((attribution, measured))
                .or_default()
                .push((usage, complete));
        }
        // An identity seen without counters joins its model's measured requests when there are any.
        let measured: std::collections::BTreeSet<Attribution> = grouped
            .keys()
            .filter(|(_, mask)| mask.contains(&true))
            .map(|(attribution, _)| attribution.clone())
            .collect();
        grouped.retain(|(attribution, mask), _| {
            mask.contains(&true) || !measured.contains(attribution)
        });
        // A zero-usage synthetic error only reports when nothing else ran.
        let only_group = grouped.len() == 1;
        grouped.retain(|(attribution, _), samples| {
            *attribution
                != Attribution::Unresolved {
                    reason: Missing::SyntheticModel,
                }
                || only_group
                || samples
                    .iter()
                    .any(|(usage, _)| usage.counters().iter().any(|n| n.unwrap_or(0) > 0))
        });
        if grouped.is_empty() {
            let attribution = if matches!(delivery, Some("not_sent" | "rejected")) {
                Attribution::NotExecuted
            } else {
                Attribution::Unresolved {
                    reason: Missing::NoUsageReported,
                }
            };
            grouped.insert((attribution, [false; 5]), Vec::new());
        }
        let mut reports = Vec::new();
        for ((attribution, _), samples) in grouped {
            let sum = |field: fn(&TokenUsage) -> Option<u64>| -> Option<u64> {
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
            attribution.apply(&mut properties)?;
            properties["totalTokens"] = serde_json::json!(usage.total());
            properties["outcome"] = serde_json::json!(outcome);
            properties["coverage"] =
                serde_json::json!(if usage.counters().iter().all(Option::is_none) {
                    "missing"
                } else if samples.iter().all(|(usage, complete)| {
                    *complete && usage.input_tokens.is_some() && usage.output_tokens.is_some()
                }) {
                    "complete"
                } else {
                    "partial"
                });
            reports.push(properties);
        }
        Ok(reports)
    }

    #[cfg(test)]
    pub(crate) fn conn_for_tests(&self) -> &rusqlite::Connection {
        &self.conn
    }

    /// Every execution's report properties at `outcome`, without closing or staging.
    #[cfg(test)]
    pub(crate) fn test_usage_reports(&self, outcome: &str) -> Result<Vec<serde_json::Value>> {
        let executions = self
            .conn
            .prepare("SELECT u.execution_id, u.harness, u.report_id, t.delivery_state FROM chat_usage_executions u LEFT JOIN chat_turns t ON t.id = u.turn_id ORDER BY u.execution_id")?
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, Option<String>>(3)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut reports = Vec::new();
        for (execution, harness, report_id, delivery) in executions {
            reports.extend(self.usage_report_properties(
                &execution,
                &harness,
                &report_id,
                outcome,
                delivery.as_deref(),
            )?);
        }
        Ok(reports)
    }

    /// Closes and stages under one write lock, so a sample committed by another process is either
    /// in the report or rejected as late; never accepted and unreported.
    pub(crate) fn finalize_usage_execution(&self, execution_id: &str, outcome: &str) -> Result<()> {
        let tx = self.begin_immediate()?;
        let Some((turn_id, harness, report_id, suppressed)) = tx
            .query_row(
                "UPDATE chat_usage_executions SET outcome = ?2 WHERE execution_id = ?1 AND outcome IS NULL RETURNING turn_id, harness, report_id, suppressed",
                params![execution_id, outcome],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, bool>(3)?)),
            )
            .optional()?
        else {
            return Ok(());
        };
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
                delivery
                    .as_deref()
                    .filter(|_| !execution_id.starts_with("native:")),
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
            "UPDATE run_telemetry SET report_json = NULL, pending_since = NULL",
            [],
        )?;
        tx.execute(
            "UPDATE chat_usage_executions SET suppressed = 1 WHERE outcome IS NULL",
            [],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Upserts one native sample: an identity-only write keeps measured counters, and a measured
    /// write replaces them (snapshots, not deltas). `exclusive` skips a sample already measured
    /// by another execution, so re-read native records count once.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_attributed_sample(
        &self,
        execution_id: &str,
        sample_id: &str,
        harness: &str,
        attribution: &Attribution,
        usage: &TokenUsage,
        complete: bool,
        exclusive: bool,
    ) -> Result<()> {
        usage.validate()?;
        let (model, provider) = attribution.model();
        let unmeasured = serde_json::to_string(&TokenUsage::default())?;
        let measured = usage != &TokenUsage::default();
        self.conn.execute("INSERT INTO chat_usage_samples (execution_id, sample_id, harness, model, provider, usage_json, complete, attribution_json) SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8 WHERE EXISTS (SELECT 1 FROM chat_usage_executions WHERE execution_id = ?1 AND outcome IS NULL) AND NOT (?10 AND EXISTS (SELECT 1 FROM chat_usage_samples o WHERE o.sample_id = ?2 AND o.harness = ?3 AND o.execution_id <> ?1 AND o.usage_json <> ?11)) ON CONFLICT(execution_id, sample_id) DO UPDATE SET usage_json = CASE WHEN ?9 THEN excluded.usage_json ELSE usage_json END, complete = CASE WHEN ?9 THEN excluded.complete ELSE complete END, model = COALESCE(excluded.model, model), provider = CASE WHEN excluded.model IS NULL AND model IS NOT NULL THEN provider ELSE excluded.provider END, attribution_json = CASE WHEN excluded.model IS NULL AND model IS NOT NULL THEN attribution_json ELSE excluded.attribution_json END", params![execution_id, sample_id, harness, model, provider, serde_json::to_string(usage)?, complete, serde_json::to_string(attribution)?, measured, exclusive, unmeasured])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Store {
        fn record_usage_sample(
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
                false,
            )
        }
    }

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
        store
            .record_usage_sample(
                "execution",
                "sample",
                "claude-code",
                Some("claude-opus-5-5"),
                None,
                &TokenUsage::default(),
            )
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
            .reserve_run_telemetry(&run.id, Some(&identity), None, None, Some(&report))
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
        for _ in 0..2 {
            store
                .record_cumulative_usage(
                    "execution",
                    "codex",
                    "thread",
                    "native-turn",
                    &Attribution::Unresolved {
                        reason: Missing::IdentityNotReported,
                    },
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
                &Attribution::Unresolved {
                    reason: Missing::IdentityNotReported,
                },
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
                &Attribution::Unresolved {
                    reason: Missing::IdentityNotReported,
                },
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
                &Attribution::Unresolved {
                    reason: Missing::IdentityNotReported,
                },
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

    fn exact(model: &str) -> Attribution {
        Attribution::Exact {
            model: model.into(),
            provider: None,
        }
    }

    fn measured(input: u64, output: u64) -> TokenUsage {
        TokenUsage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            ..Default::default()
        }
    }

    /// Report rows as (attribution, model, reason, coverage, input).
    type Row = (String, Option<String>, Option<String>, String, Option<u64>);

    fn rows(store: &Store) -> Vec<Row> {
        store
            .test_usage_reports("done")
            .unwrap()
            .iter()
            .map(|r| {
                (
                    r["attribution"].as_str().unwrap().into(),
                    r["model"].as_str().map(Into::into),
                    r["attributionReason"].as_str().map(Into::into),
                    r["coverage"].as_str().unwrap().into(),
                    r["inputTokens"].as_u64(),
                )
            })
            .collect()
    }

    fn row(
        attribution: &str,
        model: Option<&str>,
        reason: Option<&str>,
        coverage: &str,
        input: Option<u64>,
    ) -> Row {
        (
            attribution.into(),
            model.map(Into::into),
            reason.map(Into::into),
            coverage.into(),
            input,
        )
    }

    /// Native samples → reports: identity survives without tokens, children and exceptions stay
    /// explicit, and nothing executed is never a model.
    #[test]
    fn usage_reports_keep_identities_tokens_and_exceptions_separate() {
        type Sample = (&'static str, Attribution, TokenUsage, bool);
        let unknown_child = Attribution::Unresolved {
            reason: Missing::ChildModelUnknown,
        };
        let synthetic = Attribution::Unresolved {
            reason: Missing::SyntheticModel,
        };
        let cases: Vec<(&str, Vec<Sample>, Vec<_>)> =
            vec![
            (
                "identity joins its measured requests; another model's identity alone is missing",
                vec![
                    ("a", exact("m1"), measured(10, 2), true),
                    ("a-id", exact("m1"), TokenUsage::default(), false),
                    ("b", exact("m2"), TokenUsage::default(), false),
                ],
                vec![
                    row("exact", Some("m1"), None, "complete", Some(10)),
                    row("exact", Some("m2"), None, "missing", None),
                ],
            ),
            (
                "an unidentified child never takes the parent's model",
                vec![
                    ("p", exact("m1"), measured(5, 1), true),
                    ("c", unknown_child.clone(), measured(3, 1), true),
                ],
                vec![
                    row("exact", Some("m1"), None, "complete", Some(5)),
                    row("unresolved", None, Some("child_model_unknown"), "complete", Some(3)),
                ],
            ),
            (
                "a zero-usage synthetic error only reports alone",
                vec![
                    ("p", exact("m1"), measured(5, 1), true),
                    ("s", synthetic.clone(), measured(0, 0), true),
                ],
                vec![row("exact", Some("m1"), None, "complete", Some(5))],
            ),
            (
                "a synthetic error alone is explicit",
                vec![("s", synthetic, measured(0, 0), true)],
                vec![row("unresolved", None, Some("synthetic_model"), "complete", Some(0))],
            ),
            (
                "a delivered turn with no native evidence is unresolved",
                vec![],
                vec![row("unresolved", None, Some("no_usage_reported"), "missing", None)],
            ),
        ];
        for (name, samples, expected) in cases {
            let dir = std::env::temp_dir().join(format!("orx-reports-{}", uuid::Uuid::new_v4()));
            let store = Store::open_at(dir.clone()).unwrap();
            store.begin_usage_execution("e", "t", "codex").unwrap();
            for (id, attribution, usage, complete) in samples {
                store
                    .record_attributed_sample(
                        "e",
                        id,
                        "codex",
                        &attribution,
                        &usage,
                        complete,
                        false,
                    )
                    .unwrap();
            }
            assert_eq!(rows(&store), expected, "{name}");
            drop(store);
            std::fs::remove_dir_all(dir).unwrap();
        }
        // Never delivered: not executed, not a model.
        let dir = std::env::temp_dir().join(format!("orx-reports-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store.conn.execute_batch("INSERT INTO chat_turns (id, session_id, assistant_message_id, client_turn_id, request_hash, prepared_input, settings_json, state, delivery_state, created_at, updated_at) VALUES ('t', 's', 'm', 'c', 'h', '', '{}', 'failed', 'rejected', 1, 1)").unwrap();
        store.begin_usage_execution("e", "t", "codex").unwrap();
        assert_eq!(
            rows(&store),
            vec![row("not_executed", None, None, "missing", None)]
        );
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Re-read native records count once across executions, and a sub-agent's own execution
    /// outlives its parent turn's finalization until it closes natively or at recovery.
    #[test]
    fn native_records_count_once_and_late_children_close_independently() {
        let dir = std::env::temp_dir().join(format!("orx-native-exec-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store.conn.execute_batch("INSERT INTO chat_sessions (id, project_id, harness, created_at, updated_at) VALUES ('s', 'p', 'opencode', 1, 1);
            INSERT INTO chat_turns (id, session_id, assistant_message_id, client_turn_id, request_hash, prepared_input, settings_json, state, delivery_state, created_at, updated_at) VALUES ('t', 's', 'm', 'c', 'h', '', '{}', 'running', 'accepted', 1, 1);").unwrap();
        store
            .begin_usage_execution("turn", "t", "opencode")
            .unwrap();
        let record = |execution: &str, id: &str, usage: TokenUsage| {
            store
                .record_attributed_sample(
                    execution,
                    id,
                    "opencode",
                    &exact("m"),
                    &usage,
                    true,
                    true,
                )
                .unwrap()
        };
        record("turn", "step-1", measured(10, 1));
        record("turn", "step-2", TokenUsage::default());
        assert!(store
            .begin_native_execution("native:late", "s", "opencode")
            .unwrap());
        record("native:late", "step-1", measured(10, 1));
        record("native:late", "step-2", measured(20, 2));
        store.finalize_turn_usage("t", "done").unwrap();
        assert!(store
            .native_sample_exists("opencode", "step-2", true)
            .unwrap());
        let open: Vec<String> = store
            .conn
            .prepare("SELECT execution_id FROM chat_usage_executions WHERE outcome IS NULL")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert_eq!(open, ["native:late"]);
        let late: Vec<String> = store
            .conn
            .prepare("SELECT sample_id FROM chat_usage_samples WHERE execution_id = 'native:late'")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert_eq!(late, ["step-2"]);
        store
            .conn
            .execute_batch("UPDATE chat_turns SET state = 'completed'")
            .unwrap();
        store.recover_terminal_usage().unwrap();
        let open: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM chat_usage_executions WHERE outcome IS NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(open, 0);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A chat-launched run without forwarded identity binds at its terminal transition to the
    /// launch part whose native output printed it, never to the chat's other models.
    #[test]
    fn runs_bind_to_their_printed_launch_part_or_stay_explicit() {
        let launch = |id: &str, run: &str| {
            serde_json::json!({"id": id, "type": "tool", "state": {"status": "completed",
                "input": {"command": "orx exp run e"}, "output": format!("Run {run} started")}})
        };
        let cases = [
            // (parts, invokers, expected attribution, reason)
            (
                vec![launch("p1", "7f3a-run")],
                vec![("p1", "m1")],
                "exact",
                None,
            ),
            (
                vec![serde_json::json!({"id": "task", "children": [launch("c1", "7f3a-run")]})],
                vec![("task", "m1")],
                "unresolved",
                Some("child_model_unknown"),
            ),
            (
                vec![launch("p1", "7f3a-run"), launch("p2", "7f3a-run")],
                vec![("p1", "m1"), ("p2", "m2")],
                "unresolved",
                Some("invoker_ambiguous"),
            ),
            (
                vec![launch("p1", "9c1b-run")],
                vec![("p1", "m1")],
                "unresolved",
                Some("invoker_not_linked"),
            ),
        ];
        for (parts, invokers, attribution, reason) in cases {
            let dir = std::env::temp_dir().join(format!("orx-run-bind-{}", uuid::Uuid::new_v4()));
            let store = Store::open_at(dir.clone()).unwrap();
            store.conn.execute("INSERT INTO chat_sessions (id, project_id, harness, created_at, updated_at) VALUES ('s', 'p', 'cursor', 1, 1)", []).unwrap();
            store.conn.execute("INSERT INTO chat_messages (id, session_id, role, parts_json, created_at) VALUES ('m', 's', 'assistant', ?1, 1)", [serde_json::to_string(&parts).unwrap()]).unwrap();
            for (part, model) in invokers {
                let identity = InvocationIdentity {
                    harness: "cursor".into(),
                    model: model.into(),
                    provider: None,
                };
                store
                    .record_native_invocation(part, &identity, Some("s"))
                    .unwrap();
            }
            let event = serde_json::json!({"events": [{"eventId": "ev", "properties": {"status": "failed"}}]});
            let tx = store.begin().unwrap();
            store
                .reserve_run_telemetry(
                    "7f3a-run",
                    None,
                    Some("s"),
                    None,
                    Some(&("ev".into(), event)),
                )
                .unwrap();
            store
                .upsert_run(&StoredRun {
                    id: "7f3a-run".into(),
                    experiment_id: "e".into(),
                    project_id: "p".into(),
                    status: "starting".into(),
                    backend_json: "{}".into(),
                    command: String::new(),
                    created_at: 2,
                    updated_at: 2,
                    ended_at: None,
                    exit_code: None,
                    commit_sha: None,
                    result_markdown: None,
                    cancel_requested: false,
                    chat_session_id: Some("s".into()),
                })
                .unwrap();
            tx.commit().unwrap();
            assert!(store
                .update_status("7f3a-run", RunStatus::Done, Some(3), Some(0))
                .unwrap());
            let staged = store.pending_telemetry().unwrap();
            let properties = &staged[0].1["events"][0]["properties"];
            assert_eq!(properties["harness"], "cursor");
            assert_eq!(properties["status"], "done");
            assert_eq!(properties["attribution"], attribution, "{parts:?}");
            assert_eq!(
                properties["attributionReason"].as_str(),
                reason,
                "{parts:?}"
            );
            assert_eq!(
                properties["model"].as_str(),
                (attribution == "exact").then_some("m1")
            );
            drop(store);
            std::fs::remove_dir_all(dir).unwrap();
        }
        // Outside chat: an agent CLI marker is external, nothing is manual.
        for (origin, harness, attribution) in [
            (Some("codex"), Some("codex"), "unresolved"),
            (Some("unknown"), None, "unresolved"),
            (None, None, "manual"),
        ] {
            let dir = std::env::temp_dir().join(format!("orx-run-origin-{}", uuid::Uuid::new_v4()));
            let store = Store::open_at(dir.clone()).unwrap();
            let event = serde_json::json!({"events": [{"eventId": "ev", "properties": {"status": "failed"}}]});
            store
                .reserve_run_telemetry("run", None, None, origin, Some(&("ev".into(), event)))
                .unwrap();
            let report: String = store
                .conn
                .query_row("SELECT report_json FROM run_telemetry", [], |row| {
                    row.get(0)
                })
                .unwrap();
            let report: serde_json::Value = serde_json::from_str(&report).unwrap();
            let properties = &report["events"][0]["properties"];
            assert_eq!(properties["harness"].as_str(), harness);
            assert_eq!(properties["attribution"], attribution);
            drop(store);
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    /// A fast run can end before its launch tool's output persists: the report waits (durably,
    /// across a restart) while the launching turn runs, then emits once — exact once the printed
    /// part appears, or invoker_not_linked once the turn settles without it.
    #[test]
    fn terminal_runs_wait_for_their_launch_output_and_emit_once() {
        for (evidence, restart) in [(true, false), (true, true), (false, true)] {
            let dir =
                std::env::temp_dir().join(format!("orx-run-pending-{}", uuid::Uuid::new_v4()));
            let mut store = Store::open_at(dir.clone()).unwrap();
            store.conn.execute_batch("INSERT INTO chat_sessions (id, project_id, harness, created_at, updated_at) VALUES ('s', 'p', 'opencode', 1, 1);
                INSERT INTO chat_turns (id, session_id, assistant_message_id, client_turn_id, request_hash, prepared_input, settings_json, state, delivery_state, created_at, updated_at) VALUES ('t', 's', 'm', 'c', 'h', '', '{}', 'running', 'accepted', 1, 1);
                INSERT INTO chat_messages (id, session_id, role, parts_json, created_at) VALUES ('m', 's', 'assistant', '[]', 1);").unwrap();
            let event = serde_json::json!({"events": [{"eventId": "ev", "properties": {"status": "failed"}}]});
            let tx = store.begin().unwrap();
            store
                .reserve_run_telemetry(
                    "7f3a-run",
                    None,
                    Some("s"),
                    None,
                    Some(&("ev".into(), event)),
                )
                .unwrap();
            store
                .upsert_run(&StoredRun {
                    id: "7f3a-run".into(),
                    experiment_id: "e".into(),
                    project_id: "p".into(),
                    status: "starting".into(),
                    backend_json: "{}".into(),
                    command: String::new(),
                    created_at: 2,
                    updated_at: 2,
                    ended_at: None,
                    exit_code: None,
                    commit_sha: None,
                    result_markdown: None,
                    cancel_requested: false,
                    chat_session_id: Some("s".into()),
                })
                .unwrap();
            tx.commit().unwrap();
            assert!(store
                .update_status("7f3a-run", RunStatus::Done, Some(3), Some(0))
                .unwrap());
            store.settle_pending_runs().unwrap();
            assert!(
                store.pending_telemetry().unwrap().is_empty(),
                "waits for the turn"
            );
            if restart {
                drop(store);
                store = Store::open_at(dir.clone()).unwrap();
            }
            if evidence {
                let parts = serde_json::json!([{"id": "call_1", "type": "tool", "state": {"status": "completed",
                    "input": {"command": "orx exp run e"}, "output": "Run 7f3a-run done"}}]);
                store
                    .conn
                    .execute(
                        "UPDATE chat_messages SET parts_json = ?1",
                        [parts.to_string()],
                    )
                    .unwrap();
                let identity = InvocationIdentity {
                    harness: "opencode".into(),
                    model: "big-pickle".into(),
                    provider: Some("opencode".into()),
                };
                store
                    .record_native_invocation("call_1", &identity, Some("s"))
                    .unwrap();
            } else {
                store
                    .conn
                    .execute("UPDATE chat_turns SET state = 'completed'", [])
                    .unwrap();
            }
            store.settle_pending_runs().unwrap();
            store.settle_pending_runs().unwrap();
            let staged = store.pending_telemetry().unwrap();
            assert_eq!(staged.len(), 1);
            let properties = &staged[0].1["events"][0]["properties"];
            assert_eq!(properties["status"], "done");
            if evidence {
                assert_eq!(properties["attribution"], "exact");
                assert_eq!(properties["model"], "big-pickle");
            } else {
                assert_eq!(properties["attributionReason"], "invoker_not_linked");
            }
            drop(store);
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    /// A run launched from an OpenCode chat inside a host Codex inherits the host's context: it
    /// must not report the host's model, and the chat's own native binding takes over.
    #[test]
    fn inherited_foreign_context_never_overrides_the_launching_chats_harness() {
        let dir = std::env::temp_dir().join(format!("orx-run-foreign-{}", uuid::Uuid::new_v4()));
        let store = Store::open_at(dir.clone()).unwrap();
        store.conn.execute("INSERT INTO chat_sessions (id, project_id, harness, created_at, updated_at) VALUES ('s', 'p', 'opencode', 1, 1)", []).unwrap();
        let host = InvocationIdentity {
            harness: "codex".into(),
            model: "gpt-6-astra".into(),
            provider: None,
        };
        let event =
            serde_json::json!({"events": [{"eventId": "ev", "properties": {"status": "failed"}}]});
        store
            .reserve_run_telemetry(
                "run",
                Some(&host),
                Some("s"),
                None,
                Some(&("ev".into(), event)),
            )
            .unwrap();
        let (identity, report): (Option<String>, String) = store
            .conn
            .query_row(
                "SELECT identity_json, report_json FROM run_telemetry",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let report: serde_json::Value = serde_json::from_str(&report).unwrap();
        let properties = &report["events"][0]["properties"];
        assert!(identity.is_none());
        assert_eq!(properties["harness"], "opencode");
        assert_eq!(properties["model"], serde_json::Value::Null);
        assert_eq!(properties["attributionReason"], "invoker_not_linked");
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
