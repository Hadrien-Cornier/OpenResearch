//! `orx mcp-gate` — the plan-mode permission bridge.
//!
//! Hidden from `--help`: it's not a user command, it's the stdio MCP server
//! Claude Code spawns via `--mcp-config` and consults via
//! `--permission-prompt-tool mcp__orx__approve` (see
//! `local::harness::claude::write_mcp_config`). Every permission decision the
//! CLI would have shown as an interactive prompt arrives here as a
//! `tools/call`; we relay it to the running `orx up` over localhost HTTP —
//! which auto-decides by policy or surfaces a card and *blocks until the user
//! answers* — and hand the decision back. That held call is what turns
//! headless plan mode into desktop-style mid-turn approvals.
//!
//! The wire is MCP's stdio transport: newline-delimited JSON-RPC 2.0, the same
//! framing as `local::codex` (this end is the server). Only three methods
//! matter — `initialize`, `tools/list`, `tools/call` — so it's hand-rolled
//! rather than pulling in an MCP crate.
//!
//! Failure posture: never hang Claude and never allow by accident. Any
//! transport or orx-side error answers `deny` with the reason; unknown methods
//! get a JSON-RPC error; a missing env contract exits nonzero (Claude reports
//! the server as failed and plan mode degrades to its default gating).

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::error::{anyhow, Result};

/// Env contract injected by `write_mcp_config` (values ride the MCP server's
/// `env` block, so they survive however Claude spawns us).
struct GateEnv {
    up_port: u16,
    session_id: String,
    token: String,
}

impl GateEnv {
    fn from_env() -> Result<Self> {
        let up_port = std::env::var("ORX_UP_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .ok_or_else(|| anyhow!("ORX_UP_PORT missing or invalid"))?;
        let session_id =
            std::env::var("ORX_SESSION_ID").map_err(|_| anyhow!("ORX_SESSION_ID missing"))?;
        let token =
            std::env::var("ORX_GATE_TOKEN").map_err(|_| anyhow!("ORX_GATE_TOKEN missing"))?;
        Ok(Self {
            up_port,
            session_id,
            token,
        })
    }
}

pub async fn run() -> Result<()> {
    let env = GateEnv::from_env()?;
    // The long-poll deliberately blocks for as long as the user thinks; only
    // connecting is bounded. (orx up itself times pending cards out.)
    let http = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| anyhow!("http client: {e}"))?;

    // Single writer task: tool calls are handled concurrently (Claude may
    // check several tools at once), so responses funnel through one channel
    // to keep stdout lines whole.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Value>();
    let writer = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(msg) = out_rx.recv().await {
            let mut line = msg.to_string();
            line.push('\n');
            if stdout.write_all(line.as_bytes()).await.is_err() {
                break;
            }
            let _ = stdout.flush().await;
        }
    });

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    while let Some(line) = lines.next_line().await.unwrap_or(None) {
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let id = msg.get("id").cloned();
        match msg.get("method").and_then(Value::as_str) {
            Some("initialize") => {
                // Echo the client's protocol version: we do nothing
                // version-specific, and echoing avoids a handshake mismatch.
                let version = msg
                    .pointer("/params/protocolVersion")
                    .cloned()
                    .unwrap_or_else(|| json!("2025-06-18"));
                let _ = out_tx.send(reply(
                    id,
                    json!({
                        "protocolVersion": version,
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "orx", "version": env!("CARGO_PKG_VERSION") },
                    }),
                ));
            }
            Some("tools/list") => {
                let _ = out_tx.send(reply(
                    id,
                    json!({
                        "tools": [{
                            "name": "approve",
                            "description": "Ask the orx user to approve a tool call",
                            "inputSchema": { "type": "object", "additionalProperties": true },
                        }]
                    }),
                ));
            }
            Some("tools/call") => {
                // Handled concurrently: a held approval must not block the next
                // permission check (Claude can run tools in parallel).
                let args = msg
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let url = format!("http://127.0.0.1:{}/api/internal/permissions", env.up_port);
                let body = json!({
                    "sessionId": env.session_id,
                    "token": env.token,
                    "toolName": args.get("tool_name").and_then(Value::as_str).unwrap_or(""),
                    "toolInput": args.get("input").cloned().unwrap_or_else(|| json!({})),
                    "toolUseId": args.get("tool_use_id").and_then(Value::as_str),
                });
                let http = http.clone();
                let out = out_tx.clone();
                tokio::spawn(async move {
                    let decision = relay(&http, &url, body).await.unwrap_or_else(|e| {
                        json!({
                            "behavior": "deny",
                            "message": format!("orx approval bridge unavailable: {e}"),
                        })
                    });
                    // The permission-prompt-tool contract: the decision rides
                    // JSON-*stringified* inside an MCP text content block.
                    let _ = out.send(reply(
                        id,
                        json!({ "content": [{ "type": "text", "text": decision.to_string() }] }),
                    ));
                });
            }
            // Notifications (no id) are fire-and-forget; anything else with an
            // id gets a proper method-not-found so the client never stalls.
            _ => {
                if let Some(id) = id {
                    let _ = out_tx.send(json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": "method not found" },
                    }));
                }
            }
        }
    }

    drop(out_tx);
    let _ = writer.await;
    Ok(())
}

fn reply(id: Option<Value>, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "result": result })
}

/// POST the permission request to `orx up` and return its decision JSON.
/// The response body is the decision verbatim (`{"behavior": ...}`).
async fn relay(http: &reqwest::Client, url: &str, body: Value) -> Result<Value> {
    let resp = http
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow!("{e}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!("orx up answered {}", resp.status()));
    }
    resp.json::<Value>().await.map_err(|e| anyhow!("{e}"))
}

pub async fn run_antigravity() -> Result<()> {
    use tokio::io::AsyncReadExt;
    let mut input = String::new();
    tokio::io::stdin().read_to_string(&mut input).await?;
    if let Ok(payload) = serde_json::from_str::<Value>(&input) {
        if payload.get("initialNumSteps").is_some() {
            let conversation = payload
                .get("conversationId")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("Missing native conversation identity"))?;
            let step = payload
                .get("initialNumSteps")
                .and_then(Value::as_i64)
                .filter(|step| *step >= 0)
                .ok_or_else(|| anyhow!("Missing native invocation step"))?;
            let model = payload
                .get("modelName")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("Missing native invocation model"))?;
            let identity = crate::store::InvocationIdentity {
                harness: "antigravity".into(),
                model: model.to_string(),
                provider: None,
            };
            // A sub-agent's conversation is not the chat's native session; the chat id still owns it.
            let owner = std::env::var("ORX_SESSION_ID").unwrap_or_else(|_| conversation.into());
            record_invocation(
                &crate::store::Store::open()?,
                &payload,
                conversation,
                step,
                &identity,
                &owner,
                std::env::var("ORX_USAGE_EXECUTION_ID").ok().as_deref(),
            )?;
            println!("{{}}");
            return Ok(());
        }
    }
    let decision = antigravity_decision(&input).await.unwrap_or_else(|error| {
        json!({"decision": "deny", "reason": format!("OpenResearch approval bridge unavailable: {error}")})
    });
    println!("{decision}");
    Ok(())
}

/// Persist one native invocation's identity under its planner step (the key its stream usage
/// uses), its tool parts (crash-safe without harness memory), and, given the open execution, as
/// identity samples plus one `child_model_unknown` marker per child each spawn created.
pub(crate) fn record_invocation(
    store: &crate::store::Store,
    payload: &Value,
    conversation: &str,
    initial_steps: i64,
    identity: &crate::store::InvocationIdentity,
    owner: &str,
    execution: Option<&str>,
) -> Result<()> {
    use crate::local::harness::antigravity::{
        invocation_planner, invocation_sample_id, planner_tool_steps, requested_subagents,
        spawn_sample_id, spawned_children, tool_part_id, transcript_rows, unmatched_invocation_id,
    };
    let rows = payload
        .get("transcriptPath")
        .and_then(Value::as_str)
        .map(|path| transcript_rows(std::path::Path::new(path)))
        .unwrap_or_default();
    let planner = invocation_planner(&rows, initial_steps);
    let planner_step = planner.and_then(|row| row.get("step_index")?.as_i64());
    let sample = match planner_step {
        Some(step) => invocation_sample_id(conversation, step),
        // Explicit, never matched to a later row: a later invocation could own it.
        None => unmatched_invocation_id(conversation, initial_steps),
    };
    store.record_native_invocation(&sample, identity, Some(owner))?;
    let mut spawns = Vec::new();
    if let (Some(planner), Some(step)) = (planner, planner_step) {
        let calls = planner_tool_steps(&rows, planner);
        for (_, output) in &calls {
            if let Some(tool_step) = output.get("step_index").and_then(Value::as_i64) {
                store.record_native_invocation(
                    &tool_part_id(conversation, tool_step),
                    identity,
                    Some(owner),
                )?;
            }
        }
        let paired = !calls.is_empty();
        let spawn_calls = planner
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
            .filter(|(_, call)| {
                call.get("name").and_then(Value::as_str) == Some("invoke_subagent")
            });
        for (index, call) in spawn_calls {
            let children = if paired {
                spawned_children(&calls[index].1)
            } else {
                Vec::new()
            };
            let shortfall = requested_subagents(call).saturating_sub(children.len());
            spawns.extend(
                children
                    .into_iter()
                    .chain((0..shortfall).map(|missing| format!("#{index}.{missing}")))
                    .map(|child| spawn_sample_id(conversation, step, &child)),
            );
        }
    }
    let Some(execution) = execution else {
        return Ok(());
    };
    store.record_attributed_sample(
        execution,
        &sample,
        "antigravity",
        &crate::store::Attribution::native(
            "antigravity",
            Some(&identity.model),
            None,
            crate::store::Missing::IdentityNotReported,
        ),
        &Default::default(),
        false,
    )?;
    for spawn in spawns {
        store.record_attributed_sample(
            execution,
            &spawn,
            "antigravity",
            &crate::store::Attribution::Unresolved {
                reason: crate::store::Missing::ChildModelUnknown,
            },
            &Default::default(),
            false,
        )?;
    }
    Ok(())
}

async fn antigravity_decision(input: &str) -> Result<Value> {
    if std::env::var("ORX_AGY_GATE").as_deref() == Ok("bypass") {
        return Ok(json!({"decision": "allow"}));
    }
    let payload: Value = serde_json::from_str(input)?;
    let name = payload
        .pointer("/toolCall/name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| anyhow!("missing tool name"))?;
    let args = payload
        .pointer("/toolCall/args")
        .ok_or_else(|| anyhow!("missing tool arguments"))?;
    let env = GateEnv::from_env()?;
    if workspace_read(name, args, &payload) {
        return Ok(json!({"decision": "allow"}));
    }
    let (tool, input) = crate::local::harness::antigravity::normalize_tool(name, Some(args));
    let http = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(3))
        .build()?;
    let decision = relay(
        &http,
        &format!("http://127.0.0.1:{}/api/internal/permissions", env.up_port),
        json!({"sessionId":env.session_id,"token":env.token,"toolName":tool,"toolInput":input}),
    )
    .await?;
    Ok(match decision.get("behavior").and_then(Value::as_str) {
        Some("allow") => json!({"decision":"allow"}),
        _ => {
            json!({"decision":"deny", "reason":decision.get("message").and_then(Value::as_str).unwrap_or("Action denied")})
        }
    })
}

fn workspace_read(name: &str, args: &Value, payload: &Value) -> bool {
    let key = match name {
        "view_file" => "AbsolutePath",
        "list_dir" => "DirectoryPath",
        "grep_search" => "SearchPath",
        "find_by_name" => "SearchDirectory",
        _ => return false,
    };
    let Some(path) = args
        .get(key)
        .and_then(Value::as_str)
        .and_then(|path| std::fs::canonicalize(path).ok())
    else {
        return false;
    };
    payload
        .get("workspacePaths")
        .and_then(Value::as_array)
        .is_some_and(|roots| {
            roots
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|root| std::fs::canonicalize(root).ok())
                .any(|root| path.starts_with(root))
        })
}

#[cfg(test)]
mod antigravity_tests {
    use super::*;

    #[test]
    fn only_known_reads_inside_the_workspace_skip_approval() {
        let root = std::env::current_dir().unwrap();
        let payload = json!({"workspacePaths":[root]});
        assert!(workspace_read(
            "view_file",
            &json!({"AbsolutePath":root.join("Cargo.toml")}),
            &payload
        ));
        assert!(!workspace_read(
            "view_file",
            &json!({"AbsolutePath":"/etc/passwd"}),
            &payload
        ));
        assert!(!workspace_read(
            "run_command",
            &json!({"AbsolutePath":root}),
            &payload
        ));
        assert!(!workspace_read("view_file", &json!({}), &payload));
    }

    fn write_rows(path: &std::path::Path, rows: &[Value]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            path,
            rows.iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n")
                + "\n",
        )
        .unwrap();
    }

    fn samples(dir: &std::path::Path) -> Vec<(String, crate::store::Attribution)> {
        let conn = rusqlite::Connection::open(dir.join("orx.db")).unwrap();
        let rows = conn
            .prepare(
                "SELECT sample_id, attribution_json FROM chat_usage_samples ORDER BY sample_id",
            )
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get::<_, String>(1)?)))
            .unwrap()
            .map(|row| {
                let (id, json) = row.unwrap();
                (id, serde_json::from_str(&json).unwrap())
            })
            .collect();
        rows
    }

    fn identity(model: &str) -> crate::store::InvocationIdentity {
        crate::store::InvocationIdentity {
            harness: "antigravity".into(),
            model: model.into(),
            provider: None,
        }
    }

    fn exact(model: &str) -> crate::store::Attribution {
        crate::store::Attribution::Exact {
            model: model.into(),
            provider: None,
        }
    }

    /// Native agy 1.2.14 shape: system/user steps are inserted at or after `initialNumSteps`, so the
    /// invocation is its first planner from there; a failed invocation with no planner stays an
    /// explicit unmatched identity and never claims the next invocation's planner.
    #[test]
    fn invocation_hook_maps_its_own_planner_and_never_a_later_one() {
        let dir = std::env::temp_dir().join(format!("orx-agy-hook-{}", uuid::Uuid::new_v4()));
        let store = crate::store::Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "antigravity")
            .unwrap();
        let transcript = dir.join("brain/conv/logs/transcript_full.jsonl");
        let mut rows = vec![
            json!({"step_index":2,"type":"USER_INPUT","status":"DONE"}),
            json!({"step_index":3,"type":"SYSTEM_MESSAGE","status":"DONE"}),
            json!({"step_index":4,"type":"PLANNER_RESPONSE","status":"DONE","tool_calls":[
                {"name":"run_command","args":{"CommandLine":"orx exp run e"}}]}),
            json!({"step_index":5,"type":"GENERIC","status":"DONE","content":"run 1b2c"}),
        ];
        write_rows(&transcript, &rows);
        let payload = json!({"transcriptPath": transcript});
        let hook = |initial, model: &str| {
            record_invocation(
                &store,
                &payload,
                "conv",
                initial,
                &identity(model),
                "session",
                Some("execution"),
            )
            .unwrap()
        };
        hook(3, "gemini-3.1-pro-high");
        // The next invocation fails before any model output.
        rows.push(json!({"step_index":6,"type":"USER_INPUT","status":"DONE"}));
        write_rows(&transcript, &rows);
        hook(6, "claude-sonnet-4-6");
        rows.push(json!({"step_index":7,"type":"SYSTEM_MESSAGE","status":"DONE"}));
        rows.push(json!({"step_index":8,"type":"PLANNER_RESPONSE","status":"DONE","content":"ok"}));
        write_rows(&transcript, &rows);
        hook(7, "gemini-3.8-flash-high");
        let recorded = |key: &str| {
            store
                .native_invocation_identity("antigravity", key)
                .unwrap()
                .map(|identity| identity.model)
        };
        assert_eq!(
            recorded("antigravity:conv:step:4").as_deref(),
            Some("gemini-3.1-pro-high")
        );
        assert_eq!(recorded("antigravity:conv:step:3"), None);
        assert_eq!(
            recorded("tool-conv-5").as_deref(),
            Some("gemini-3.1-pro-high")
        );
        assert_eq!(
            recorded("antigravity:conv:step:8").as_deref(),
            Some("gemini-3.8-flash-high")
        );
        drop(store);
        assert_eq!(
            samples(&dir),
            [
                (
                    "antigravity:conv:invocation:6".into(),
                    exact("claude-sonnet-4-6")
                ),
                (
                    "antigravity:conv:step:4".into(),
                    exact("gemini-3.1-pro-high")
                ),
                (
                    "antigravity:conv:step:8".into(),
                    exact("gemini-3.8-flash-high")
                ),
            ]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// One native `invoke_subagent` creating two children (output shape observed live, compact
    /// transcript with JSON-encoded args): each child gets its own spawn marker and its own
    /// identity per invocation; a requested child the output never named stays explicit.
    #[test]
    fn spawns_link_each_named_child_and_keep_distinct_child_identities() {
        let dir = std::env::temp_dir().join(format!("orx-agy-children-{}", uuid::Uuid::new_v4()));
        let store = crate::store::Store::open_at(dir.clone()).unwrap();
        store
            .begin_usage_execution("execution", "turn", "antigravity")
            .unwrap();
        let parent = dir.join("parent/transcript.jsonl");
        let subagents =
            json!([{"Model":"inherit","Prompt":"reply A"},{"Model":"inherit","Prompt":"reply B"},
            {"Model":"inherit","Prompt":"reply C"}])
            .to_string();
        write_rows(
            &parent,
            &[
                json!({"step_index":10,"type":"SYSTEM_MESSAGE","status":"DONE"}),
                json!({"step_index":11,"type":"PLANNER_RESPONSE","status":"DONE","tool_calls":[
                {"name":"invoke_subagent","args":{"Subagents":subagents,"toolAction":"\"Invoking\""}}]}),
                json!({"step_index":12,"type":"GENERIC","status":"DONE","content":
                "Created At: x\nCompleted At: x\nCreated the following subagents:\n{\n  \"conversationId\":  \"child-a\",\n  \"logAbsoluteUri\":  \"file:///a\",\n  \"workspaceUris\":  []\n}\n{\n  \"conversationId\":  \"child-b\",\n  \"logAbsoluteUri\":  \"file:///b\",\n  \"workspaceUris\":  []\n}\nThe subagents will send you a message when they have completed their task."}),
            ],
        );
        record_invocation(
            &store,
            &json!({"transcriptPath": parent}),
            "parent",
            10,
            &identity("gemini-3.8-flash-high"),
            "session",
            Some("execution"),
        )
        .unwrap();
        // Child A changes model between invocations; child B has its own; child C never hooks.
        let child_rows = |dir: &str| {
            let path = std::path::Path::new(dir).join("transcript_full.jsonl");
            write_rows(
                &path,
                &[
                    json!({"step_index":0,"type":"SYSTEM_MESSAGE","status":"DONE"}),
                    json!({"step_index":1,"type":"PLANNER_RESPONSE","status":"DONE","content":"ok"}),
                    json!({"step_index":2,"type":"SYSTEM_MESSAGE","status":"DONE"}),
                    json!({"step_index":3,"type":"PLANNER_RESPONSE","status":"DONE","content":"ok"}),
                ],
            );
            json!({"transcriptPath": path})
        };
        for (child, initial, model) in [
            ("child-a", 0, "gemini-3.1-pro-high"),
            ("child-a", 2, "claude-sonnet-4-6"),
            ("child-b", 0, "gemini-3.8-flash-high"),
        ] {
            record_invocation(
                &store,
                &child_rows(dir.join(child).to_str().unwrap()),
                child,
                initial,
                &identity(model),
                "session",
                Some("execution"),
            )
            .unwrap();
        }
        drop(store);
        let unknown = crate::store::Attribution::Unresolved {
            reason: crate::store::Missing::ChildModelUnknown,
        };
        assert_eq!(
            samples(&dir),
            [
                (
                    "antigravity:child-a:step:1".into(),
                    exact("gemini-3.1-pro-high")
                ),
                (
                    "antigravity:child-a:step:3".into(),
                    exact("claude-sonnet-4-6")
                ),
                (
                    "antigravity:child-b:step:1".into(),
                    exact("gemini-3.8-flash-high")
                ),
                (
                    "antigravity:parent:step:11".into(),
                    exact("gemini-3.8-flash-high")
                ),
                (
                    "antigravity:parent:step:11:subagent:#0.0".into(),
                    unknown.clone()
                ),
                (
                    "antigravity:parent:step:11:subagent:child-a".into(),
                    unknown.clone()
                ),
                (
                    "antigravity:parent:step:11:subagent:child-b".into(),
                    unknown
                ),
            ]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn bypass_env_var_allows_immediately() {
        std::env::set_var("ORX_AGY_GATE", "bypass");
        let res = antigravity_decision("invalid-json").await.unwrap();
        assert_eq!(res["decision"], "allow");
        std::env::remove_var("ORX_AGY_GATE");
    }
}
