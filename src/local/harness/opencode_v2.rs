//! V2 admits prompts before execution finishes; reconcile its durable projection.
use super::*;
use crate::local::opencode::AgentEndpoint;
use futures::future::{BoxFuture, FutureExt};

async fn get(endpoint: &AgentEndpoint, path: &str) -> Result<Value> {
    Ok(endpoint
        .client
        .get(format!("{}{path}", endpoint.base_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

async fn post(endpoint: &AgentEndpoint, path: &str, body: &Value) -> Result<()> {
    endpoint
        .client
        .post(format!("{}{path}", endpoint.base_url))
        .json(body)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

pub(super) async fn run_turn(
    ctx: &mut TurnCtx,
    store: NativeStore,
    binary: crate::local::opencode::ResolvedBinary,
    database: PathBuf,
) -> Result<()> {
    ensure_runtime(ctx, store, binary, database).await?;
    let endpoint = ctx
        .host
        .opencode
        .endpoint_for(&ctx.session_id)
        .await
        .ok_or_else(|| anyhow!("OpenCode stopped during setup"))?;
    let mut model = ctx
        .model
        .as_deref()
        .and_then(|id| id.split_once('/'))
        .map(|(provider, id)| json!({"providerID":provider,"id":id}));
    if let (Some(model), Some(variant)) = (
        model.as_mut(),
        opencode_variant(ctx.reasoning_level.as_deref()),
    ) {
        model["variant"] = json!(variant);
    }
    let native_id = if let Some(id) = &ctx.native_session_id {
        get(&endpoint, &format!("/api/session/{id}")).await?;
        id.clone()
    } else {
        let directory =
            crate::local::git::existing_session_worktree_path(&ctx.project, &ctx.session_id);
        let mut body =
            json!({"agent":opencode_agent(ctx.plan_mode),"location":{"directory":directory}});
        if let Some(model) = &model {
            body["model"] = model.clone();
        }
        let session: Value = endpoint
            .client
            .post(format!("{}/api/session", endpoint.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let id = session
            .pointer("/data/id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("OpenCode V2 session response has no id"))?
            .to_owned();
        ctx.persist_native_session_id(&id)?;
        id
    };
    let path = format!("/api/session/{native_id}");
    post(
        &endpoint,
        &format!("{path}/agent"),
        &json!({"agent":opencode_agent(ctx.plan_mode)}),
    )
    .await?;
    if let Some(model) = model {
        post(&endpoint, &format!("{path}/model"), &json!({"model":model})).await?;
    }
    let before = get(&endpoint, &endpoint.v2_export_path(&native_id)).await?;
    let previous: HashSet<String> = messages(&before)?
        .iter()
        .filter_map(|m| m["id"].as_str().map(str::to_owned))
        .collect();
    let prompt_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let started_at = crate::store::now_ms();
    let mut captured = Captured::default();
    track_turn(ctx, &native_id, started_at);
    ctx.persist_delivery(DeliveryState::Unknown)?;
    // A transport failure can follow durable admission. Never replay this POST.
    let admission = post(
        &endpoint,
        &format!("{path}/prompt"),
        &json!({"id":prompt_id,"text":ctx.text}),
    )
    .await;
    if admission.is_ok() {
        ctx.persist_delivery(DeliveryState::Accepted)?;
    }
    let mut surfaced = HashSet::new();
    let mut was_idle = false;
    let result: Result<()> = async {
        loop {
            // ponytail: full projected history polling; switch to paged durable log if long chats make this costly.
            let projection = get(&endpoint, &endpoint.v2_export_path(&native_id)).await?;
            let current = messages(&projection)?;
            let delivered = current.iter().any(|m| m["id"].as_str() == Some(&prompt_id));
            merge_projection(
                ctx,
                &endpoint,
                &native_id,
                current,
                &previous,
                started_at,
                &mut captured,
            )
            .await?;
            if delivered && ctx.delivery_state() != DeliveryState::Accepted {
                ctx.persist_delivery(DeliveryState::Accepted)?;
            }
            prompts(ctx, &endpoint, &path, &mut surfaced).await?;
            let inbox = get(&endpoint, &format!("{path}/inbox")).await?;
            let queued = inbox["data"]
                .as_array()
                .ok_or_else(|| anyhow!("OpenCode V2 inbox response is invalid"))?
                .iter()
                .any(|item| item["id"].as_str() == Some(&prompt_id));
            let active = get(&endpoint, "/api/session/active").await?;
            let active = active["data"]
                .as_object()
                .ok_or_else(|| anyhow!("OpenCode V2 active response is invalid"))?;
            for session in captured.sessions(&native_id) {
                if active.contains_key(&session) {
                    observe_retry(
                        &*ctx,
                        &endpoint,
                        &session,
                        session != native_id,
                        &mut captured,
                    )
                    .await;
                }
            }
            let running = active.contains_key(&native_id);
            if observed_idle(&mut was_idle, queued, running) {
                let final_projection = get(&endpoint, &endpoint.v2_export_path(&native_id)).await?;
                let final_messages = messages(&final_projection)?;
                let delivered = delivered
                    || final_messages
                        .iter()
                        .any(|m| m["id"].as_str() == Some(&prompt_id));
                let answered = merge_projection(
                    ctx,
                    &endpoint,
                    &native_id,
                    final_messages,
                    &previous,
                    started_at,
                    &mut captured,
                )
                .await?;
                if delivered
                    && answered
                    && final_projection
                        .pointer("/data/info/outcome")
                        .and_then(Value::as_str)
                        == Some("succeeded")
                {
                    ctx.persist_delivery(DeliveryState::Accepted)?;
                    ctx.mark_final_text_tail();
                    return Ok(());
                }
                if let Err(error) = admission {
                    return Err(error);
                }
                return Err(anyhow!(
                    "OpenCode V2 became idle without completing this prompt"
                ));
            }
            ctx.flush()?;
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }
    .await;
    let roots: Vec<(String, String)> = captured
        .background
        .iter()
        .filter_map(|child| Some((child.clone(), captured.descendants.get(child)?.clone())))
        .collect();
    if let Some(sink) = (!roots.is_empty())
        .then(|| hold_execution(ctx, &native_id, started_at, json!(roots)))
        .flatten()
    {
        let native = native_id.clone();
        tokio::spawn(async move {
            watch_background(&sink, &endpoint, &native, &roots, started_at).await;
            sink.release();
        });
    }
    result
}

fn observed_idle(was_idle: &mut bool, queued: bool, running: bool) -> bool {
    let idle = !queued && !running;
    let settled = *was_idle && idle;
    *was_idle = idle;
    settled
}

async fn merge_projection(
    ctx: &mut TurnCtx,
    endpoint: &AgentEndpoint,
    native_id: &str,
    messages: &[Value],
    previous: &HashSet<String>,
    started_at: i64,
    captured: &mut Captured,
) -> Result<bool> {
    let mut answered = false;
    let woken = woken_runs(messages);
    for message in messages
        .iter()
        .filter(|m| m["id"].as_str().is_some_and(|id| !previous.contains(id)))
    {
        // A run another execution's subagent result woke is that execution's to account.
        if foreign_run(message, &woken, captured) {
            for (part, _) in projected_parts(message) {
                ctx.upsert_part_preserving_children(part);
            }
            continue;
        }
        capture(ctx, message, false, captured);
        if message["type"] != "assistant" {
            continue;
        }
        if message.get("retry").is_none_or(Value::is_null)
            && message
                .pointer("/time/completed")
                .is_some_and(|value| !value.is_null())
        {
            if let Some(error) = message.get("error").filter(|error| !error.is_null()) {
                let message = error_text(error);
                ctx.mark_native_retry_exhausted();
                ctx.mark_terminal_failure("opencode_terminal", &message);
                return Err(anyhow!("OpenCode V2: {message}"));
            }
        }
        answered |= message
            .pointer("/time/completed")
            .is_some_and(|v| !v.is_null());
        if let Some(used) = opencode_used_tokens(message.get("tokens")) {
            ctx.report_usage(ContextUsage {
                used_tokens: used,
                context_window: None,
            });
        }
        if let Some(retry) = message.get("retry").filter(|retry| !retry.is_null()) {
            ctx.show_retry_status(
                "native",
                &error_text(&retry["error"]),
                retry["attempt"].as_i64().unwrap_or(1),
                None,
                retry["at"].as_i64(),
            );
        } else {
            ctx.clear_retry_status();
        }
        let mut parts = projected_parts(message);
        for (part, content) in &mut parts {
            if let Some(child) = subagent_session(content, native_id, captured) {
                part.children =
                    subagent_children(&*ctx, endpoint, native_id, child, started_at, captured)
                        .await?;
            }
        }
        for (part, _) in parts {
            ctx.upsert_part_preserving_children(part);
        }
    }
    Ok(answered)
}

/// Native evidence already recorded, so repeated polls of an unchanged projection write nothing.
#[derive(Default)]
struct Captured {
    samples: HashMap<String, String>,
    invokers: HashSet<String>,
    /// This turn's subagent sessions at any depth, each with its parent session.
    descendants: HashMap<String, String>,
    /// Background subagents spawned this turn, which can outlive it.
    background: HashSet<String>,
}

impl Captured {
    fn sessions(&self, native_id: &str) -> Vec<String> {
        std::iter::once(native_id.to_string())
            .chain(self.descendants.keys().cloned())
            .collect()
    }

    fn record(
        &mut self,
        sink: &dyn UsageSink,
        id: &str,
        attribution: crate::store::Attribution,
        (usage, complete): (crate::store::TokenUsage, bool),
    ) {
        let fingerprint = json!([attribution, usage, complete]).to_string();
        if self.samples.insert(id.to_owned(), fingerprint.clone()) != Some(fingerprint) {
            sink.sample(id, attribution, usage, complete);
        }
    }
}

/// The session a settled `subagent` tool part of `parent` ran, noting a background one.
fn subagent_session<'a>(
    content: &'a Value,
    parent: &str,
    captured: &mut Captured,
) -> Option<&'a str> {
    if content["type"] != "tool" || content["name"] != "subagent" {
        return None;
    }
    let child = content.pointer("/state/metadata/sessionID")?.as_str()?;
    captured
        .descendants
        .insert(child.to_owned(), parent.to_owned());
    if content.pointer("/state/metadata/status") == Some(&json!("running")) {
        captured.background.insert(child.to_owned());
    }
    Some(child)
}

/// A subagent's transcript with its own subagents nested, capturing this turn's steps at every
/// depth. A continued subagent session also holds earlier turns' messages.
fn subagent_children<'a>(
    sink: &'a dyn UsageSink,
    endpoint: &'a AgentEndpoint,
    parent_id: &'a str,
    child_id: &'a str,
    started_at: i64,
    captured: &'a mut Captured,
) -> BoxFuture<'a, Result<Vec<WirePart>>> {
    async move {
        let child = get(endpoint, &endpoint.v2_export_path(child_id)).await?;
        if child.pointer("/data/info/parentID").and_then(Value::as_str) != Some(parent_id) {
            return Err(anyhow!("OpenCode returned an unrelated subagent session"));
        }
        let mut children = Vec::new();
        for message in messages(&child)? {
            let current = message
                .pointer("/time/created")
                .and_then(Value::as_i64)
                .is_some_and(|created| created >= started_at);
            if current {
                capture(sink, message, true, captured);
            }
            if message["type"] != "assistant" {
                continue;
            }
            for (mut part, content) in projected_parts(message) {
                if let Some(grandchild) =
                    subagent_session(content, child_id, captured).filter(|_| current)
                {
                    part.children = subagent_children(
                        sink, endpoint, child_id, grandchild, started_at, captured,
                    )
                    .await?;
                }
                children.push(part);
            }
        }
        Ok(children)
    }
    .boxed()
}

/// V2 runs one provider step per assistant message, and its `model` is the step's resolved
/// native model. Compactions carry their own model and tokens.
fn capture(sink: &dyn UsageSink, message: &Value, child: bool, captured: &mut Captured) {
    let (Some(id), Some("assistant" | "compaction")) =
        (message["id"].as_str(), message["type"].as_str())
    else {
        return;
    };
    let attribution = crate::store::Attribution::native(
        "opencode",
        message.pointer("/model/id").and_then(Value::as_str),
        message.pointer("/model/providerID").and_then(Value::as_str),
        opencode_missing(child),
    );
    if let Some(sample) = executed_usage(message) {
        captured.record(sink, id, attribution.clone(), sample);
    }
    if let crate::store::Attribution::Exact { model, provider } = &attribution {
        for (part, content) in projected_parts(message) {
            if content["type"] == "tool" && captured.invokers.insert(part.id.clone()) {
                sink.invoker(&part.id, model, provider.as_deref());
            }
        }
    }
}

/// A step that failed before the provider responded never executed; one with tokens, a streamed
/// response or output did.
fn executed_usage(message: &Value) -> Option<(crate::store::TokenUsage, bool)> {
    let tokens = message.get("tokens").filter(|tokens| !tokens.is_null());
    (tokens.is_some()
        || message
            .pointer("/time/streamed")
            .is_some_and(|t| !t.is_null())
        || message["content"]
            .as_array()
            .is_some_and(|content| !content.is_empty()))
    .then(|| opencode_sample(tokens))
}

/// A native retry reuses the assistant message and never persists a failed attempt's counters;
/// only the unsettled message shows that attempt's model, so record it as a known request.
async fn observe_retry(
    sink: &dyn UsageSink,
    endpoint: &AgentEndpoint,
    session: &str,
    child: bool,
    captured: &mut Captured,
) {
    let path = format!("/api/session/{session}/message?type=assistant&order=desc&limit=1");
    let Ok(latest) = get(endpoint, &path).await else {
        return;
    };
    let Some(message) = latest.pointer("/data/0") else {
        return;
    };
    if let (Some(id), Some(attempt)) = (
        message["id"].as_str(),
        message.pointer("/retry/attempt").and_then(Value::as_i64),
    ) {
        let attribution = crate::store::Attribution::native(
            "opencode",
            message.pointer("/model/id").and_then(Value::as_str),
            message.pointer("/model/providerID").and_then(Value::as_str),
            opencode_missing(child),
        );
        captured.record(
            sink,
            &format!("{id}:attempt:{attempt}"),
            attribution,
            (crate::store::TokenUsage::default(), false),
        );
    }
}

/// V2 interrupts asynchronously and spares background subagents; stop those too, then capture the
/// settled tree before the shared interrupt finalizes usage.
pub(super) async fn capture_interrupted(
    endpoint: &AgentEndpoint,
    sink: &dyn UsageSink,
    native_id: &str,
    started_at: i64,
) {
    let mut captured = Captured::default();
    wait_idle(endpoint, &[native_id.to_string()]).await;
    let _ = capture_tree(sink, endpoint, native_id, started_at, &mut captured, false).await;
    let background: Vec<String> = captured.background.iter().cloned().collect();
    if !background.is_empty() {
        for session in &background {
            let _ = post(
                endpoint,
                &format!("/api/session/{session}/interrupt"),
                &json!({}),
            )
            .await;
        }
        wait_idle(endpoint, &background).await;
        // Each cancelled subagent reports back and wakes its parent; stop that run as well.
        let _ = post(
            endpoint,
            &format!("/api/session/{native_id}/interrupt"),
            &json!({}),
        )
        .await;
        wait_idle(endpoint, &captured.sessions(native_id)).await;
        let _ = capture_tree(sink, endpoint, native_id, started_at, &mut captured, false).await;
    }
    for session in captured.sessions(native_id) {
        observe_retry(
            sink,
            endpoint,
            &session,
            session != native_id,
            &mut captured,
        )
        .await;
    }
}

/// Keeps capturing background subagents after their turn ended, and the parent runs their results
/// natively wake with no app turn, until none of the tree is active and each result was delivered.
pub(super) async fn watch_background(
    sink: &dyn UsageSink,
    endpoint: &AgentEndpoint,
    native_id: &str,
    roots: &[(String, String)],
    started_at: i64,
) {
    let mut captured = Captured {
        background: roots.iter().map(|(child, _)| child.clone()).collect(),
        ..Default::default()
    };
    let mut waited = 0;
    loop {
        // Read before capturing, so a session idle here has settled everything captured below.
        let active = get(endpoint, "/api/session/active").await.ok();
        for (child, parent) in roots {
            let _ =
                subagent_children(sink, endpoint, parent, child, started_at, &mut captured).await;
        }
        // Includes the woken parent runs (only those this execution's subagents woke).
        let _ = capture_tree(sink, endpoint, native_id, started_at, &mut captured, true).await;
        let delivered: HashSet<String> = get(endpoint, &endpoint.v2_export_path(native_id))
            .await
            .ok()
            .and_then(|projection| Some(woken_deliveries(messages(&projection).ok()?).collect()))
            .unwrap_or_default();
        let busy = active.is_some_and(|active| {
            std::iter::once(native_id)
                .chain(captured.descendants.keys().map(String::as_str))
                .chain(roots.iter().map(|(child, _)| child.as_str()))
                .any(|session| active["data"].get(session).is_some())
        });
        let undelivered = captured
            .background
            .iter()
            .any(|child| !delivered.contains(child));
        if !busy && (!undelivered || waited >= super::DELIVERY_GRACE_POLLS) {
            break;
        }
        if !busy {
            waited += 1;
        }
        tokio::time::sleep(BACKGROUND_POLL).await;
    }
}

/// Assistant runs a subagent's delivered result woke (native `synthetic` message with
/// `metadata.childID`, then the parent's run until the next user message) → that subagent.
fn woken_runs(messages: &[Value]) -> HashMap<String, String> {
    let mut woken = HashMap::new();
    let mut by = None;
    for message in messages {
        match message["type"].as_str() {
            Some("user") => by = None,
            Some("synthetic") => {
                by = (message.pointer("/metadata/source") == Some(&json!("subagent")))
                    .then(|| message.pointer("/metadata/childID")?.as_str())
                    .flatten()
                    .map(str::to_owned);
            }
            _ => {
                if let (Some(child), Some(id)) = (&by, message["id"].as_str()) {
                    woken.insert(id.to_owned(), child.clone());
                }
            }
        }
    }
    woken
}

fn woken_deliveries(messages: &[Value]) -> impl Iterator<Item = String> + '_ {
    messages
        .iter()
        .filter(|message| message["type"] == "synthetic")
        .filter(|message| message.pointer("/metadata/source") == Some(&json!("subagent")))
        .filter_map(|message| Some(message.pointer("/metadata/childID")?.as_str()?.to_owned()))
}

fn foreign_run(message: &Value, woken: &HashMap<String, String>, captured: &Captured) -> bool {
    message["id"]
        .as_str()
        .and_then(|id| woken.get(id))
        .is_some_and(|child| !captured.background.contains(child))
}

/// `woken_only`: after the turn, only runs its own subagents' results woke (later turns' own
/// messages belong to them).
async fn capture_tree(
    sink: &dyn UsageSink,
    endpoint: &AgentEndpoint,
    native_id: &str,
    started_at: i64,
    captured: &mut Captured,
    woken_only: bool,
) -> Result<()> {
    let projection = get(endpoint, &endpoint.v2_export_path(native_id)).await?;
    let all = messages(&projection)?;
    let woken = woken_runs(all);
    for message in all.iter().filter(|m| {
        m.pointer("/time/created")
            .and_then(Value::as_i64)
            .is_some_and(|created| created >= started_at)
    }) {
        let is_woken = woken.contains_key(message["id"].as_str().unwrap_or_default());
        if foreign_run(message, &woken, captured) || (woken_only && !is_woken) {
            continue;
        }
        if is_woken {
            for (part, _) in projected_parts(message) {
                if part.kind == "tool" {
                    sink.tool_evidence(&part);
                }
            }
        }
        capture(sink, message, false, captured);
        for content in message["content"].as_array().into_iter().flatten() {
            if let Some(child) = subagent_session(content, native_id, captured) {
                subagent_children(sink, endpoint, native_id, child, started_at, captured).await?;
            }
        }
    }
    Ok(())
}

/// Waits (bounded) until none of `sessions` is natively active.
async fn wait_idle(endpoint: &AgentEndpoint, sessions: &[String]) {
    for _ in 0..20 {
        match get(endpoint, "/api/session/active").await {
            Ok(active)
                if !sessions
                    .iter()
                    .any(|session| active["data"].get(session).is_some()) =>
            {
                return
            }
            Err(_) => return,
            Ok(_) => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}

fn messages(projection: &Value) -> Result<&Vec<Value>> {
    projection
        .pointer("/data/messages")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("OpenCode V2 history response is invalid"))
}

fn error_text(error: &Value) -> String {
    error
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| error.to_string())
}

fn wire_parts(message: &Value) -> Vec<WirePart> {
    projected_parts(message)
        .into_iter()
        .map(|(part, _)| part)
        .collect()
}

fn projected_parts(message: &Value) -> Vec<(WirePart, &Value)> {
    let Some(id) = message["id"].as_str() else {
        return Vec::new();
    };
    message["content"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, content)| {
            let mut part = content.clone();
            part["id"] = json!(format!("{id}:{index}"));
            if content["type"] == "tool" {
                part["tool"] = content["name"].clone();
                let state = &content["state"];
                if state["status"] == "streaming" {
                    part["state"]["status"] = json!("running");
                }
                if let Some(error) = state.get("error") {
                    part["state"]["error"] = json!(error_text(error));
                }
                if let Some(output) = state["content"].as_array() {
                    part["state"]["output"] = json!(output
                        .iter()
                        .filter_map(|item| item["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n"));
                }
            }
            to_wire_part(&part).map(|part| (part, content))
        })
        .collect()
}

async fn prompts(
    ctx: &mut TurnCtx,
    endpoint: &AgentEndpoint,
    path: &str,
    surfaced: &mut HashSet<String>,
) -> Result<()> {
    let permissions = get(endpoint, &format!("{path}/permission")).await?;
    for request in permissions["data"].as_array().into_iter().flatten() {
        let Some(id) = request["id"].as_str() else {
            continue;
        };
        if surfaced.contains(id) {
            continue;
        }
        if opencode_auto_approve(ctx.permission_mode)
            && post(
                endpoint,
                &format!("{path}/permission/{id}/reply"),
                &json!({"reply":"always"}),
            )
            .await
            .is_ok()
        {
            surfaced.insert(id.to_owned());
            continue;
        }
        surface_card(
            ctx,
            WirePrompt {
                kind: "permission".into(),
                native_id: Some(id.into()),
                tool: request["action"].as_str().map(str::to_owned),
                tool_input: Some(request["resources"].clone()),
                ..Default::default()
            },
        );
        surfaced.insert(id.to_owned());
    }
    let forms = get(endpoint, &format!("{path}/form")).await?;
    for form in forms["data"].as_array().into_iter().flatten() {
        let Some(id) = form["id"].as_str() else {
            continue;
        };
        let fields = form["fields"]
            .as_array()
            .ok_or_else(|| anyhow!("OpenCode V2 form has no fields"))?;
        let mut answers = serde_json::Map::new();
        saved_form_answers(id, fields, &ctx.assistant.parts, &mut answers)?;
        for field in fields.iter().filter(|field| field_active(field, &answers)) {
            let key = field["key"]
                .as_str()
                .ok_or_else(|| anyhow!("OpenCode V2 form field has no key"))?;
            let native_id = serde_json::to_string(&(id, key))?;
            if surfaced.contains(&native_id) {
                continue;
            }
            surface_card(ctx, form_card(form, field, native_id.clone())?);
            surfaced.insert(native_id);
        }
    }
    let pending: HashSet<_> = forms["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|form| form["id"].as_str())
        .collect();
    for part in &mut ctx.assistant.parts {
        if let Some(prompt) = &mut part.prompt {
            if let Some(id) = &prompt.native_id {
                if let Ok((form, _)) = serde_json::from_str::<(String, String)>(id) {
                    if !pending.contains(form.as_str()) {
                        prompt.resolved = true;
                    }
                }
            }
        }
    }
    Ok(())
}

fn field_active(field: &Value, answers: &serde_json::Map<String, Value>) -> bool {
    field["when"]
        .as_array()
        .into_iter()
        .flatten()
        .all(|condition| {
            let Some(answer) = condition["key"].as_str().and_then(|key| answers.get(key)) else {
                return false;
            };
            let equal = answer
                .as_array()
                .map(|items| items.contains(&condition["value"]))
                .unwrap_or_else(|| answer == &condition["value"]);
            match condition["op"].as_str() {
                Some("eq") => equal,
                Some("neq") => !equal,
                _ => false,
            }
        })
}

fn form_card(form: &Value, field: &Value, native_id: String) -> Result<WirePrompt> {
    let mut options: Vec<_> = field["options"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|option| {
            Some(WireQuestionOption {
                label: option["label"].as_str()?.into(),
                description: option["description"].as_str().map(str::to_owned),
            })
        })
        .collect();
    if field["type"] == "boolean" {
        options = ["Yes", "No"]
            .into_iter()
            .map(|label| WireQuestionOption {
                label: label.into(),
                description: None,
            })
            .collect();
    }
    let question = field["title"]
        .as_str()
        .or_else(|| form["title"].as_str())
        .unwrap_or("Answer");
    let question = if field["type"] == "external" {
        format!(
            "{question}\n\n{}",
            field["url"].as_str().unwrap_or_default()
        )
    } else {
        question.to_owned()
    };
    if field["type"] == "external" {
        options = vec![WireQuestionOption {
            label: "Done".into(),
            description: None,
        }];
    }
    Ok(WirePrompt {
        kind: "question".into(),
        native_id: Some(native_id),
        header: form["title"].as_str().map(str::to_owned),
        question: Some(match field["description"].as_str() {
            Some(description) => format!("{question}\n\n{description}"),
            None => question,
        }),
        options,
        multi_select: field["type"] == "multiselect",
        ..Default::default()
    })
}

pub(super) async fn reply(
    ctx: &ResumeCtx,
    endpoint: &AgentEndpoint,
    prompt: &WirePrompt,
    answer: &PromptAnswer,
) -> Result<()> {
    let session = ctx
        .native_session_id
        .as_deref()
        .ok_or_else(|| anyhow!("OpenCode session has no native id"))?;
    let id = prompt
        .native_id
        .as_deref()
        .ok_or_else(|| anyhow!("OpenCode prompt has no id"))?;
    let path = format!("/api/session/{session}");
    if prompt.kind == "permission" {
        return post(
            endpoint,
            &format!("{path}/permission/{id}/reply"),
            &json!({"reply":if answer.approve {"always"} else {"reject"}}),
        )
        .await;
    }
    let (form_id, field_key): (String, String) = serde_json::from_str(id)?;
    let submitted = submitted_answers(&answer.answers, answer.note.as_ref());
    if submitted.is_empty() {
        if !endpoint.legacy_v2_api {
            endpoint
                .client
                .delete(format!("{}{path}/form/{form_id}", endpoint.base_url))
                .send()
                .await?
                .error_for_status()?;
            return Ok(());
        }
        return post(
            endpoint,
            &format!("{path}/form/{form_id}/cancel"),
            &json!({}),
        )
        .await;
    }
    let form = get(endpoint, &format!("{path}/form/{form_id}")).await?;
    let fields = form["data"]["fields"]
        .as_array()
        .ok_or_else(|| anyhow!("OpenCode V2 form has no fields"))?;
    let mut values = serde_json::Map::new();
    let messages = crate::store::Store::open()?.list_chat_messages(&ctx.session_id)?;
    for message in messages {
        let parts: Vec<WirePart> = serde_json::from_str(&message.parts_json)?;
        saved_form_answers(&form_id, fields, &parts, &mut values)?;
    }
    let field = fields
        .iter()
        .find(|field| field["key"].as_str() == Some(&field_key))
        .ok_or_else(|| anyhow!("OpenCode form field no longer exists"))?;
    values.insert(field_key, field_answer(field, submitted)?);
    let active: Vec<_> = fields
        .iter()
        .filter(|field| field_active(field, &values))
        .collect();
    if active.iter().all(|field| {
        field["key"]
            .as_str()
            .is_some_and(|key| values.contains_key(key))
    }) {
        values.retain(|key, _| {
            active
                .iter()
                .any(|field| field["key"].as_str() == Some(key) && field["type"] != "external")
        });
        post(
            endpoint,
            &format!("{path}/form/{form_id}/reply"),
            &json!({"answer":values}),
        )
        .await?;
    }
    Ok(())
}

fn saved_form_answers(
    form_id: &str,
    fields: &[Value],
    parts: &[WirePart],
    answers: &mut serde_json::Map<String, Value>,
) -> Result<()> {
    for part in parts {
        let Some(prompt) = part.prompt.as_ref().filter(|prompt| prompt.resolved) else {
            continue;
        };
        let Some(native_id) = &prompt.native_id else {
            continue;
        };
        let Ok((saved_form, key)) = serde_json::from_str::<(String, String)>(native_id) else {
            continue;
        };
        if saved_form != form_id {
            continue;
        }
        let submitted = submitted_answers(&prompt.answers, prompt.note.as_ref());
        if submitted.is_empty() {
            continue;
        }
        if let Some(field) = fields
            .iter()
            .find(|field| field["key"].as_str() == Some(&key))
        {
            answers.insert(key, field_answer(field, submitted)?);
        }
    }
    Ok(())
}

fn field_answer(field: &Value, answers: &[String]) -> Result<Value> {
    let values: Vec<_> = answers
        .iter()
        .map(|answer| {
            field["options"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|option| option["label"].as_str() == Some(answer))
                .and_then(|option| option["value"].as_str())
                .unwrap_or(answer)
                .to_owned()
        })
        .collect();
    let first = values
        .first()
        .ok_or_else(|| anyhow!("Missing form answer"))?;
    match field["type"].as_str() {
        Some("multiselect") => Ok(json!(values)),
        Some("boolean") => match first.to_ascii_lowercase().as_str() {
            "true" | "yes" => Ok(json!(true)),
            "false" | "no" => Ok(json!(false)),
            _ => Err(anyhow!("Choose Yes or No")),
        },
        Some("integer") => Ok(json!(first
            .parse::<i64>()
            .map_err(|_| anyhow!("Enter a whole number"))?)),
        Some("number") => {
            let number = first
                .parse::<f64>()
                .map_err(|_| anyhow!("Enter a number"))?;
            if !number.is_finite() {
                return Err(anyhow!("Enter a finite number"));
            }
            Ok(json!(number))
        }
        _ => Ok(json!(first)),
    }
}

fn reconnect_command(path: &Path) -> String {
    let path = path.to_string_lossy();
    #[cfg(windows)]
    {
        format!(
            "$env:OPENCODE_DB='{}'; opencode auth login",
            path.replace('\'', "''")
        )
    }
    #[cfg(not(windows))]
    {
        format!(
            "OPENCODE_DB='{}' opencode auth login",
            path.replace('\'', "'\"'\"'")
        )
    }
}

fn selectable_model(model: &Value, connected: &HashSet<&str>) -> bool {
    if model["enabled"] != true {
        return false;
    }
    // V2 scopes the catalog to available providers; anonymous Zen must also be explicitly free.
    model["providerID"] != "opencode"
        || connected.contains("opencode")
        || model["cost"].as_array().is_some_and(|tiers| {
            !tiers.is_empty()
                && tiers.iter().all(|cost| {
                    cost["input"].as_f64() == Some(0.0) && cost["output"].as_f64() == Some(0.0)
                })
        })
}

pub(super) async fn detect(
    binary: crate::local::opencode::ResolvedBinary,
    mut info: HarnessInfo,
) -> HarnessInfo {
    let catalog = tokio::spawn(async move {
        let db = native_store::prepare_opencode(NativeStore::Isolated)?;
        let _lease = crate::local::opencode::prepare_database(&binary, &db).await?;
        let mut cmd = tokio::process::Command::new(&binary.path);
        crate::local::local_models::prepare_env(&mut cmd, None)?;
        cmd.env("OPENCODE_DB", _lease.path())
            .env("OPENCODE_CONFIG_PROJECT_DISABLE", "1")
            .current_dir(
                db.parent()
                    .ok_or_else(|| anyhow!("OpenCode database has no parent"))?,
            );
        let (mut child, endpoint) = crate::local::opencode::start_server(&binary, cmd).await?;
        let result = discover_models(&endpoint).await;
        let _ = child.kill().await;
        let _ = child.wait().await;
        result
    })
    .await;
    match catalog {
        Ok(Ok((models, integrations))) => {
            let connected: HashSet<&str> = integrations["data"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|integration| {
                    integration["connections"]
                        .as_array()
                        .is_some_and(|items| !items.is_empty())
                })
                .filter_map(|integration| integration["id"].as_str())
                .collect();
            info.models = models["data"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|model| selectable_model(model, &connected))
                .filter_map(|model| {
                    let provider = model["providerID"].as_str()?;
                    let id = model["id"].as_str()?;
                    let variants: Vec<_> = model["variants"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|v| v["id"].as_str())
                        .collect();
                    Some(
                        ModelInfo::new(format!("{provider}/{id}"))
                            .with_label(model["name"].as_str(), None)
                            .with_reasoning(&variants),
                    )
                })
                .collect();
            info.authenticated = !connected.is_empty();
            info.agent_ready = !info.models.is_empty();
            info.auth_state = if info.authenticated || info.agent_ready {
                HarnessAuthState::Ready
            } else {
                HarnessAuthState::NeedsLogin
            };
            if !info.agent_ready {
                if info.authenticated {
                    info.agent_note = Some("OpenCode V2 listed no enabled models. Check its model configuration and re-check OpenCode.".into());
                } else {
                    info.agent_note = Some(format!(
                        "Connect a provider for OpenResearch's OpenCode V2 database: `{}`",
                        reconnect_command(&native_store::opencode_db(NativeStore::Isolated))
                    ));
                }
            }
        }
        Ok(Err(error)) => info.agent_note = Some(error.to_string()),
        Err(error) => info.agent_note = Some(format!("OpenCode discovery failed: {error}")),
    }
    info
}

async fn discover_models(endpoint: &AgentEndpoint) -> Result<(Value, Value)> {
    let mut catalog = None;
    let result = tokio::time::timeout(Duration::from_secs(30), async {
        if endpoint.legacy_v2_api {
            post(endpoint, "/api/plugin/await-activation", &json!({})).await?;
        }
        loop {
            let models = get(endpoint, "/api/model").await?;
            let integrations = get(endpoint, "/api/integration").await?;
            let ready = models["data"]
                .as_array()
                .is_some_and(|models| !models.is_empty());
            catalog = Some((models, integrations));
            if endpoint.legacy_v2_api || ready {
                return Ok::<_, anyhow::Error>(());
            }
            // OpenCode 2.0.4 removed await-activation; cold catalogs can initially be empty.
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    })
    .await;
    if let Ok(result) = result {
        result?;
    }
    catalog.ok_or_else(|| anyhow!("OpenCode model discovery timed out"))
}

pub(super) async fn generate(
    binary: crate::local::opencode::ResolvedBinary,
    model: Option<String>,
    prompt: String,
    timeout: Duration,
) -> Result<String> {
    tokio::spawn(async move {
        let db = native_store::prepare_opencode(NativeStore::Isolated)?;
        let lease = crate::local::opencode::prepare_database(&binary, &db).await?;
        let mut cmd = tokio::process::Command::new(&binary.path);
        crate::local::local_models::prepare_env(&mut cmd, model.as_deref())?;
        cmd.env("OPENCODE_DB", lease.path())
            .env("OPENCODE_CONFIG_PROJECT_DISABLE", "1")
            .current_dir(std::env::temp_dir());
        let (mut child, endpoint) = crate::local::opencode::start_server(&binary, cmd).await?;
        let mut body = json!({"prompt":prompt});
        if let Some((provider, model)) = model.as_deref().and_then(|model| model.split_once('/')) {
            body["model"] = json!({"providerID":provider,"id":model});
        }
        let result = tokio::time::timeout(timeout, async {
            let response: Value = endpoint
                .client
                .post(format!(
                    "{}{}",
                    endpoint.base_url,
                    endpoint.v2_generate_path()
                ))
                .json(&body)
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            response
                .pointer("/data/text")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("OpenCode generation returned no text"))
        })
        .await
        .map_err(|_| anyhow!("OpenCode generation timed out"))
        .and_then(|result| result);
        let _ = child.kill().await;
        let _ = child.wait().await;
        result
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Merges a captured beta-19271 turn (shell, then a subagent running shell) twice, against
    /// its real child export plus an earlier turn's step from a continued subagent session.
    #[tokio::test]
    async fn native_v2_turn_captures_each_step_once_and_only_this_turns_child_steps() {
        use axum::{routing::get, Json, Router};
        let fixture: Value =
            serde_json::from_str(include_str!("fixtures/opencode-v2-export.json")).unwrap();
        let main = messages(&fixture["main"]).unwrap().clone();
        let started_at = main[0]["time"]["created"].as_i64().unwrap();
        let mut child = fixture["child"].clone();
        let mut earlier = child["data"]["messages"][1].clone();
        earlier["id"] = json!("msg_earlier_turn");
        earlier["time"]["created"] = json!(started_at - 1);
        child["data"]["messages"]
            .as_array_mut()
            .unwrap()
            .insert(0, earlier);
        let app = Router::new().route(
            "/api/session/{id}/export",
            get(move || {
                let child = child.clone();
                async move { Json(child) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = AgentEndpoint {
            base_url: format!("http://{}", listener.local_addr().unwrap()),
            client: reqwest::Client::new(),
            protocol: crate::local::opencode::Protocol::V2,
            legacy_v2_api: true,
        };
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut ctx = TurnCtx::test_stub();
        let mut captured = Captured::default();
        for _ in 0..2 {
            let answered = merge_projection(
                &mut ctx,
                &endpoint,
                "ses_f0c2f7575ffe15gETnxld4d1P5",
                &main,
                &HashSet::new(),
                started_at,
                &mut captured,
            )
            .await
            .unwrap();
            assert!(answered);
        }
        server.abort();
        let mut samples: Vec<_> = captured.samples.keys().map(String::as_str).collect();
        samples.sort();
        assert_eq!(
            samples,
            [
                "msg_0f3d08b130017eFyyxhSAmLVQe",
                "msg_0f3d094ef001q6aLjtCXxt1tJO",
                "msg_0f3d09ccf001sU8UDnw1G33CT2",
                "msg_0f3d0a2ba001sxgWDHNk1sFrk4",
            ]
        );
        assert_eq!(
            captured.invokers,
            HashSet::from([
                "msg_0f3d08b130017eFyyxhSAmLVQe:1".to_string(),
                "msg_0f3d08b130017eFyyxhSAmLVQe:2".to_string(),
                "msg_0f3d094ef001q6aLjtCXxt1tJO:1".to_string(),
            ])
        );
        let subagent = ctx
            .assistant
            .parts
            .iter()
            .find(|part| part.id == "msg_0f3d08b130017eFyyxhSAmLVQe:2")
            .unwrap();
        assert!(subagent
            .children
            .iter()
            .any(|part| part.id == "msg_0f3d094ef001q6aLjtCXxt1tJO:1"));
        let (usage, complete) = executed_usage(&main[1]).unwrap();
        assert!(complete);
        assert_eq!(usage.input_tokens, Some(6338 + 145));
        assert_eq!(usage.output_tokens, Some(97));
    }

    #[test]
    fn only_a_streamed_or_measured_step_executed() {
        let failed = json!({"id":"msg_x","type":"assistant","agent":"build","model":{"id":"big-pickle","providerID":"opencode"},"content":[],"time":{"created":1,"completed":2},"finish":"error","error":{"type":"provider.auth","message":"Unauthorized"}});
        assert!(executed_usage(&failed).is_none());
        let mut interrupted = failed.clone();
        interrupted["time"]["streamed"] = json!(1);
        let (usage, complete) = executed_usage(&interrupted).unwrap();
        assert_eq!(usage, crate::store::TokenUsage::default());
        assert!(!complete);
        let mut partial = failed.clone();
        partial["content"] = json!([{"type":"text","text":"Half"}]);
        assert!(executed_usage(&partial).is_some());
    }

    /// A fake V2 server over fixed session exports. `active` answers `/api/session/active` in turn
    /// (the last repeats); interrupts are logged.
    async fn fake_v2(
        exports: HashMap<String, Value>,
        latest: Value,
        active: Vec<Value>,
    ) -> (
        AgentEndpoint,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        tokio::task::JoinHandle<()>,
    ) {
        use axum::{extract::Path, routing::get, routing::post, Json, Router};
        let interrupts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = interrupts.clone();
        let polls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let app = Router::new()
            .route(
                "/api/session/{id}/export",
                get(move |Path(id): Path<String>| {
                    let export = exports[&id].clone();
                    async move { Json(export) }
                }),
            )
            .route(
                "/api/session/{id}/message",
                get(move || {
                    let latest = latest.clone();
                    async move { Json(latest) }
                }),
            )
            .route(
                "/api/session/active",
                get(move || {
                    let poll = polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let active = active[poll.min(active.len() - 1)].clone();
                    async move { Json(active) }
                }),
            )
            .route(
                "/api/session/{id}/interrupt",
                post(move |Path(id): Path<String>| {
                    log.lock().unwrap().push(id);
                    async { Json(json!({"interrupted":true})) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = AgentEndpoint {
            base_url: format!("http://{}", listener.local_addr().unwrap()),
            client: reqwest::Client::new(),
            protocol: crate::local::opencode::Protocol::V2,
            legacy_v2_api: true,
        };
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (endpoint, interrupts, server)
    }

    fn step(id: &str, session: &str, model: &str, created: i64, content: Value) -> Value {
        json!({"id":id,"sessionID":session,"type":"assistant","model":{"id":model,"providerID":"p"},"content":content,"tokens":{"input":10,"output":2,"reasoning":0,"cache":{"read":0,"write":0}},"time":{"created":created,"streamed":created,"completed":created}})
    }

    fn subagent(session: &str, status: &str) -> Value {
        json!({"type":"tool","name":"subagent","callID":format!("call_{session}"),"state":{"status":"completed","input":{},"content":[],"metadata":{"sessionID":session,"status":status}}})
    }

    fn export(session: &str, parent: Option<&str>, messages: Vec<Value>) -> Value {
        json!({"data":{"info":{"id":session,"parentID":parent},"messages":messages}})
    }

    /// An interrupt stops this turn's background subagent (V2 spares it natively) and the parent it
    /// wakes, then captures every depth, the retrying step's model, and nothing from earlier turns.
    #[tokio::test]
    async fn interrupted_v2_turn_captures_its_whole_tree() {
        let shell = json!([{"type":"tool","name":"shell","callID":"call_sh","state":{"status":"completed","input":{"command":"true"},"content":[]}}]);
        let exports = HashMap::from([
            (
                "ses_main".to_string(),
                export(
                    "ses_main",
                    None,
                    vec![
                        step("msg_old", "ses_main", "main-model", 5, json!([])),
                        step(
                            "msg_main",
                            "ses_main",
                            "main-model",
                            10,
                            json!([
                                subagent("ses_child", "completed"),
                                subagent("ses_bg", "running")
                            ]),
                        ),
                    ],
                ),
            ),
            (
                "ses_child".to_string(),
                export(
                    "ses_child",
                    Some("ses_main"),
                    vec![
                        step("msg_child_old", "ses_child", "child-model", 5, json!([])),
                        step(
                            "msg_child",
                            "ses_child",
                            "child-model",
                            11,
                            json!([subagent("ses_grand", "completed")]),
                        ),
                    ],
                ),
            ),
            (
                "ses_grand".to_string(),
                export(
                    "ses_grand",
                    Some("ses_child"),
                    vec![step("msg_grand", "ses_grand", "grand-model", 12, shell)],
                ),
            ),
            (
                "ses_bg".to_string(),
                export(
                    "ses_bg",
                    Some("ses_main"),
                    vec![step("msg_bg", "ses_bg", "bg-model", 13, json!([]))],
                ),
            ),
        ]);
        let retrying = json!({"data":[{"id":"msg_retrying","type":"assistant","model":{"id":"main-model","providerID":"p"},"content":[],"retry":{"attempt":1,"at":14,"error":{"type":"provider.overloaded","message":"busy"}},"time":{"created":14}}],"cursor":{}});
        let (endpoint, interrupts, server) =
            fake_v2(exports, retrying, vec![json!({"data":{}})]).await;
        let recorded = super::super::tests::Recorded::default();
        capture_interrupted(&endpoint, &recorded, "ses_main", 10).await;
        server.abort();

        assert_eq!(*interrupts.lock().unwrap(), ["ses_bg", "ses_main"]);
        let samples = recorded.samples.into_inner().unwrap();
        let mut ids: Vec<_> = samples.keys().map(String::as_str).collect();
        ids.sort();
        assert_eq!(
            ids,
            [
                "msg_bg",
                "msg_child",
                "msg_grand",
                "msg_main",
                "msg_retrying:attempt:1"
            ]
        );
        let model = |id: &str| match &samples[id].0 {
            crate::store::Attribution::Exact { model, .. } => model.clone(),
            other => panic!("{id}: {other:?}"),
        };
        assert_eq!(model("msg_grand"), "grand-model");
        assert_eq!(model("msg_retrying:attempt:1"), "main-model");
        assert_eq!(
            samples["msg_retrying:attempt:1"].1,
            crate::store::TokenUsage::default()
        );
        assert!(!samples["msg_retrying:attempt:1"].2);
        assert!(samples["msg_grand"].2);
        let invokers = recorded.invokers.into_inner().unwrap();
        assert_eq!(invokers["msg_grand:0"], "grand-model");
        assert_eq!(invokers["msg_child:0"], "child-model");
    }

    /// Real beta-19271 runs: a subagent that spawned its own subagent (`experimental.subagent_depth`
    /// 2), and a background subagent whose result later woke the parent.
    #[tokio::test]
    async fn real_v2_grandchild_and_background_subagents_are_captured() {
        let fixture: Value =
            serde_json::from_str(include_str!("fixtures/opencode-v2-nested-export.json")).unwrap();
        let replay = |run: &str| {
            let run = &fixture[run];
            let main = run["main"]["data"]["info"]["id"].as_str().unwrap();
            let mut exports = HashMap::from([(main.to_string(), run["main"].clone())]);
            for (id, export) in run["sessions"].as_object().unwrap() {
                exports.insert(id.clone(), export.clone());
            }
            let started_at = run["main"]["data"]["messages"][0]["time"]["created"]
                .as_i64()
                .unwrap();
            (main, exports, started_at)
        };

        let (main, exports, started_at) = replay("nested");
        let (endpoint, _, server) = fake_v2(
            exports,
            json!({"data":[],"cursor":{}}),
            vec![json!({"data":{}})],
        )
        .await;
        let recorded = super::super::tests::Recorded::default();
        let mut captured = Captured::default();
        capture_tree(&recorded, &endpoint, main, started_at, &mut captured, false)
            .await
            .unwrap();
        server.abort();
        assert_eq!(
            captured.descendants["ses_f0c08d898ffeAtB9QwOBElycQc"],
            "ses_f0c08e2cfffeVYNY9f4SvHj7xR",
            "the grandchild's parent is the child"
        );
        let samples = recorded.samples.into_inner().unwrap();
        assert_eq!(samples.len(), 6, "two steps per session, every depth");
        assert!(samples.values().all(|(attribution, _, complete)| *complete
            && matches!(attribution, crate::store::Attribution::Exact { model, .. } if model == "big-pickle")));
        let invokers = recorded.invokers.into_inner().unwrap();
        assert!(
            invokers.contains_key("msg_0f3f727ac001GiaRSITeAuyytF:1"),
            "grandchild shell"
        );
        assert!(
            invokers.contains_key("msg_0f3f71d75001f70EWLiFqQefgU:1"),
            "child spawn"
        );

        let (main, exports, started_at) = replay("background");
        let (endpoint, _, server) = fake_v2(
            exports,
            json!({"data":[],"cursor":{}}),
            vec![json!({"data":{}})],
        )
        .await;
        let recorded = super::super::tests::Recorded::default();
        let mut captured = Captured::default();
        capture_tree(&recorded, &endpoint, main, started_at, &mut captured, false)
            .await
            .unwrap();
        server.abort();
        assert_eq!(
            captured.background,
            HashSet::from(["ses_f0c0c01a4ffebVNIrsOFXsVCSI".to_string()])
        );
        let samples = recorded.samples.into_inner().unwrap();
        assert!(
            samples["msg_0f3f46634001V8sKmW4maT6c8v"].2,
            "background step"
        );
        // The parent run the result natively woke (no app turn) belongs to this execution.
        const WOKEN: &str = "msg_0f3f46be7001vGS4IGSz10v12P";
        assert!(samples[WOKEN].2);

        // A later app turn (started after the spawn, before the wake) never accounts that run.
        let (endpoint, _, server) = fake_v2(
            replay("background").1,
            json!({"data":[],"cursor":{}}),
            vec![json!({"data":{}})],
        )
        .await;
        let later = super::super::tests::Recorded::default();
        capture_tree(
            &later,
            &endpoint,
            main,
            1790799270000,
            &mut Captured::default(),
            false,
        )
        .await
        .unwrap();
        assert!(!later.samples.into_inner().unwrap().contains_key(WOKEN));

        // With no later turn at all, the held execution's watcher accounts only the woken run
        // (plus the subagent), never the turn's own steps again.
        let watched = super::super::tests::Recorded::default();
        watch_background(
            &watched,
            &endpoint,
            main,
            &[(
                "ses_f0c0c01a4ffebVNIrsOFXsVCSI".to_string(),
                main.to_string(),
            )],
            started_at,
        )
        .await;
        server.abort();
        let watched = watched.samples.into_inner().unwrap();
        assert!(watched[WOKEN].2);
        assert!(!watched.contains_key("msg_0f3f3ee270019YUmIFWT2pdkiB"));
    }

    /// A background subagent keeps being captured after its turn until its tree goes quiet.
    #[tokio::test]
    async fn background_watch_captures_until_the_subagent_tree_is_quiet() {
        let exports = HashMap::from([(
            "ses_bg".to_string(),
            export(
                "ses_bg",
                Some("ses_main"),
                vec![step("msg_bg", "ses_bg", "bg-model", 13, json!([]))],
            ),
        )]);
        let (endpoint, _, server) = fake_v2(
            exports,
            json!({"data":[],"cursor":{}}),
            vec![
                json!({"data":{"ses_bg":{}}}),
                json!({"data":{"ses_bg":{}}}),
                json!({"data":{}}),
            ],
        )
        .await;
        let recorded = super::super::tests::Recorded::default();
        watch_background(
            &recorded,
            &endpoint,
            "ses_main",
            &[("ses_bg".to_string(), "ses_main".to_string())],
            10,
        )
        .await;
        server.abort();
        let samples = recorded.samples.into_inner().unwrap();
        assert!(samples["msg_bg"].2);
    }

    #[tokio::test]
    async fn discovery_waits_for_a_cold_v2_catalog() {
        use axum::{routing::get, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let requests = Arc::new(AtomicUsize::new(0));
        let count = requests.clone();
        let app = Router::new()
            .route(
                "/api/model",
                get(move || {
                    let count = count.clone();
                    async move {
                        Json(if count.fetch_add(1, Ordering::SeqCst) == 0 {
                            json!({"data":[]})
                        } else {
                            json!({"data":[{"id":"fixture"}]})
                        })
                    }
                }),
            )
            .route(
                "/api/integration",
                get(|| async { Json(json!({"data":[]})) }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = AgentEndpoint {
            base_url: format!("http://{}", listener.local_addr().unwrap()),
            client: reqwest::Client::new(),
            protocol: crate::local::opencode::Protocol::V2,
            legacy_v2_api: false,
        };
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (models, _) = discover_models(&endpoint).await.unwrap();
        assert_eq!(models["data"][0]["id"], "fixture");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        server.abort();
    }

    #[test]
    fn free_text_and_numeric_notes_survive_form_replies_and_reload() {
        let note = "a custom answer".to_string();
        let submitted = submitted_answers(&[], Some(&note));
        assert_eq!(
            field_answer(&json!({"type":"string"}), submitted).unwrap(),
            json!(note)
        );
        assert!(submitted_answers(&[], Some(&"  ".to_string())).is_empty());
        let chosen = vec!["Selected option".to_string()];
        assert_eq!(submitted_answers(&chosen, Some(&note)), chosen);

        let fields = vec![
            json!({"key":"count","type":"integer"}),
            json!({"key":"text","type":"string"}),
        ];
        let make_part = |key: &str, resolved, note: Option<&str>| {
            WirePart::prompt(
                key,
                WirePrompt {
                    kind: "question".into(),
                    native_id: Some(serde_json::to_string(&("frm_test", key)).unwrap()),
                    resolved,
                    note: note.map(str::to_owned),
                    ..Default::default()
                },
            )
        };
        let parts = vec![
            make_part("count", true, Some("42")),
            make_part("text", false, Some("not submitted")),
            make_part("text", true, None),
        ];
        let mut saved = serde_json::Map::new();
        saved_form_answers("frm_test", &fields, &parts, &mut saved).unwrap();
        assert_eq!(saved.get("count"), Some(&json!(42)));
        assert!(!saved.contains_key("text"));
    }

    #[test]
    fn filtered_content_keeps_its_subagent_metadata() {
        let message = json!({"id":"msg_test","content":[{"type":"future-part"},{"type":"text","text":"working"},{"type":"tool","name":"subagent","state":{"status":"running","metadata":{"sessionID":"ses_child"}}}]});
        let parts = projected_parts(&message);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1].0.id, "msg_test:2");
        assert_eq!(
            parts[1]
                .1
                .pointer("/state/metadata/sessionID")
                .and_then(Value::as_str),
            Some("ses_child")
        );
    }

    #[test]
    fn one_idle_snapshot_cannot_finish_a_just_admitted_prompt() {
        let mut idle = false;
        assert!(!observed_idle(&mut idle, false, false));
        assert!(!observed_idle(&mut idle, false, true));
        assert!(!observed_idle(&mut idle, true, false));
        assert!(!observed_idle(&mut idle, false, false));
        assert!(observed_idle(&mut idle, false, false));
    }

    #[test]
    fn conditional_forms_wait_for_dependencies_and_preserve_types() {
        let field = json!({"key":"count","type":"integer","when":[{"key":"continue","op":"eq","value":true}]});
        let mut answers = serde_json::Map::new();
        assert!(!field_active(&field, &answers));
        answers.insert("continue".into(), json!(true));
        assert!(field_active(&field, &answers));
        assert_eq!(field_answer(&field, &["3".into()]).unwrap(), json!(3));
        assert!(field_answer(&field, &["3.5".into()]).is_err());
        assert_eq!(
            field_answer(&json!({"type":"boolean"}), &["Yes".into()]).unwrap(),
            json!(true)
        );
        assert!(field_answer(&json!({"type":"number"}), &["NaN".into()]).is_err());
    }
    #[test]
    fn projected_parts_and_form_values_preserve_native_shapes() {
        let parts = wire_parts(
            &json!({"id":"msg_one","content":[{"type":"text","text":"hello"},{"type":"tool","id":"call","name":"read","state":{"status":"completed","input":{},"content":[{"type":"text","text":"file"}]}}]}),
        );
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].text.as_deref(), Some("hello"));
        assert_eq!(
            parts[1].state.as_ref().unwrap().output.as_deref(),
            Some("file")
        );
        let form = json!({"fields":[{"key":"answer","type":"string","options":[{"label":"Continue","value":"yes"}]}]});
        assert_eq!(
            field_answer(&form["fields"][0], &["Continue".into()]).unwrap(),
            json!("yes")
        );
    }
}

#[cfg(test)]
mod catalog_tests {
    use super::*;

    #[test]
    fn anonymous_zen_requires_explicitly_free_enabled_models() {
        let mut model =
            json!({"providerID":"opencode", "enabled":true, "cost":[{"input":0,"output":0}]});
        let anonymous = HashSet::new();
        assert!(selectable_model(&model, &anonymous));
        model["cost"][0]["output"] = json!(1);
        assert!(!selectable_model(&model, &anonymous));
        assert!(selectable_model(&model, &HashSet::from(["opencode"])));
        model["cost"] = json!([]);
        assert!(!selectable_model(&model, &anonymous));
        model["providerID"] = json!("custom-local-provider");
        assert!(selectable_model(&model, &anonymous));
        model["enabled"] = json!(false);
        assert!(!selectable_model(&model, &anonymous));
    }
}
