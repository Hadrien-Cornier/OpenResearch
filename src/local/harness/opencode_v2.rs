//! V2 admits prompts before execution finishes; reconcile its durable projection.
use super::*;
use crate::local::opencode::AgentEndpoint;
use futures::future::{BoxFuture, FutureExt};

async fn get(endpoint: &AgentEndpoint, path: &str) -> Result<Value> {
    fetch(&endpoint.client, &endpoint.base_url, path).await
}

async fn fetch(client: &reqwest::Client, base: &str, path: &str) -> Result<Value> {
    Ok(client
        .get(format!("{base}{path}"))
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

/// A session's export shape from its message list: native export drops unsettled steps (no
/// `time.completed`), which an OpenCode restart leaves unsettled for good.
async fn transcript(history: History<'_>, session: &str) -> Result<Value> {
    let (client, base) = match history {
        History::Server(client, base) => (client, base),
        History::Database(path) => {
            return stored(path, session, native_store::opencode_database::v2_history).await
        }
    };
    // Info first, so the messages read after it are at least as new as its outcome.
    let info = fetch(client, base, &format!("/api/session/{session}")).await?;
    let mut messages = Vec::new();
    let mut page = "order=asc".to_owned();
    loop {
        let list = fetch(
            client,
            base,
            &format!("/api/session/{session}/message?limit=200&{page}"),
        )
        .await?;
        let items = list["data"]
            .as_array()
            .ok_or_else(|| anyhow!("OpenCode V2 message list is invalid"))?;
        messages.extend(items.iter().cloned());
        match list.pointer("/cursor/next").and_then(Value::as_str) {
            Some(cursor) if items.len() == 200 => page = format!("cursor={cursor}"),
            _ => break,
        }
    }
    Ok(json!({"data":{"info":info["data"],"messages":messages}}))
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
    let before = transcript((&endpoint).into(), &native_id).await?;
    let previous: HashSet<String> = messages(&before)?
        .iter()
        .filter_map(|m| m["id"].as_str().map(str::to_owned))
        .collect();
    let prompt_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let started_at = crate::store::now_ms();
    let mut captured = Captured::default();
    persist_scope(
        ctx,
        json!({"v2": true, "native": native_id, "startedAt": started_at, "roots": [],
            "prompt": prompt_id}),
    );
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
            // ponytail: full history (200-message pages) each poll; switch to the durable event log if long chats make this costly.
            let projection = transcript((&endpoint).into(), &native_id).await?;
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
                let final_projection = transcript((&endpoint).into(), &native_id).await?;
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
    // A failed turn may have missed root steps: keep watching the prompt's run.
    let root = result.is_err();
    hold_and_watch(
        ctx,
        (root || !roots.is_empty()).then(|| {
            json!({"v2": true, "native": native_id, "startedAt": started_at, "roots": roots,
                "prompt": root.then_some(&prompt_id)})
        }),
        &endpoint.base_url,
    );
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
                part.children = subagent_children(
                    &*ctx,
                    endpoint.into(),
                    native_id,
                    child,
                    started_at,
                    captured,
                )
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
pub(super) struct Captured {
    samples: HashMap<String, String>,
    invokers: HashSet<String>,
    /// Tool parts kept as run evidence, by their last recorded state.
    evidence: HashMap<String, String>,
    /// This turn's subagent sessions at any depth, each with its parent session.
    descendants: HashMap<String, String>,
    /// Background subagents spawned this turn, which can outlive it.
    background: HashSet<String>,
}

impl Captured {
    /// A held execution's background subagents (child → parent session), as its watcher starts.
    pub(super) fn watching(roots: &[(String, String)]) -> Self {
        Self {
            background: roots.iter().map(|(child, _)| child.clone()).collect(),
            descendants: roots.iter().cloned().collect(),
            ..Default::default()
        }
    }

    pub(super) fn sessions(&self, native_id: &str) -> Vec<String> {
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

/// The session a `subagent` tool part of `parent` runs, noting a background one.
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
    // A foreground subagent's part also reports `running` until the part itself settles.
    if content.pointer("/state/status") != Some(&json!("running"))
        && content.pointer("/state/metadata/status") == Some(&json!("running"))
    {
        captured.background.insert(child.to_owned());
    }
    Some(child)
}

/// A subagent's transcript with its own subagents nested, capturing this turn's steps at every
/// depth. A continued subagent session also holds earlier turns' messages.
fn subagent_children<'a>(
    sink: &'a dyn UsageSink,
    history: History<'a>,
    parent_id: &'a str,
    child_id: &'a str,
    started_at: i64,
    captured: &'a mut Captured,
) -> BoxFuture<'a, Result<Vec<WirePart>>> {
    async move {
        let child = transcript(history, child_id).await?;
        if child.pointer("/data/info/parentID").and_then(Value::as_str) != Some(parent_id) {
            return Err(anyhow!("OpenCode returned an unrelated subagent session"));
        }
        let mut children = Vec::new();
        let all = messages(&child)?;
        let woken = woken_runs(all);
        for message in all {
            // A run another execution's subagent woke is that execution's, as in the chat session.
            let current = message
                .pointer("/time/created")
                .and_then(Value::as_i64)
                .is_some_and(|created| created >= started_at)
                && !foreign_run(message, &woken, captured);
            if current {
                capture(sink, message, true, captured);
            }
            if message["type"] != "assistant" {
                continue;
            }
            for (mut part, content) in projected_parts(message) {
                // An earlier turn's spawn is not this turn's descendant, even if still running.
                if let Some(grandchild) = current
                    .then(|| subagent_session(content, child_id, captured))
                    .flatten()
                {
                    part.children = subagent_children(
                        sink, history, child_id, grandchild, started_at, captured,
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
    for (part, _) in projected_parts(message) {
        if part.kind != "tool" {
            continue;
        }
        // After the turn no transcript holds a subagent's or woken run's tool parts.
        let state = serde_json::to_string(&part).unwrap_or_default();
        if captured.evidence.insert(part.id.clone(), state.clone()) != Some(state) {
            sink.tool_evidence(&part);
        }
        if let crate::store::Attribution::Exact { model, provider } = &attribution {
            if captured.invokers.insert(part.id.clone()) {
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
) -> Result<()> {
    let mut captured = Captured::default();
    wait_idle(endpoint, &[native_id.to_string()]).await;
    let mut read = capture_tree(
        sink,
        endpoint.into(),
        native_id,
        started_at,
        &mut captured,
        false,
        None,
    )
    .await
    .map(drop);
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
        // A full re-read from the turn's start supersedes the first read.
        read = capture_tree(
            sink,
            endpoint.into(),
            native_id,
            started_at,
            &mut captured,
            false,
            None,
        )
        .await
        .map(drop);
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
    read
}

/// One watcher poll of a held execution's background subagents and the runs their results natively
/// woke (no app turn) in each session they report to: the tree's still-active sessions, and
/// whether a subagent's result is still undelivered.
pub(super) async fn poll_background(
    sink: &dyn UsageSink,
    endpoint: &AgentEndpoint,
    (native_id, prompt): (&str, Option<&str>),
    roots: &[(String, String)],
    started_at: i64,
    captured: &mut Captured,
) -> Result<(Vec<String>, bool)> {
    // Read before capturing, so a session idle here has settled everything captured below.
    let active = get(endpoint, "/api/session/active").await?;
    for (child, parent) in roots {
        subagent_children(sink, endpoint.into(), parent, child, started_at, captured).await?;
    }
    let parents: HashSet<String> = std::iter::once(native_id.to_string())
        .chain(
            captured
                .background
                .iter()
                .filter_map(|child| captured.descendants.get(child).cloned()),
        )
        .collect();
    let mut delivered = HashSet::new();
    for parent in &parents {
        let prompt = prompt.filter(|_| parent == native_id);
        delivered.extend(
            capture_tree(
                sink,
                endpoint.into(),
                parent,
                started_at,
                captured,
                true,
                prompt,
            )
            .await?,
        );
    }
    let busy = parents
        .iter()
        .chain(captured.descendants.keys())
        .filter(|session| active["data"].get(session.as_str()).is_some())
        .cloned()
        .collect();
    let undelivered = captured
        .background
        .iter()
        .any(|child| !delivered.contains(child));
    Ok((busy, undelivered))
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

/// Message ids of `prompt`'s own run: everything after it until the next prompt, or the native
/// idle marker that closes the turn.
fn prompt_run<'a>(messages: &'a [Value], prompt: &str) -> HashSet<&'a str> {
    messages
        .iter()
        .skip_while(|message| message["id"] != prompt)
        .skip(1)
        .take_while(|message| !matches!(message["type"].as_str(), Some("user" | "idle")))
        .filter_map(|message| message["id"].as_str())
        .collect()
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
/// messages belong to them), plus `prompt`'s own run if given. Returns the subagents whose results
/// `native_id` received.
pub(super) async fn capture_tree(
    sink: &dyn UsageSink,
    history: History<'_>,
    native_id: &str,
    started_at: i64,
    captured: &mut Captured,
    woken_only: bool,
    prompt: Option<&str>,
) -> Result<HashSet<String>> {
    let projection = transcript(history, native_id).await?;
    let all = messages(&projection)?;
    let woken = woken_runs(all);
    let run = prompt
        .map(|prompt| prompt_run(all, prompt))
        .unwrap_or_default();
    for message in all.iter().filter(|m| {
        m.pointer("/time/created")
            .and_then(Value::as_i64)
            .is_some_and(|created| created >= started_at)
    }) {
        let id = message["id"].as_str().unwrap_or_default();
        let owned = woken.contains_key(id) || run.contains(id);
        if foreign_run(message, &woken, captured) || (woken_only && !owned) {
            continue;
        }
        capture(sink, message, false, captured);
        for content in message["content"].as_array().into_iter().flatten() {
            if let Some(child) = subagent_session(content, native_id, captured) {
                subagent_children(sink, history, native_id, child, started_at, captured).await?;
            }
        }
    }
    Ok(woken_deliveries(all).collect())
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
        let app = transcripts(move |_| child.clone(), json!({"data":[]}));
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

    /// The live merge captures a terminally failed step before failing the turn: one that streamed
    /// keeps its observed model without counters; one that failed before streaming records nothing.
    #[tokio::test]
    async fn failed_v2_steps_keep_the_model_only_when_executed() {
        let endpoint = AgentEndpoint {
            base_url: "http://127.0.0.1:9".into(),
            client: reqwest::Client::new(),
            protocol: crate::local::opencode::Protocol::V2,
            legacy_v2_api: true,
        };
        let failed = json!({"id":"msg_x","type":"assistant","agent":"build","model":{"id":"big-pickle","providerID":"opencode"},"content":[],"time":{"created":1,"completed":2},"finish":"error","error":{"type":"provider.auth","message":"Unauthorized"}});
        let mut after = failed.clone();
        after["time"]["streamed"] = json!(1);
        for (message, executed) in [(failed, false), (after, true)] {
            let mut ctx = TurnCtx::test_stub();
            let mut captured = Captured::default();
            let merged = merge_projection(
                &mut ctx,
                &endpoint,
                "ses_main",
                &[message],
                &HashSet::new(),
                0,
                &mut captured,
            )
            .await;
            assert!(merged.is_err());
            let exact = crate::store::Attribution::Exact {
                model: "big-pickle".into(),
                provider: Some("opencode".into()),
            };
            let expected = json!([exact, crate::store::TokenUsage::default(), false]).to_string();
            assert_eq!(captured.samples.get("msg_x"), executed.then_some(&expected));
        }
    }

    /// Serves `export(session)` as V2's session info and message list; a newest-first read (as
    /// [`observe_retry`] makes) gets `latest`.
    fn transcripts(
        export: impl Fn(&str) -> Value + Clone + Send + Sync + 'static,
        latest: Value,
    ) -> axum::Router {
        use axum::{
            extract::{Path, Query},
            routing::get,
            Json, Router,
        };
        let info = export.clone();
        Router::new()
            .route(
                "/api/session/{id}",
                get(move |Path(id): Path<String>| {
                    let data = info(&id)["data"]["info"].clone();
                    async move { Json(json!({"data": data})) }
                }),
            )
            .route(
                "/api/session/{id}/message",
                get(
                    move |Path(id): Path<String>, Query(query): Query<HashMap<String, String>>| {
                        let page = if query.get("order").map(String::as_str) == Some("desc") {
                            latest.clone()
                        } else {
                            json!({"data": export(&id)["data"]["messages"], "cursor": {}})
                        };
                        async move { Json(page) }
                    },
                ),
            )
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
        use axum::{extract::Path, routing::get, routing::post, Json};
        let interrupts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = interrupts.clone();
        let polls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let app = transcripts(move |id| exports[id].clone(), latest)
            .route(
                "/api/session/active",
                get(move || {
                    let poll = polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let active = active[poll.min(active.len() - 1)].clone();
                    // `null` stands for a failed status read.
                    async move {
                        (!active.is_null())
                            .then_some(Json(active))
                            .ok_or(axum::http::StatusCode::SERVICE_UNAVAILABLE)
                    }
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
    /// wakes, then captures every depth, the retrying step's model, and nothing from earlier turns,
    /// not even a background subagent an earlier turn left running in a continued session.
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
                        // Interrupted before its provider responded: never executed.
                        json!({"id":"msg_unstarted","sessionID":"ses_main","type":"assistant",
                            "model":{"id":"main-model","providerID":"p"},"content":[],
                            "time":{"created":15,"completed":16}}),
                    ],
                ),
            ),
            (
                "ses_child".to_string(),
                export(
                    "ses_child",
                    Some("ses_main"),
                    vec![
                        step(
                            "msg_child_old",
                            "ses_child",
                            "child-model",
                            5,
                            json!([subagent("ses_old_bg", "running")]),
                        ),
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
        capture_interrupted(&endpoint, &recorded, "ses_main", 10)
            .await
            .unwrap();
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
        capture_tree(
            &recorded,
            (&endpoint).into(),
            main,
            started_at,
            &mut captured,
            false,
            None,
        )
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
        capture_tree(
            &recorded,
            (&endpoint).into(),
            main,
            started_at,
            &mut captured,
            false,
            None,
        )
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
            (&endpoint).into(),
            main,
            1790799270000,
            &mut Captured::default(),
            false,
            None,
        )
        .await
        .unwrap();
        assert!(!later.samples.into_inner().unwrap().contains_key(WOKEN));

        // With no later turn at all, the held execution's watcher accounts only the woken run
        // (plus the subagent), never the turn's own steps again.
        let watched = super::super::tests::Recorded::default();
        super::super::settle_watch(
            &watched,
            &json!({"v2": true, "native": main, "startedAt": started_at,
                "roots": [["ses_f0c0c01a4ffebVNIrsOFXsVCSI", main]]}),
            super::super::tests::live(endpoint.clone()),
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
        let exports = HashMap::from([
            (
                "ses_bg".to_string(),
                export(
                    "ses_bg",
                    Some("ses_main"),
                    vec![step("msg_bg", "ses_bg", "bg-model", 13, json!([]))],
                ),
            ),
            ("ses_main".to_string(), export("ses_main", None, vec![])),
        ]);
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
        super::super::settle_watch(
            &recorded,
            &json!({"v2": true, "native": "ses_main", "startedAt": 10,
                "roots": [["ses_bg", "ses_main"]]}),
            super::super::tests::live(endpoint),
        )
        .await;
        server.abort();
        let samples = recorded.samples.into_inner().unwrap();
        assert!(samples["msg_bg"].2);
    }

    /// A status read failing while the subagent still runs must not look idle or end the watch:
    /// the same watcher retries and captures the step that finishes once the server answers again.
    #[tokio::test]
    async fn background_watch_retries_a_failed_status_read_until_the_tree_is_quiet() {
        use axum::{http::StatusCode, routing::get, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let polls = std::sync::Arc::new(AtomicUsize::new(0));
        let seen = polls.clone();
        let app = Router::new()
            .route(
                "/api/session/active",
                get(move || {
                    let poll = polls.fetch_add(1, Ordering::SeqCst);
                    async move {
                        match poll {
                            0 => Ok(Json(json!({"data":{"ses_bg":{}}}))),
                            1 => Err(StatusCode::SERVICE_UNAVAILABLE),
                            _ => Ok(Json(json!({"data":{}}))),
                        }
                    }
                }),
            )
            .merge(transcripts(
                move |id| {
                    // The step settles only after the status outage.
                    let mut bg = step("msg_bg", "ses_bg", "bg-model", 13, json!([]));
                    if seen.load(Ordering::SeqCst) < 3 {
                        bg["tokens"] = Value::Null;
                    }
                    let delivery = json!({"id":"msg_d","type":"synthetic","time":{"created":20},
                        "metadata":{"source":"subagent","childID":"ses_bg"}});
                    if id == "ses_bg" {
                        export("ses_bg", Some("ses_main"), vec![bg])
                    } else {
                        export("ses_main", None, vec![delivery])
                    }
                },
                json!({"data":[]}),
            ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = AgentEndpoint {
            base_url: format!("http://{}", listener.local_addr().unwrap()),
            client: reqwest::Client::new(),
            protocol: crate::local::opencode::Protocol::V2,
            legacy_v2_api: true,
        };
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let recorded = super::super::tests::Recorded::default();
        super::super::settle_watch(
            &recorded,
            &json!({"v2": true, "native": "ses_main", "startedAt": 10,
                "roots": [["ses_bg", "ses_main"]]}),
            super::super::tests::live(endpoint),
        )
        .await;
        server.abort();
        let samples = recorded.samples.into_inner().unwrap();
        assert_eq!(samples["msg_bg"].1.input_tokens, Some(10));
        assert!(samples["msg_bg"].2);
        assert!(!samples.contains_key("opencode-background:unrecoverable"));
    }

    /// Real beta-19271 restart: a background child's streamed step never settles, so it is read from
    /// the message list; replays record it once, a never-sent step nothing, failed reads retry.
    #[tokio::test]
    async fn restart_recovers_an_unsettled_background_step() {
        use axum::{routing::get, Json};
        use std::sync::atomic::{AtomicUsize, Ordering};
        const CHILD: &str = "ses_f0b57ccd6ffeJEVH5aWpJ2tsh8";
        const PARENT: &str = "ses_f0b60a09effeOCZeRrax8lxEZt";
        const STEP: &str = "msg_0f4a8337e001Nn559YxJ92dNZD";
        let unsettled = json!({"id":STEP,"sessionID":CHILD,"type":"assistant","agent":"general",
            "model":{"id":"big-pickle","providerID":"opencode","variant":"default"},
            "content":[{"type":"text","text":"I'll run the sleep once."},
                {"type":"tool","id":"call_function_ihoznrnawkj1_1","name":"shell","executed":false,
                "state":{"status":"running","input":{"command":"sleep 180","timeout":200000},"metadata":{}},
                "time":{"created":1790811060749i64,"ran":1790811060800i64}}],
            "time":{"created":1790811060567i64,"streamed":1790811060802i64}});
        let unsent = json!({"id":"msg_unsent","sessionID":CHILD,"type":"assistant",
            "model":{"id":"other-model","providerID":"opencode"},"content":[],
            "time":{"created":1790811060900i64}});
        let scope = json!({"v2": true, "native": PARENT, "startedAt": 1790811053399i64,
            "roots": [[CHILD, PARENT]]});
        let mut replays = Vec::new();
        for _ in 0..2 {
            let reads = std::sync::Arc::new(AtomicUsize::new(0));
            let (unsettled, unsent) = (unsettled.clone(), unsent.clone());
            let app = transcripts(
                move |id| {
                    if id != CHILD {
                        return export(PARENT, None, vec![]);
                    }
                    // Info, then the first message-list read, which fails.
                    if reads.fetch_add(1, Ordering::SeqCst) == 1 {
                        return Value::Null;
                    }
                    export(CHILD, Some(PARENT), vec![unsettled.clone(), unsent.clone()])
                },
                json!({"data":[]}),
            )
            .route(
                "/api/session/active",
                get(|| async { Json(json!({"data":{}})) }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = AgentEndpoint {
                base_url: format!("http://{}", listener.local_addr().unwrap()),
                client: reqwest::Client::new(),
                protocol: crate::local::opencode::Protocol::V2,
                legacy_v2_api: false,
            };
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let recorded = super::super::tests::Recorded::default();
            super::super::settle_watch(&recorded, &scope, super::super::tests::live(endpoint))
                .await;
            server.abort();
            assert_eq!(
                recorded.invokers.into_inner().unwrap()[&format!("{STEP}:1")],
                "big-pickle"
            );
            assert!(recorded
                .evidence
                .into_inner()
                .unwrap()
                .iter()
                .any(|part| part.id == format!("{STEP}:1")));
            replays.push(json!(recorded.samples.into_inner().unwrap()));
        }
        let exact = crate::store::Attribution::Exact {
            model: "big-pickle".into(),
            provider: Some("opencode".into()),
        };
        let expected = json!({STEP: [exact, crate::store::TokenUsage::default(), false]});
        assert_eq!(replays, [expected.clone(), expected]);
    }

    /// A native V2 database at `path` holding `sessions` (id, parent, claimed, messages).
    fn v2_database(path: &std::path::Path, sessions: &[(&str, Option<&str>, bool, Vec<Value>)]) {
        let db = rusqlite::Connection::open(path).unwrap();
        db.execute_batch("CREATE TABLE session_v2 (id TEXT PRIMARY KEY, parent_id TEXT, time_suspended INTEGER); CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT, type TEXT, seq INTEGER, data TEXT);").unwrap();
        for (id, parent, claimed, messages) in sessions {
            let claimed = claimed.then_some(1);
            db.execute(
                "INSERT INTO session_v2 VALUES (?1, ?2, ?3)",
                rusqlite::params![id, parent, claimed],
            )
            .unwrap();
            for (seq, message) in messages.iter().enumerate() {
                let mut data = message.clone();
                let data_map = data.as_object_mut().unwrap();
                let (message_id, kind) = (
                    data_map.remove("id").unwrap(),
                    data_map.remove("type").unwrap(),
                );
                db.execute(
                    "INSERT INTO session_message VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![
                        message_id.as_str(),
                        id,
                        kind.as_str(),
                        seq as i64,
                        data.to_string()
                    ],
                )
                .unwrap();
            }
        }
    }

    fn user(id: &str, created: i64) -> Value {
        json!({"id":id,"type":"user","text":"t","time":{"created":created}})
    }

    /// V2 crash recovery reads the native database, starting no server: the prompt's unsettled run
    /// and its child only; an unreadable database retries and replays add no counters.
    #[tokio::test]
    async fn restart_recovers_the_prompt_run_from_the_native_database() {
        let mut root = step(
            "msg_root",
            "ses_main",
            "root-model",
            13,
            json!([subagent("ses_child", "completed")]),
        );
        root["tokens"] = Value::Null;
        root["time"]["completed"] = Value::Null;
        let main = vec![
            user("msg_older", 5),
            step("msg_old", "ses_main", "old-model", 11, json!([])),
            user("msg_prompt", 12),
            root,
            json!({"id":"msg_d","type":"synthetic","time":{"created":14},
                "metadata":{"source":"subagent","childID":"ses_earlier_bg"}}),
            step("msg_foreign", "ses_main", "foreign-model", 15, json!([])),
            json!({"id":"msg_idle","type":"idle","outcome":"succeeded","time":{"created":16}}),
            step("msg_after_idle", "ses_main", "idle-model", 17, json!([])),
            user("msg_later", 20),
            step("msg_later_step", "ses_main", "later-model", 21, json!([])),
        ];
        let child = vec![step("msg_child", "ses_child", "child-model", 13, json!([]))];
        let measured = opencode_sample(Some(&child[0]["tokens"])).0;
        let dir = std::env::temp_dir().join(format!("orx-oc-v2db-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let (staged, database) = (dir.join("staged.db"), dir.join("opencode.db"));
        v2_database(
            &staged,
            &[
                ("ses_main", None, false, main),
                ("ses_child", Some("ses_main"), false, child),
            ],
        );
        // The database is unreadable until it appears.
        let appear = tokio::spawn({
            let database = database.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                std::fs::rename(staged, database).unwrap();
            }
        });
        let (store, db) = super::super::tests::usage_store();
        let sink = super::super::tests::StoreSink(std::sync::Mutex::new(store));
        let scope =
            json!({"v2":true,"native":"ses_main","startedAt":10,"roots":[],"prompt":"msg_prompt"});
        let owner = || async { super::super::Owner::Gone };
        super::super::settle_stored(&sink, &scope, &database, owner).await;
        appear.await.unwrap();
        let first = super::super::tests::ledger(&db);
        super::super::settle_stored(&sink, &scope, &database, owner).await;
        assert_eq!(super::super::tests::ledger(&db), first);
        let rows: Vec<(String, crate::store::Attribution, crate::store::TokenUsage)> = first
            .into_iter()
            .map(|(id, attribution, usage)| {
                let attribution = serde_json::from_str(&attribution).unwrap();
                (id, attribution, serde_json::from_str(&usage).unwrap())
            })
            .collect();
        let exact = |model: &str| crate::store::Attribution::Exact {
            model: model.into(),
            provider: Some("p".into()),
        };
        assert_eq!(
            rows,
            [
                ("msg_child".to_string(), exact("child-model"), measured),
                (
                    "msg_root".to_string(),
                    exact("root-model"),
                    crate::store::TokenUsage::default()
                ),
            ]
        );
    }

    /// Recovery never seals partial work as complete: an unfinished native turn, a database another
    /// process holds past the grace, or a lost session each leave that session's own marker.
    #[tokio::test]
    async fn stored_recovery_marks_partial_work_with_the_lost_sessions_reason() {
        use crate::store::{Attribution::Unresolved, Missing};
        let root = step(
            "msg_root",
            "ses_main",
            "root-model",
            13,
            json!([subagent("ses_child", "completed")]),
        );
        let child = (
            "ses_child",
            Some("ses_main"),
            false,
            vec![step("msg_child", "ses_child", "child-model", 13, json!([]))],
        );
        let main = |claimed| {
            (
                "ses_main",
                None,
                claimed,
                vec![user("msg_prompt", 12), root.clone()],
            )
        };
        let root_lost = ("opencode-root:unrecoverable", Missing::IdentityNotReported);
        let child_lost = (
            "opencode-background:unrecoverable",
            Missing::ChildModelUnknown,
        );
        // An unknown owner never holds open work native already finished.
        for (sessions, unknown, marker, recorded_samples) in [
            (vec![main(true), child.clone()], false, Some(root_lost), 3),
            (vec![main(false), child.clone()], true, None, 2),
            (vec![child.clone()], false, Some(root_lost), 1),
            (vec![main(false)], false, Some(child_lost), 2),
        ] {
            let path =
                std::env::temp_dir().join(format!("orx-oc-v2db-{}.db", uuid::Uuid::new_v4()));
            v2_database(&path, &sessions);
            let recorded = super::super::tests::Recorded::default();
            let scope = json!({"v2":true,"native":"ses_main","startedAt":10,"roots":[],
                "prompt":"msg_prompt"});
            let owner = || async move {
                match unknown {
                    true => super::super::Owner::Unknown,
                    false => super::super::Owner::Gone,
                }
            };
            super::super::settle_stored(&recorded, &scope, &path, owner).await;
            let samples = recorded.samples.into_inner().unwrap();
            if let Some((marker, reason)) = marker {
                assert_eq!(
                    samples[marker].0,
                    Unresolved { reason },
                    "{marker} {unknown}"
                );
            }
            assert_eq!(samples.len(), recorded_samples, "{samples:?}");
        }
    }

    /// A cancellation whose native history read failed reports the failure, so its caller keeps the
    /// persisted scope for a retry.
    #[tokio::test]
    async fn interrupted_capture_reports_a_failed_history_read() {
        use axum::{http::StatusCode, routing::get, Json, Router};
        let app = Router::new()
            .route(
                "/api/session/active",
                get(|| async { Json(json!({"data":{}})) }),
            )
            .route(
                "/api/session/{id}",
                get(|| async { StatusCode::SERVICE_UNAVAILABLE }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = AgentEndpoint {
            base_url: format!("http://{}", listener.local_addr().unwrap()),
            client: reqwest::Client::new(),
            protocol: crate::local::opencode::Protocol::V2,
            legacy_v2_api: false,
        };
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let recorded = super::super::tests::Recorded::default();
        let read = capture_interrupted(&endpoint, &recorded, "ses_main", 10).await;
        server.abort();
        assert!(read.is_err());
    }

    /// While the turn's own server still runs its tree, recovery waits for it rather than sealing;
    /// that server says when the work settled.
    #[tokio::test]
    async fn stored_recovery_waits_for_the_turns_own_server() {
        use axum::{routing::get, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let path = std::env::temp_dir().join(format!("orx-oc-v2db-{}.db", uuid::Uuid::new_v4()));
        let root = step("msg_root", "ses_main", "root-model", 13, json!([]));
        v2_database(
            &path,
            &[("ses_main", None, false, vec![user("msg_prompt", 12), root])],
        );
        let polls = std::sync::Arc::new(AtomicUsize::new(0));
        let seen = polls.clone();
        let app = Router::new().route(
            "/api/session/active",
            get(move || async move {
                let busy = seen.fetch_add(1, Ordering::SeqCst) < 2;
                Json(if busy {
                    json!({"data":{"ses_main":{}}})
                } else {
                    json!({"data":{}})
                })
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = AgentEndpoint {
            base_url: format!("http://{}", listener.local_addr().unwrap()),
            client: reqwest::Client::new(),
            protocol: crate::local::opencode::Protocol::V2,
            legacy_v2_api: false,
        };
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let recorded = super::super::tests::Recorded::default();
        let scope =
            json!({"v2":true,"native":"ses_main","startedAt":10,"roots":[],"prompt":"msg_prompt"});
        let owner = || {
            let endpoint = endpoint.clone();
            async move { super::super::Owner::Live(endpoint) }
        };
        super::super::settle_stored(&recorded, &scope, &path, owner).await;
        server.abort();
        assert_eq!(polls.load(Ordering::SeqCst), 3);
        let samples = recorded.samples.into_inner().unwrap();
        assert_eq!(samples.keys().collect::<Vec<_>>(), ["msg_root"]);
    }

    /// A 404 on a session's info read is confirmed natively and ends the watch once with that
    /// session's reason; a 404 on the active list names no session and is retried.
    #[tokio::test]
    async fn a_missing_session_ends_the_watch_with_its_own_reason() {
        use crate::store::{Attribution::Unresolved, Missing};
        use axum::{extract::Path, http::StatusCode, routing::get, Json, Router};
        use std::sync::atomic::{AtomicUsize, Ordering};
        for (gone, marker) in [
            (
                "ses_main",
                Some(("opencode-root:unrecoverable", Missing::IdentityNotReported)),
            ),
            (
                "ses_bg",
                Some((
                    "opencode-background:unrecoverable",
                    Missing::ChildModelUnknown,
                )),
            ),
            ("active", None),
        ] {
            let actives = std::sync::Arc::new(AtomicUsize::new(0));
            let app = Router::new()
                .route(
                    "/api/session/active",
                    get(move || async move {
                        if gone == "active" && actives.fetch_add(1, Ordering::SeqCst) == 0 {
                            Err(StatusCode::NOT_FOUND)
                        } else {
                            Ok(Json(json!({"data":{}})))
                        }
                    }),
                )
                .route(
                    "/api/session/{id}",
                    get(move |Path(id): Path<String>| async move {
                        let parent = (id == "ses_bg").then_some("ses_main");
                        if id == gone {
                            Err(StatusCode::NOT_FOUND)
                        } else {
                            Ok(Json(json!({"data":{"id":id,"parentID":parent}})))
                        }
                    }),
                )
                .route(
                    "/api/session/{id}/message",
                    get(|| async { Json(json!({"data":[],"cursor":{}})) }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = AgentEndpoint {
                base_url: format!("http://{}", listener.local_addr().unwrap()),
                client: reqwest::Client::new(),
                protocol: crate::local::opencode::Protocol::V2,
                legacy_v2_api: false,
            };
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let confirmed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let recorded = super::super::tests::Recorded::default();
            let scope = json!({"v2":true,"native":"ses_main","startedAt":10,
                "roots":[["ses_bg","ses_main"]],"prompt":"msg_prompt"});
            let endpoint = |missing: Option<String>| {
                let (endpoint, confirmed) = (endpoint.clone(), confirmed.clone());
                async move {
                    match missing {
                        Some(session) => {
                            confirmed.lock().unwrap().push(session.clone());
                            Err(anyhow!("OpenCode session {session} no longer exists"))
                        }
                        None => Ok(Some(endpoint)),
                    }
                }
            };
            tokio::time::timeout(
                Duration::from_secs(5),
                super::super::settle_watch(&recorded, &scope, endpoint),
            )
            .await
            .expect("a confirmed 404 ends the watch");
            server.abort();
            let samples = recorded.samples.into_inner().unwrap();
            match marker {
                Some((marker, reason)) => {
                    assert_eq!(*confirmed.lock().unwrap(), [gone]);
                    assert_eq!(samples[marker].0, Unresolved { reason });
                    assert_eq!(samples.len(), 1);
                }
                None => assert!(samples.is_empty() && confirmed.lock().unwrap().is_empty()),
            }
        }
    }

    /// Histories past one page follow native's `cursor.next` (never combined with `order`) until a
    /// short page, oldest first; a full last page ends on the empty page after it.
    #[tokio::test]
    async fn transcript_pages_through_long_histories() {
        use axum::{
            extract::{Path, Query},
            http::StatusCode,
            routing::get,
            Json, Router,
        };
        for total in [450, 400] {
            let all: Vec<Value> = (0..total)
                .map(|i| json!({"id":format!("msg_{i:03}"),"type":"user"}))
                .collect();
            let pages = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let served = pages.clone();
            let app = Router::new()
                .route(
                    "/api/session/{id}",
                    get(|Path(id): Path<String>| async move { Json(json!({"data":{"id":id}})) }),
                )
                .route(
                    "/api/session/{id}/message",
                    get(move |Query(query): Query<HashMap<String, String>>| {
                        served.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        // Native: the first page needs `order=asc` (default desc), a cursor excludes it.
                        let start = match (query.get("cursor"), query.get("order")) {
                            (Some(cursor), None) => cursor[2..].parse::<usize>().unwrap() + 1,
                            (None, Some(order)) if order == "asc" => 0,
                            _ => return std::future::ready(Err(StatusCode::BAD_REQUEST)),
                        };
                        let limit: usize = query["limit"].parse().unwrap();
                        assert!(limit <= 200);
                        let page = &all[start.min(all.len())..(start + limit).min(all.len())];
                        let next = page.last().map(|_| format!("c:{}", start + page.len() - 1));
                        std::future::ready(Ok(Json(json!({"data":page,"cursor":{"next":next}}))))
                    }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = AgentEndpoint {
                base_url: format!("http://{}", listener.local_addr().unwrap()),
                client: reqwest::Client::new(),
                protocol: crate::local::opencode::Protocol::V2,
                legacy_v2_api: false,
            };
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let got = transcript((&endpoint).into(), "ses_long").await.unwrap();
            server.abort();
            let ids: Vec<String> = messages(&got)
                .unwrap()
                .iter()
                .map(|m| m["id"].as_str().unwrap().to_owned())
                .collect();
            let expected: Vec<String> = (0..total).map(|i| format!("msg_{i:03}")).collect();
            assert_eq!(ids, expected);
            assert_eq!(got.pointer("/data/info/id"), Some(&json!("ses_long")));
            assert_eq!(pages.load(std::sync::atomic::Ordering::SeqCst), 3);
        }
    }

    /// A foreground subagent's part reports `running` until it settles; only a settled part whose
    /// subagent still runs is a background one.
    #[test]
    fn only_a_settled_spawn_is_background() {
        let mut captured = Captured::default();
        let mut running = subagent("ses_fg", "running");
        running["state"]["status"] = json!("running");
        assert_eq!(
            subagent_session(&running, "ses_main", &mut captured),
            Some("ses_fg")
        );
        subagent_session(&subagent("ses_bg", "running"), "ses_main", &mut captured);
        assert_eq!(captured.background, HashSet::from(["ses_bg".to_string()]));
        assert_eq!(captured.descendants.len(), 2);
    }

    /// A run a background grandchild's result woke in a continued child session belongs to the turn
    /// that spawned the grandchild, even while a later turn is using that child session.
    #[tokio::test]
    async fn a_grandchild_woken_child_run_belongs_to_the_turn_that_spawned_the_grandchild() {
        let shell = json!([{"type":"tool","name":"shell","callID":"call_run","state":{"status":"completed","input":{"command":"orx exp run exp"},"content":[]}}]);
        let delivery = json!({"id":"msg_delivery","type":"synthetic","time":{"created":30},
            "metadata":{"source":"subagent","childID":"ses_grand"}});
        let exports = HashMap::from([
            (
                "ses_main".to_string(),
                export(
                    "ses_main",
                    None,
                    vec![
                        step(
                            "msg_a",
                            "ses_main",
                            "main-model",
                            10,
                            json!([subagent("ses_child", "completed")]),
                        ),
                        step(
                            "msg_b",
                            "ses_main",
                            "main-model",
                            20,
                            json!([subagent("ses_child", "completed")]),
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
                        step(
                            "msg_child_a",
                            "ses_child",
                            "child-model",
                            11,
                            json!([subagent("ses_grand", "running")]),
                        ),
                        step("msg_child_b", "ses_child", "child-model", 21, json!([])),
                        delivery,
                        step("msg_child_woken", "ses_child", "child-model", 31, shell),
                    ],
                ),
            ),
            (
                "ses_grand".to_string(),
                export(
                    "ses_grand",
                    Some("ses_child"),
                    vec![step("msg_grand", "ses_grand", "grand-model", 12, json!([]))],
                ),
            ),
        ]);
        let (endpoint, _, server) = fake_v2(
            exports,
            json!({"data":[],"cursor":{}}),
            vec![json!({"data":{}})],
        )
        .await;

        // Turn B (started at 20) continues the child session: the woken run is not its own.
        let later = super::super::tests::Recorded::default();
        let mut captured = Captured::default();
        capture_tree(
            &later,
            (&endpoint).into(),
            "ses_main",
            20,
            &mut captured,
            false,
            None,
        )
        .await
        .unwrap();
        let mut ids: Vec<_> = later.samples.into_inner().unwrap().into_keys().collect();
        ids.sort();
        assert_eq!(ids, ["msg_b", "msg_child_b"]);
        assert!(captured.background.is_empty());

        // Turn A's held watcher accounts the grandchild and the child run it woke, with its tool.
        let held = super::super::tests::Recorded::default();
        super::super::settle_watch(
            &held,
            &json!({"v2": true, "native": "ses_main", "startedAt": 10,
                "roots": [["ses_grand", "ses_child"]]}),
            super::super::tests::live(endpoint),
        )
        .await;
        server.abort();
        let mut ids: Vec<_> = held.samples.into_inner().unwrap().into_keys().collect();
        ids.sort();
        assert_eq!(ids, ["msg_child_woken", "msg_grand"]);
        assert_eq!(
            held.invokers.into_inner().unwrap()["msg_child_woken:0"],
            "child-model"
        );
        assert!(held
            .evidence
            .into_inner()
            .unwrap()
            .iter()
            .any(|part| part.id == "msg_child_woken:0"));
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
