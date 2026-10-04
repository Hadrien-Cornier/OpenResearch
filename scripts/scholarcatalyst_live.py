#!/usr/bin/env python3
"""Run a private, budget-limited ScholarCatalyst diagnostic with OpenRouter."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
import time
import threading
import tempfile
import fcntl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import scholarcatalyst_eval as ev


MODEL = "z-ai/glm-5.3-flash"
PROVIDER = "z-ai"
QUANTIZATION = "fp8"
INPUT_PRICE = 0.15
OUTPUT_PRICE = 0.50
TOTAL_BUDGET = 5.00
ARM_INPUT_CAP = 80_000
ARM_OUTPUT_CAP = 24_000
MAX_OUTPUT = 4096
MAX_TOOL_ROUNDS = 4
MAX_DISCOVER_CALLS = 6
REQUEST_TIMEOUT = 240
DISCOVER_TIMEOUT = 60
API_URL = "https://openrouter.ai/api/v1/chat/completions"
ARXIV_RE = re.compile(r"(?:arxiv(?:\.org/(?:abs|pdf)/)?[:/ ]*)?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


class LiveError(Exception):
    pass


def canonical_arxiv(value: str) -> Optional[str]:
    match = ARXIV_RE.search(value or "")
    return "arxiv_" + match.group(1) if match else None


def end_of_cutoff_month(value: str) -> Optional[str]:
    parsed = ev.parse_month(value)
    if parsed is None or parsed[1] == 0:
        return None
    year, month = parsed
    day = (31 if month in (1, 3, 5, 7, 8, 10, 12) else
           30 if month in (4, 6, 9, 11) else
           29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28)
    return "{:04d}-{:02d}-{:02d}".format(year, month, day)


def jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise LiveError("{}:{}: invalid JSON".format(path, line_number)) from error
                if not isinstance(row, dict):
                    raise LiveError("{}:{}: each row must be an object".format(path, line_number))
                yield row


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    path.chmod(0o600)


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def make_cohort(bench: Path, out: Path, seed: int) -> Dict[str, Any]:
    if (out / "tasks.jsonl").exists() or (out / "manifest.private.json").exists():
        raise LiveError("refuse to overwrite an existing cohort")
    queries, by_id = ev.load_queries(bench / "queries.jsonl")
    corpus = ev.load_corpus(bench / "corpus.jsonl")
    rng = random.Random(seed)
    groups = {kind: [q for q in queries if q["type"] == kind]
              for kind in ("core_query", "subfield_query")}
    for values in groups.values():
        rng.shuffle(values)
    selected: List[Dict[str, str]] = []
    project_ids: Set[str] = set()
    quotas = (("core_query", 25), ("subfield_query", 25))
    for kind, quota in quotas:
        for query in groups[kind]:
            # A source paper is the released dataset's project identifier.
            project = query["paper_id"] or query["id"]
            if project in project_ids:
                continue
            project_ids.add(project)
            selected.append(query)
            if sum(item["type"] == kind for item in selected) == quota:
                break
        if sum(item["type"] == kind for item in selected) != quota:
            raise LiveError("cannot select {} distinct source projects for {}".format(quota, kind))
    tasks = [ev.task_record(q) for q in selected]
    private_dir(out)
    ev.write_jsonl(out / "tasks.jsonl", tasks)
    titles = ev.title_index(corpus)
    manifest = {
        "schema_version": 1,
        "protocol": "private_live_diagnostic",
        "seed": seed,
        "tasks_sha256": ev.sha256(out / "tasks.jsonl"),
        "data_sha256": {name: ev.sha256(bench / name) for name in
                        ("queries.jsonl", "corpus.jsonl")},
        "queries": [{
            "task_id": ev.task_record(q)["id"],
            "query_id": q["id"],
            "source_id": q["paper_id"],
            "source_title": (q["paper_title"] or
                             corpus.get(q["paper_id"], {}).get("title", "")),
            "source_alias_ids": sorted(ev.source_aliases(q, corpus, titles)),
            "type": q["type"],
            "cutoff": q["cutoff"],
        } for q in selected],
        "source_projects": len(project_ids),
    }
    write_json(out / "manifest.private.json", manifest)
    return {"task_count": len(tasks), "core_query": 25, "subfield_query": 25,
            "source_projects": len(project_ids), "seed": seed,
            "tasks": str(out / "tasks.jsonl"), "private_manifest": str(out / "manifest.private.json")}


def read_map(out: Path) -> Dict[str, Dict[str, Any]]:
    manifest = json.loads((out / "manifest.private.json").read_text(encoding="utf-8"))
    tasks_hash = ev.sha256(out / "tasks.jsonl")
    if manifest.get("tasks_sha256") != tasks_hash:
        raise LiveError("task file does not match the private manifest")
    return {row["task_id"]: row for row in manifest["queries"]}


def corpus_indexes(bench: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]], Dict[str, List[str]]]:
    corpus = ev.load_corpus(bench / "corpus.jsonl")
    by_arxiv: Dict[str, List[str]] = {}
    by_title = ev.title_index(corpus)
    for doc_id, row in corpus.items():
        canon = canonical_arxiv(doc_id)
        if canon:
            by_arxiv.setdefault(canon, []).append(doc_id)
    return corpus, by_arxiv, by_title


def actual_date(record: Dict[str, Any], corpus_row: Optional[Dict[str, str]]) -> str:
    date = record.get("publicationDate") or record.get("published") or ""
    if date:
        return str(date)
    if corpus_row and corpus_row.get("published"):
        return corpus_row["published"]
    docid = canonical_arxiv(str(record.get("id", ""))) or ""
    parsed = ev.parse_month("", docid)
    return "{:04d}-{:02d}".format(*parsed) if parsed else ""


def dates_after_cutoff(record: Dict[str, Any], corpus_row: Optional[Dict[str, str]],
                       doc_id: Optional[str], cutoff: str) -> Tuple[bool, bool]:
    if ev.parse_month(cutoff) is None:
        return False, False
    dates = [str(value) for value in (record.get("publicationDate"), record.get("published"),
             (corpus_row or {}).get("published")) if value]
    if not dates and doc_id:
        parsed = ev.parse_month("", canonical_arxiv(doc_id) or doc_id)
        if parsed:
            dates = ["{:04d}-{:02d}".format(*parsed)]
    known = [value for value in dates if ev.parse_month(value, doc_id or "") is not None]
    if not known:
        return False, True
    return any(ev.is_after_cutoff(value, doc_id or "", cutoff) for value in known), False


def normalize_candidate(record: Dict[str, Any], corpus: Dict[str, Dict[str, str]],
                        by_arxiv: Dict[str, List[str]], by_title: Dict[str, List[str]],
                        task: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    raw_id = str(record.get("id", ""))
    canonical = canonical_arxiv(raw_id)
    choices = by_arxiv.get(canonical, []) if canonical else []
    if not choices and raw_id in corpus:
        choices = [raw_id]
    title = str(record.get("title", ""))
    if not choices and title:
        choices = by_title.get(ev.normalize_title(title), [])
    doc_id = choices[0] if len(choices) == 1 else None
    corpus_row = corpus.get(doc_id) if doc_id else None
    published = actual_date(record, corpus_row)
    alias_ids = set(task["source_alias_ids"])
    source_title = ev.normalize_title(task.get("source_title", ""))
    candidate_title = ev.normalize_title(title or (corpus_row or {}).get("title", ""))
    source_hit = (bool(doc_id and doc_id in alias_ids) or
                  bool(canonical and task.get("source_id") and canonical == canonical_arxiv(task["source_id"])) or
                  bool(source_title and candidate_title and source_title == candidate_title))
    if source_hit:
        return None, "source"
    cutoff = task.get("cutoff", "")
    if ev.parse_month(cutoff) is not None:
        future, unknown = dates_after_cutoff(record, corpus_row, doc_id or raw_id, cutoff)
        if unknown:
            return None, "unknown_date"
        if future:
            return None, "future"
    # The model sees live paper metadata only. Benchmark membership and corpus IDs stay private.
    safe = {"arxiv_id": canonical.removeprefix("arxiv_") if canonical else "",
            "title": title or (corpus_row or {}).get("title", ""),
            "abstract": str(record.get("abstract", ""))[:1800], "published": published,
            "snippets": bounded_snippets(record.get("snippets", record.get("match", []))),
            "_doc_id": doc_id, "_key": ""}
    safe["_key"] = doc_id or canonical or "title_" + hashlib.sha256(
        ev.normalize_title(safe["title"]).encode("utf-8")).hexdigest()[:20]
    return safe, "unmapped" if doc_id is None else "allowed"


def bounded_snippets(value: Any) -> List[str]:
    snippets: List[str] = []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [item.get("snippet", "") if isinstance(item, dict) else item for item in value]
    else:
        values = []
    for item in values[:3]:
        if isinstance(item, str) and item.strip():
            snippets.append(item.strip()[:500])
    return snippets


TOOL_DEFS = [{"type": "function", "function": {
    "name": "orx_discover_embedding", "description": "Search alphaXiv by semantic meaning.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
    "name": "orx_discover_keyword", "description": "Search alphaXiv full text by exact terms.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False}}}]


class LiveRunner:
    def __init__(self, bench: Path, out: Path, config: Dict[str, Any],
                 discover_fn=None, request_fn=None):
        self.bench, self.out, self.config = bench, out, config
        self.lock = threading.RLock()
        self.discover_fn = discover_fn or self.discover
        self.request_fn = request_fn or self.request
        self.corpus, self.by_arxiv, self.by_title = corpus_indexes(bench)
        self.tasks = list(jsonl(out / "tasks.jsonl"))
        self.task_map = read_map(out)
        self.labels = self._load_labels()
        self.cache_path = out / "retrieval-cache.private.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.ledger_path = out / "budget-ledger.private.json"
        self.lock_path = out / ".live-run.lock"
        self.ledger = self._load_ledger()

    def _load_labels(self) -> Dict[str, Set[str]]:
        queries, by_id = ev.load_queries(self.bench / "queries.jsonl")
        positives: Dict[str, Set[str]] = {}
        for kind in ("core_query", "subfield_query"):
            rows, _ = ev.load_relations(self.bench, kind, by_id, self.corpus)
            positives.update(rows)
        return {row["id"]: positives[query["query_id"]]
                for row in self.tasks for query in [self.task_map[row["id"]]]}

    def _load_ledger(self) -> Dict[str, Any]:
        default = {"schema_version": 1, "hard_budget_usd": self.config["total_budget"],
                   "model": self.config["model"], "provider": self.config["provider"],
                   "config_fingerprint": self.config_fingerprint(),
                   "arms": {}, "inflight": {}, "halted": False}
        if not self.ledger_path.exists():
            return default
        ledger = json.loads(self.ledger_path.read_text())
        if ledger.get("inflight") or ledger.get("halted"):
            raise LiveError("budget ledger has an uncertain request; do not resume or repeat it")
        if ledger.get("config_fingerprint") != self.config_fingerprint():
            raise LiveError("model, price, cap, reasoning, data, cohort, or skill config differs from the ledger")
        return ledger

    def config_fingerprint(self) -> str:
        skill_path = Path(__file__).resolve().parents[1] / "agent-skills/orx-lit-review/SKILL.md"
        value = {**self.config, "tasks_sha256": ev.sha256(self.out / "tasks.jsonl"),
            "corpus_sha256": ev.sha256(self.bench / "corpus.jsonl"),
            "queries_sha256": ev.sha256(self.bench / "queries.jsonl"),
            "core_relations_sha256": ev.sha256(self.bench / "rels/core_query.jsonl"),
            "subfield_relations_sha256": ev.sha256(self.bench / "rels/subfield_query.jsonl"),
            "manifest_sha256": ev.sha256(self.out / "manifest.private.json"),
            "runner_sha256": ev.sha256(Path(__file__)),
            "skill_sha256": ev.sha256(skill_path)}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def save_ledger(self) -> None:
        write_json(self.ledger_path, self.ledger)

    def arm_state(self, arm: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        state_key = arm + ("|" + task_id if task_id else "")
        return self.ledger["arms"].setdefault(state_key, {"prompt_tokens": 0, "completion_tokens": 0,
            "reserved_input_tokens": 0, "reserved_output_tokens": 0, "actual_cost_usd": 0.0,
            "requests": 0, "reserved_cost_usd": 0.0})

    def reserve(self, arm: str, task_id: str, body: Dict[str, Any], max_tokens: int) -> str:
      with self.lock:
        if self.ledger.get("halted"):
            raise LiveError("budget ledger halted after an uncertain or over-budget request")
        state = self.arm_state(arm, task_id)
        encoded = json.dumps({"messages": body["messages"], "tools": body.get("tools", [])},
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        input_bound = len(encoded) + 1024
        if state["prompt_tokens"] + state["reserved_input_tokens"] + input_bound > self.config["arm_input_cap"]:
            raise LiveError("{} reached its input-token cap before the next request".format(arm))
        if state["completion_tokens"] + state["reserved_output_tokens"] + max_tokens > self.config["arm_output_cap"]:
            raise LiveError("{} reached its output-token cap before the next request".format(arm))
        cost = input_bound * self.config["input_price"] / 1_000_000 + max_tokens * self.config["output_price"] / 1_000_000
        committed = sum(row["actual_cost_usd"] + row["reserved_cost_usd"] for row in self.ledger["arms"].values())
        if committed + cost > self.config["total_budget"] + 1e-12:
            raise LiveError("request reservation would exceed the hard global budget")
        rid = "req_" + hashlib.sha256((arm + str(state["requests"]) + str(time.time_ns())).encode()).hexdigest()[:20]
        state["reserved_input_tokens"] += input_bound
        state["reserved_output_tokens"] += max_tokens
        state["reserved_cost_usd"] += cost
        state["requests"] += 1
        self.ledger["inflight"][rid] = {"id": rid, "arm": arm, "task_id": task_id,
                                         "input_bound": input_bound,
                                         "max_output": max_tokens, "reserved_cost_usd": cost}
        self.save_ledger()
        return rid

    def settle(self, rid: str, usage: Dict[str, Any]) -> Dict[str, Any]:
      with self.lock:
        flight = self.ledger.get("inflight", {}).get(rid)
        if not flight:
            raise LiveError("request ledger does not match the response")
        if not isinstance(usage, dict):
            usage = {}
        required = (usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("cost"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or value < 0 for value in required):
            self.ledger["halted"] = True
            self.save_ledger()
            raise LiveError("response lacks metered tokens or cost; budget remains reserved")
        prompt, completion, actual_cost = int(required[0]), int(required[1]), float(required[2])
        if prompt > flight["input_bound"] or completion > flight["max_output"]:
            self.ledger["halted"] = True
            self.save_ledger()
            raise LiveError("metered usage exceeded its pre-request reservation")
        estimate = prompt * self.config["input_price"] / 1_000_000 + completion * self.config["output_price"] / 1_000_000
        actual_cost = max(actual_cost, estimate)
        state = self.arm_state(flight["arm"], flight["task_id"])
        state["reserved_input_tokens"] -= flight["input_bound"]
        state["reserved_output_tokens"] -= flight["max_output"]
        state["reserved_cost_usd"] -= flight["reserved_cost_usd"]
        state["prompt_tokens"] += prompt
        state["completion_tokens"] += completion
        state["actual_cost_usd"] += actual_cost
        state.setdefault("metered_cost_usd", 0.0)
        state["metered_cost_usd"] += float(required[2])
        del self.ledger["inflight"][rid]
        over_reservation = actual_cost > flight["reserved_cost_usd"] + 1e-9
        if over_reservation:
            self.ledger["halted"] = True
        self.save_ledger()
        if over_reservation:
            raise LiveError("metered cost exceeded its reservation; the ledger records the charge and halts")
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "reported_cost_usd": float(required[2]), "price_estimate_usd": estimate,
                "budgeted_cost_usd": actual_cost}

    def request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LiveError("set OPENROUTER_API_KEY to run the paid evaluation")
        request = urllib.request.Request(API_URL, data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/alphaXiv/OpenResearch",
                     "X-Title": "ScholarCatalyst diagnostic"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LiveError("provider request failed; its charge is uncertain, so the run stops") from error

    def discover(self, primitive: str, query: str, cutoff: str) -> List[Dict[str, Any]]:
        bound = end_of_cutoff_month(cutoff)
        command = ["orx", "discover", primitive, query, "--limit", "15", "--no-telemetry"]
        if bound:
            command += ["--published-before", bound]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=DISCOVER_TIMEOUT, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LiveError("orx discovery failed; preserve this partial run and stop") from error
        if result.returncode:
            raise LiveError("orx discovery returned an error; this call is not retried")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise LiveError("orx discovery returned invalid JSON") from error
        if not isinstance(value, list):
            raise LiveError("orx discovery result must be a JSON array")
        return value

    def search(self, arm: str, task: Dict[str, Any], primitive: str, query: str,
               seen: Dict[str, Dict[str, Any]], counts: Dict[str, int]) -> Dict[str, Any]:
        if primitive not in ("embedding", "keyword") or not query.strip() or len(query) > 500:
            raise LiveError("tool arguments do not match the discovery contract")
        if counts["discover_calls"] >= self.config["max_discover_calls"]:
            return {"error": "discovery call limit reached"}
        key = hashlib.sha256(json.dumps([primitive, query.strip(), task["cutoff"]],
                                       separators=(",", ":")).encode()).hexdigest()
        with self.lock:
            cached = self.cache.get(key)
        if cached is None:
            raw = self.discover_fn(primitive, query.strip(), task["cutoff"])
            with self.lock:
                cached = self.cache.setdefault(key, raw)
                write_json(self.cache_path, self.cache)
        counts["discover_calls"] += 1
        allowed: List[Dict[str, Any]] = []
        counts["excluded_source"] = counts.get("excluded_source", 0)
        counts["excluded_future"] = counts.get("excluded_future", 0)
        counts["excluded_unknown_date"] = counts.get("excluded_unknown_date", 0)
        counts["unmapped_results"] = counts.get("unmapped_results", 0)
        for record in cached:
            if not isinstance(record, dict):
                continue
            safe, disposition = normalize_candidate(record, self.corpus, self.by_arxiv,
                                                      self.by_title, task)
            if disposition == "source":
                counts["excluded_source"] += 1
            elif disposition == "future":
                counts["excluded_future"] += 1
            elif disposition == "unknown_date":
                counts["excluded_unknown_date"] += 1
            if safe is None:
                continue
            if disposition == "unmapped":
                counts["unmapped_results"] += 1
            allowed.append(safe)
        public: List[Dict[str, Any]] = []
        bytes_used = 0
        for candidate in allowed[:15]:
            row = {key: candidate[key] for key in ("arxiv_id", "title", "abstract", "published", "snippets")}
            row["abstract"] = row["abstract"][:700]
            row["snippets"] = row["snippets"][:2]
            row["snippets"] = [item[:300] for item in row["snippets"]]
            size = len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
            if bytes_used + size > 18_000:
                break
            bytes_used += size
            public.append(row)
            seen.setdefault(candidate["_key"], candidate)
        counts["tool_payload_truncated"] = counts.get("tool_payload_truncated", 0) + (len(allowed) - len(public))
        return {"results": public, "cutoff": end_of_cutoff_month(task["cutoff"])}

    def arm_prompt(self, arm: str, task: Dict[str, Any]) -> str:
        cutoff = end_of_cutoff_month(task["cutoff"]) or "unknown"
        prompt = ("Research question: " + task["question"] + "\nPublication cutoff: " + cutoff +
                  "\nGoal: find prior papers that can help advance this research question. "
                  "Return JSON only: {\"papers\":[{\"title\":\"...\",\"arxiv_id\":\"...\"}]} "
                  "with at most 15 ranked papers. Do not include the source paper.")
        if arm == "closed_book":
            return prompt + " Use your existing knowledge. Do not claim that you searched."
        if arm == "orx_skill":
            skill_path = Path(__file__).resolve().parents[1] / "agent-skills/orx-lit-review/SKILL.md"
            skill_text = skill_path.read_text(encoding="utf-8")
            return ("Apply this exact local orx literature-review skill, within the two available alphaXiv "
                    "search tools:\n\n" + skill_text + "\n\n" + prompt +
                    "\nFor final selection, keep only papers returned by successful tool calls. If no selected "
                    "paper survives, use the first 15 unique papers in tool observation order.")
        return prompt + " Use the supplied search tools. Rank only papers returned by successful tool calls. " \
            "Keep unsupported choices as slots."

    def run_model(self, arm: str, task: Dict[str, Any], seen: Dict[str, Dict[str, Any]],
                  counts: Dict[str, int], conversation: List[Dict[str, Any]],
                  allow_tools: bool) -> Tuple[str, Dict[str, Any]]:
        arm_state = self.arm_state(arm, task["id"])
        remaining_out = self.config["arm_output_cap"] - arm_state["completion_tokens"] - arm_state["reserved_output_tokens"]
        max_tokens = min(self.config["max_output"], remaining_out)
        if max_tokens <= 0:
            raise LiveError("{} reached its output-token cap".format(arm))
        tools = TOOL_DEFS if allow_tools and arm != "closed_book" else None
        body: Dict[str, Any] = {"model": self.config["model"], "provider": {
            "only": [self.config["provider"]], "allow_fallbacks": False,
            "quantizations": [self.config["quantization"]], "require_parameters": True},
            "max_price": {"prompt": self.config["input_price"],
                          "completion": self.config["output_price"]},
            "temperature": 0, "max_tokens": max_tokens,
            "reasoning": {"enabled": bool(self.config["reasoning_enabled"]),
                          "effort": self.config["reasoning_effort"]},
            "usage": {"include": True}, "messages": conversation}
        body["provider"]["max_price"] = body.pop("max_price")
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        rid = self.reserve(arm, task["id"], body, max_tokens)
        try:
            with self.lock:
                append_jsonl(self.out / "requests.private.jsonl", {"request_id": rid, "task_id": task["id"],
                    "arm": arm, "timestamp_unix": time.time(), "request": body})
            response = self.request_fn(body)
        except Exception as error:
            with self.lock:
                self.ledger["halted"] = True
                self.save_ledger()
            raise LiveError("provider request ended without confirmed metering; the reservation stays active") from error
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        charge = self.settle(rid, usage)
        with self.lock:
            append_jsonl(self.out / "captures.private.jsonl", {
                "arm": arm, "task_id": task["id"], "request_id": rid,
                "timestamp_unix": time.time(), "request": body, "response": response, "usage": charge})
        try:
            choice = response["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("assistant message must be an object")
        except (KeyError, IndexError, TypeError) as error:
            raise LiveError("provider response has no valid assistant message") from error
        return message, charge

    def run_task_arm(self, arm: str, task: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.monotonic()
        internal = self.task_map[task["id"]]
        conversation: List[Dict[str, Any]] = [{"role": "system", "content":
            "You are a literature search assistant. Return a ranked paper list as JSON."},
            {"role": "user", "content": self.arm_prompt(arm, task)}]
        observed: Dict[str, Dict[str, Any]] = {}
        counts: Dict[str, int] = {"discover_calls": 0, "excluded_source": 0,
            "excluded_future": 0, "excluded_unknown_date": 0, "unmapped_results": 0}
        raw_items: List[Dict[str, Any]] = []
        for round_index in range(self.config["max_tool_rounds"] + 1):
            message, _ = self.run_model(arm, task, observed, counts, conversation,
                                        allow_tools=round_index < self.config["max_tool_rounds"])
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise LiveError("tool_calls must be an array")
            if not calls:
                content = message.get("content") or ""
                raw_items = parse_final(content)
                break
            if arm == "closed_book" or round_index >= self.config["max_tool_rounds"]:
                raise LiveError("model exceeded its allowed tool-call rounds")
            conversation.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise LiveError("tool call must contain a function object")
                function = call.get("function", {})
                name = function.get("name")
                try:
                    args = json.loads(function.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    raise LiveError("tool arguments must be a JSON object")
                primitive = "embedding" if name == "orx_discover_embedding" else "keyword" if name == "orx_discover_keyword" else ""
                result = self.search(arm, internal, primitive, str(args.get("query", "")), observed, counts)
                conversation.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                     "content": json.dumps(result, ensure_ascii=False)})
        else:
            raise LiveError("model did not return a final answer")
        slots: List[Dict[str, Any]] = []
        invalid = 0
        seen_ranked: Set[str] = set()
        for item in raw_items[:15]:
            slot, reason = self.resolve_final(arm, item, internal, observed)
            doc_id = slot.get("doc_id")
            if doc_id and doc_id in seen_ranked:
                slot = {"doc_id": None, "title": slot.get("title", ""),
                        "source": "invalid", "reason": "duplicate"}
                reason = "duplicate"
            elif doc_id:
                seen_ranked.add(doc_id)
            if reason:
                invalid += 1
            slots.append(slot)
        if arm == "orx_skill":
            has_supported = any(slot.get("source") == "tool" for slot in slots)
        else:
            has_supported = False
        if arm == "orx_skill" and not has_supported:
            slots = [{"doc_id": row.get("_doc_id"), "title": row["title"],
                      "source": "tool_fallback" if row.get("_doc_id") else "unmapped"}
                     for row in list(observed.values())[:15]]
        candidate_ids = {row["_doc_id"] for row in observed.values() if row.get("_doc_id")}
        metrics = self.metrics(slots, self.labels[task["id"]], candidate_ids)
        return {"task_id": task["id"], "query_type": task["type"], "arm": arm,
            "ranking_slots": slots, "candidate_ids": sorted(candidate_ids),
            "candidate_recall": (len(candidate_ids & self.labels[task["id"]]) /
                                 len(self.labels[task["id"]]) if self.labels[task["id"]] else None),
            "metrics": metrics, "invalid_slots": invalid,
            "elapsed_seconds": round(time.monotonic() - started_at, 3), **counts}

    def resolve_final(self, arm: str, item: Dict[str, Any], task: Dict[str, Any],
                      observed: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        title = str(item.get("title", ""))
        arxiv = str(item.get("arxiv_id", ""))
        canon = canonical_arxiv(arxiv)
        choices = self.by_arxiv.get(canon, []) if canon else []
        normalized_title = ev.normalize_title(title)
        title_choices = self.by_title.get(normalized_title, []) if normalized_title else []
        if len(choices) > 1 or len(title_choices) > 1:
            return {"doc_id": None, "title": title, "source": "invalid", "reason": "ambiguous"}, "ambiguous"
        id_choice = choices[0] if choices else None
        title_choice = title_choices[0] if title_choices else None
        if id_choice and title_choice and id_choice != title_choice:
            return {"doc_id": None, "title": title, "source": "invalid", "reason": "id_title_conflict"}, "conflict"
        doc_id = id_choice or title_choice
        if not doc_id:
            observed_match = next((candidate for candidate in observed.values()
                if (canon and candidate.get("arxiv_id") == canon.removeprefix("arxiv_")) or
                   (normalized_title and ev.normalize_title(candidate.get("title", "")) == normalized_title)), None)
            if observed_match:
                return {"doc_id": None, "title": title or observed_match["title"],
                        "source": "tool", "arxiv_id": canon}, ""
            return {"doc_id": None, "title": title, "source": "invalid", "reason": "unmapped"}, "unmapped"
        aliases = set(task["source_alias_ids"])
        if doc_id in aliases:
            return {"doc_id": None, "title": title, "source": "invalid", "reason": "source_paper"}, "source"
        row = self.corpus[doc_id]
        if task.get("cutoff") and ev.parse_month(task["cutoff"]) is not None:
            future, unknown = dates_after_cutoff({}, row, doc_id, task["cutoff"])
            if unknown:
                return {"doc_id": None, "title": title, "source": "invalid", "reason": "unknown_date"}, "unknown_date"
            if future:
                return {"doc_id": None, "title": title, "source": "invalid", "reason": "future"}, "future"
        observed_ids = {row.get("_doc_id") for row in observed.values() if row.get("_doc_id")}
        source = "tool" if doc_id in observed_ids else "memory"
        if arm in ("plain_tools", "orx_skill") and source != "tool":
            return {"doc_id": None, "title": title or row["title"],
                    "source": "invalid", "reason": "not_returned_by_tool"}, "not_returned_by_tool"
        return {"doc_id": doc_id, "title": title or row["title"], "source": source}, ""

    @staticmethod
    def metrics(slots: List[Dict[str, Any]], positives: Set[str], candidates: Set[str]) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        duplicate_seen: Set[str] = set()
        unique_slots = []
        for slot in slots:
            doc_id = slot.get("doc_id")
            if doc_id and doc_id in duplicate_seen:
                unique_slots.append({"doc_id": None})
            else:
                unique_slots.append(slot)
                if doc_id:
                    duplicate_seen.add(doc_id)
        for k in (5, 15):
            hits = sum(1 for slot in unique_slots[:k] if slot.get("doc_id") in positives)
            values["recall@{}".format(k)] = hits / len(positives) if positives else None
        dcg = sum(1.0 / math.log2(position + 2) for position, slot in enumerate(unique_slots[:15])
                  if slot.get("doc_id") in positives)
        ideal = sum(1.0 / math.log2(position + 2) for position in range(min(len(positives), 15)))
        values["ndcg@15"] = dcg / ideal if ideal else None
        values["candidate_recall"] = len(candidates & positives) / len(positives) if positives else None
        return values

    def execute(self) -> Dict[str, Any]:
        if self.ledger.get("halted") or self.ledger.get("inflight"):
            raise LiveError("ledger has an uncertain request; do not resume")
        lock_file = (self.out / ".live-run.lock").open("a+")
        os.chmod(self.out / ".live-run.lock", 0o600)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise LiveError("another pilot process owns this output directory") from error
        try:
            self.ledger = self._load_ledger()
            return self._execute_locked()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def _execute_locked(self) -> Dict[str, Any]:
        run_path = self.out / "live-results.private.jsonl"
        completed = {(r.get("task_id"), r.get("arm")) for r in jsonl(run_path)} if run_path.exists() else set()
        arms = ("closed_book", "plain_tools", "orx_skill")
        pending = [task for task in self.tasks if any((task["id"], arm) not in completed for arm in arms)]

        def execute_task(task: Dict[str, Any]) -> None:
            for arm in arms:
                if (task["id"], arm) in completed:
                    continue
                arm_started = time.monotonic()
                try:
                    row = self.run_task_arm(arm, task)
                except LiveError as error:
                    row = self.zero_row(task, arm, str(error))
                    row["elapsed_seconds"] = round(time.monotonic() - arm_started, 3)
                with self.lock:
                    append_jsonl(run_path, row)
                    completed.add((task["id"], arm))
                    cost = sum(state["actual_cost_usd"] for state in self.ledger["arms"].values())
                    print(json.dumps({"completed_rows": len(completed), "planned_rows": len(self.tasks) * 3,
                        "task_id": task["id"], "arm": arm, "failed": bool(row.get("missing")),
                        "model_charge_usd": round(cost, 6)}), file=sys.stderr, flush=True)

        # The first pending question checks the provider and tool path before concurrent requests.
        if pending:
            execute_task(pending[0])
        if not self.ledger.get("halted"):
            with ThreadPoolExecutor(max_workers=self.config.get("workers", 4)) as pool:
                futures = [pool.submit(execute_task, task) for task in pending[1:]]
                for future in as_completed(futures):
                    future.result()
        all_rows = list(jsonl(run_path))
        report = self.make_report(all_rows)
        write_json(self.out / "live-report.private.json", report)
        return report

    def zero_row(self, task: Dict[str, Any], arm: str, error: str) -> Dict[str, Any]:
        positives = self.labels[task["id"]]
        score = 0.0 if positives else None
        return {"task_id": task["id"], "query_type": task["type"], "arm": arm,
            "ranking_slots": [], "candidate_ids": [], "candidate_recall": score,
            "metrics": {"recall@5": score, "recall@15": score,
                        "ndcg@15": score, "candidate_recall": score},
            "missing": True, "error": error[:300]}

    def make_report(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        summaries: Dict[str, Any] = {}
        for arm in ("closed_book", "plain_tools", "orx_skill"):
            arm_rows = [r for r in rows if r.get("arm") == arm]
            rows_by_task = {r["task_id"]: r for r in arm_rows}
            by_type: Dict[str, Any] = {}
            for query_type in ("core_query", "subfield_query"):
                cohort = [task for task in self.tasks if task["type"] == query_type]
                values = []
                for task in cohort:
                    row = rows_by_task.get(task["id"])
                    if row is None:
                        row = self.zero_row(task, arm, "missing result row")
                    values.append(row)
                scored = [r for r in values if self.labels[r["task_id"]]]
                metric_names = ("recall@5", "recall@15", "ndcg@15", "candidate_recall")
                by_type[query_type] = {name: (sum(r["metrics"][name] for r in scored
                    if r["metrics"][name] is not None) /
                    sum(r["metrics"][name] is not None for r in scored)) if any(
                    r["metrics"][name] is not None for r in scored) else None for name in metric_names}
                by_type[query_type]["n_queries"] = len(cohort)
                by_type[query_type]["n_scored"] = len(scored)
                by_type[query_type]["missing_rows"] = sum(bool(r.get("missing")) for r in values)
            state_rows = [state for key, state in self.ledger["arms"].items()
                          if key == arm or key.startswith(arm + "|")]
            summaries[arm] = {"metrics": by_type, "n_completed": len(arm_rows),
                "cost_usd": sum(state["actual_cost_usd"] for state in state_rows),
                "prompt_tokens": sum(state["prompt_tokens"] for state in state_rows),
                "completion_tokens": sum(state["completion_tokens"] for state in state_rows),
                "requests": sum(state["requests"] for state in state_rows),
                "reserved_cost_usd": sum(state["reserved_cost_usd"] for state in state_rows),
                "missing_rows": sum(bool(r.get("missing")) for r in arm_rows),
                "successful_rows": sum(not r.get("missing", False) for r in arm_rows)}
        return {"schema_version": 1, "protocol": "private_live_diagnostic",
            "official_benchmark_result": False, "model": self.config["model"],
            "provider": self.config["provider"], "quantization": self.config["quantization"],
            "reasoning_enabled": self.config["reasoning_enabled"],
            "settings": self.config,
            "tool_payload_limits": {"papers": 15, "bytes": 18000, "abstract_characters": 700,
                                    "snippets": 2, "snippet_characters": 300},
            "final_validation": "retain invalid and unjudged slots in both arms; differs from native skill drop step",
            "task_count": len(self.tasks), "completed_rows": len(rows),
            "source_title_unavailable": sum(not row.get("source_title") for row in self.task_map.values()),
            "source_project_selection": "one query per source paper; repeated benchmark query generation may still share upstream project history",
            "data_revision": "ScholarCatalyst pinned revision a5a73467500ada90db4e7641e0697a9591e41a4e",
            "source_policy": "alphaXiv embedding and keyword only; this replays the restricted local orx source profile",
            "arms": summaries, "ledger": str(self.ledger_path)}


def parse_final(content: str) -> List[Dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise LiveError("model final answer is not a JSON object")
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise LiveError("model final answer contains invalid JSON") from error
    papers = payload.get("papers") if isinstance(payload, dict) else None
    if not isinstance(papers, list):
        raise LiveError("model final answer must contain a papers array")
    return [item if isinstance(item, dict) else {} for item in papers[:15]]


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {"model": args.model, "provider": args.provider, "quantization": args.quantization,
        "input_price": args.input_price, "output_price": args.output_price,
        "total_budget": args.total_budget, "arm_input_cap": args.arm_input_cap,
        "arm_output_cap": args.arm_output_cap, "max_output": args.max_output,
        "max_tool_rounds": args.max_tool_rounds, "max_discover_calls": args.max_discover_calls,
        "reasoning_enabled": args.reasoning, "reasoning_effort": args.reasoning_effort,
        "workers": args.workers}


def validate_config(config: Dict[str, Any]) -> None:
    for key in ("input_price", "output_price", "total_budget"):
        value = config[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise LiveError("{} must be a finite positive number".format(key))
    if config["total_budget"] > TOTAL_BUDGET:
        raise LiveError("hard budget cannot exceed the $5 pilot ceiling")
    for key in ("arm_input_cap", "arm_output_cap", "max_output", "max_tool_rounds", "max_discover_calls"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LiveError("{} must be a positive integer".format(key))
    if config["max_output"] > 4096 or config["max_tool_rounds"] > 4 or config["max_discover_calls"] > 6:
        raise LiveError("output and tool limits cannot exceed the reviewed pilot limits")
    if not isinstance(config.get("workers", 4), int) or not 1 <= config.get("workers", 4) <= 4:
        raise LiveError("workers must be between one and four")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="select 25 core and 25 subfield questions")
    prepare.add_argument("--bench-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=530)
    run = sub.add_parser("run", help="run the three-arm pilot; paid calls require --execute")
    run.add_argument("--bench-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--execute", action="store_true", help="send paid OpenRouter requests")
    run.add_argument("--model", default=MODEL)
    run.add_argument("--provider", default=PROVIDER)
    run.add_argument("--quantization", default=QUANTIZATION)
    run.add_argument("--input-price", type=float, default=INPUT_PRICE, help="USD per million prompt tokens")
    run.add_argument("--output-price", type=float, default=OUTPUT_PRICE, help="USD per million output tokens")
    run.add_argument("--total-budget", type=float, default=TOTAL_BUDGET)
    run.add_argument("--arm-input-cap", type=int, default=ARM_INPUT_CAP)
    run.add_argument("--arm-output-cap", type=int, default=ARM_OUTPUT_CAP)
    run.add_argument("--max-output", type=int, default=MAX_OUTPUT)
    run.add_argument("--max-tool-rounds", type=int, default=MAX_TOOL_ROUNDS)
    run.add_argument("--max-discover-calls", type=int, default=MAX_DISCOVER_CALLS)
    run.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="low")
    run.add_argument("--workers", type=int, default=4)
    report = sub.add_parser("report", help="show the private live report")
    report.add_argument("--output-dir", type=Path, required=True)
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            print(json.dumps(make_cohort(args.bench_dir, args.output_dir, args.seed), indent=2))
            return 0
        if args.command == "report":
            path = args.output_dir / "live-report.private.json"
            print(path.read_text(encoding="utf-8") if path.exists() else "No report exists yet.")
            return 0
        config = config_from_args(args)
        validate_config(config)
        if not args.execute:
            tasks = list(jsonl(args.output_dir / "tasks.jsonl"))
            max_per_arm_task = (args.arm_input_cap * args.input_price +
                                args.arm_output_cap * args.output_price) / 1_000_000
            worst_total = len(tasks) * 3 * max_per_arm_task
            print(json.dumps({"paid_calls": False, "tasks": len(tasks), "arms": 3,
                "model": args.model, "provider": args.provider, "quantization": args.quantization,
                "price_per_million": {"input": args.input_price, "output": args.output_price},
                "hard_budget_usd": args.total_budget,
                "cap_based_maximum_usd": round(min(args.total_budget, worst_total), 4),
                "cap_basis": "per task and arm, using 80k input and 24k output defaults",
                "actual_request_reservations": "computed from each serialized prompt before sending",
                "next_step": "add --execute to send requests"}, indent=2))
            return 0
        private_dir(args.output_dir)
        report = LiveRunner(args.bench_dir, args.output_dir, config).execute()
        print(json.dumps(report, indent=2))
    except (LiveError, ev.EvaluationError, OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
