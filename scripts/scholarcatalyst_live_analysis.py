#!/usr/bin/env python3
"""Export aggregate live results and paired uncertainty without private questions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics

import scholarcatalyst_eval as ev
import scholarcatalyst_live as live

ARMS = ("closed_book", "plain_tools", "orx_skill")
METRICS = ("recall@5", "recall@15", "ndcg@15")


def error_category(message):
    text = message.lower()
    for fragment, category in (("input-token cap", "input_reservation_limit"),
        ("output-token cap", "output_limit"), ("hard global budget", "global_budget_limit"),
        ("metering", "uncertain_metering"), ("halted", "uncertain_metering"),
        ("reservation stays", "uncertain_metering"), ("discover", "search_error"),
        ("json", "response_schema"), ("tool arguments", "response_schema"),
        ("tool_calls", "response_schema"), ("assistant message", "response_schema")):
        if fragment in text:
            return category
    return "other_run_error"


def paired_interval(differences, seed=530, samples=10000):
    if not differences:
        return {"mean_delta": None, "interval_95": None, "n_pairs": 0}
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(samples))
    return {"mean_delta": statistics.mean(differences),
            "interval_95": [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]],
            "n_pairs": n, "bootstrap_samples": samples, "seed": seed}


def analyze(bench, evidence):
    tasks = list(live.jsonl(evidence / "tasks.jsonl"))
    manifest = json.loads((evidence / "manifest.private.json").read_text())
    if ev.sha256(evidence / "tasks.jsonl") != manifest["tasks_sha256"]:
        raise ValueError("cohort hash differs from the private manifest")
    mapping = {row["task_id"]: row for row in manifest["queries"]}
    if len({mapping[task["id"]]["source_id"] for task in tasks}) != len(tasks):
        raise ValueError("paired resampling requires one task per source paper")
    rows = list(live.jsonl(evidence / "live-results.private.jsonl"))
    keyed = {}
    for row in rows:
        key = (row["task_id"], row["arm"])
        if key in keyed:
            raise ValueError("duplicate result row")
        if row["task_id"] not in mapping or row["arm"] not in ARMS:
            raise ValueError("result is outside the planned cohort")
        keyed[key] = row
    ledger = json.loads((evidence / "budget-ledger.private.json").read_text())
    raw_report = json.loads((evidence / "live-report.private.json").read_text())
    fingerprint_inputs = {**raw_report["settings"], "tasks_sha256": ev.sha256(evidence / "tasks.jsonl"),
        "corpus_sha256": ev.sha256(bench / "corpus.jsonl"),
        "queries_sha256": ev.sha256(bench / "queries.jsonl"),
        "core_relations_sha256": ev.sha256(bench / "rels/core_query.jsonl"),
        "subfield_relations_sha256": ev.sha256(bench / "rels/subfield_query.jsonl"),
        "manifest_sha256": ev.sha256(evidence / "manifest.private.json"),
        "runner_sha256": ev.sha256(Path(live.__file__)),
        "skill_sha256": ev.sha256(Path(__file__).resolve().parents[1] / "agent-skills/orx-lit-review/SKILL.md")}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_inputs, sort_keys=True).encode()).hexdigest()
    if fingerprint != ledger["config_fingerprint"]:
        raise ValueError("data, labels, guards, runner, skill, or settings differ from the run fingerprint")
    corpus = ev.load_corpus(bench / "corpus.jsonl")
    by_arxiv = {}
    for doc_id in corpus:
        canonical = live.canonical_arxiv(doc_id)
        if canonical:
            by_arxiv.setdefault(canonical, []).append(doc_id)
    by_title = ev.title_index(corpus)
    _, queries = ev.load_queries(bench / "queries.jsonl")
    positives = {}
    for kind in ("core_query", "subfield_query"):
        labels, _ = ev.load_relations(bench, kind, queries, corpus)
        positives.update(labels)
    labels = {task["id"]: positives[mapping[task["id"]]["query_id"]] for task in tasks}
    exposed = {}
    provider_counts = Counter()
    for capture in live.jsonl(evidence / "captures.private.jsonl"):
        key = (capture["task_id"], capture["arm"])
        provider_counts[capture["response"].get("provider", "unknown")] += 1
        candidates = exposed.setdefault(key, set())
        for message in capture["request"]["messages"]:
            if message.get("role") != "tool":
                continue
            payload = json.loads(message.get("content", "{}"))
            for candidate in payload.get("results", []):
                canonical = live.canonical_arxiv(candidate.get("arxiv_id", ""))
                choices = by_arxiv.get(canonical, []) if canonical else []
                if not choices:
                    choices = by_title.get(ev.normalize_title(candidate.get("title", "")), [])
                if len(choices) == 1:
                    candidates.add(choices[0])

    def value(task_id, arm, metric):
        row = keyed.get((task_id, arm))
        return row["metrics"].get(metric) if row else (0.0 if labels[task_id] else None)

    summaries = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        states = [state for key, state in ledger["arms"].items() if key.startswith(arm + "|")]
        groups = {}
        for kind in ("core_query", "subfield_query"):
            ids = [task["id"] for task in tasks if task["type"] == kind and labels[task["id"]]]
            scores = {metric: statistics.mean(value(task_id, arm, metric) for task_id in ids)
                      if ids else None for metric in METRICS}
            scores["confirmed_exposed_candidate_recall"] = (
                statistics.mean(len(exposed.get((task_id, arm), set()) & labels[task_id]) /
                                len(labels[task_id]) for task_id in ids)
                if ids and arm != "closed_book" else None)
            scores["n_scored"] = len(ids)
            groups[kind] = scores
        missing = len(tasks) - sum(not row.get("missing") for row in arm_rows)
        summaries[arm] = {"metrics": groups, "failed_or_missing_rows": missing,
            "errors": dict(Counter(error_category(row.get("error", "missing result"))
                                   for row in arm_rows if row.get("missing"))),
            "metered_cost_usd": sum(state.get("metered_cost_usd", 0) for state in states),
            "budgeted_cost_usd": sum(state["actual_cost_usd"] for state in states),
            "prompt_tokens": sum(state["prompt_tokens"] for state in states),
            "completion_tokens": sum(state["completion_tokens"] for state in states),
            "model_requests": sum(state["requests"] for state in states),
            "mean_seconds_per_planned_row": sum(row.get("elapsed_seconds", 0) for row in arm_rows) / len(tasks),
            "mean_ranked_slots": sum(len(row.get("ranking_slots", [])) for row in arm_rows) / len(tasks),
            "invalid_selection_attempts": sum(row.get("invalid_slots", 0) for row in arm_rows),
            "unjudged_final_slots": sum(slot.get("doc_id") is None and slot.get("source") != "invalid"
                for row in arm_rows for slot in row.get("ranking_slots", [])),
            "tool_payload_omitted_papers": sum(row.get("tool_payload_truncated", 0) for row in arm_rows)}
    comparisons = {}
    for left, right in (("orx_skill", "plain_tools"), ("plain_tools", "closed_book"), ("orx_skill", "closed_book")):
        by_type = {}
        for kind in ("core_query", "subfield_query"):
            ids = [task["id"] for task in tasks if task["type"] == kind and labels[task["id"]]]
            by_type[kind] = {metric: paired_interval([value(task_id, left, metric) - value(task_id, right, metric)
                for task_id in ids]) for metric in METRICS}
        comparisons[left + "_minus_" + right] = by_type
    return {"schema_version": 1, "protocol": "restricted_live_diagnostic", "official_benchmark_result": False,
        "model": raw_report["model"], "provider": raw_report["provider"], "settings": raw_report["settings"],
        "data_revision": raw_report["data_revision"], "verified_fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
        "cohort_sha256": manifest["tasks_sha256"], "task_count": len(tasks), "completed_rows": len(rows),
        "complete": len(rows) == len(tasks) * len(ARMS) and not ledger["inflight"] and not ledger["halted"],
        "metered_cost_usd": sum(state.get("metered_cost_usd", 0) for state in ledger["arms"].values()),
        "budgeted_cost_usd": sum(state["actual_cost_usd"] for state in ledger["arms"].values()),
        "reserved_cost_usd": sum(state["reserved_cost_usd"] for state in ledger["arms"].values()),
        "provider_response_counts": dict(provider_counts), "arms": summaries, "paired_comparisons": comparisons,
        "uncertainty_method": "paired percentile bootstrap over source papers; missing and failed final runs score zero",
        "candidate_method": "known-positive papers in tool payloads of confirmed model responses; includes failed final runs",
        "limits": ["Live corpus differs from the official frozen corpus.",
                   "Sparse labels do not establish relevance of unjudged papers.",
                   "The workflow uses alphaXiv discovery only and common rank validation.",
                   "Model knowledge can include the original source projects.",
                   "A pilot gives preliminary evidence."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new path")
    report = analyze(args.bench_dir, args.evidence_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "task_count", "metered_cost_usd")}, indent=2))


if __name__ == "__main__":
    main()
