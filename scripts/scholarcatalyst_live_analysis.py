#!/usr/bin/env python3
"""Export aggregate live results and paired uncertainty without private questions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import scholarcatalyst_eval as ev
import scholarcatalyst_live as live

ARMS = ("closed_book", "plain_tools", "orx_skill")
METRICS = ("recall@5", "recall@15", "ndcg@15")
QUESTION_TYPES = ("core_query", "subfield_query")


def error_category(message):
    text = message.lower()
    for fragment, category in (("input-token cap", "input_reservation_limit"),
        ("output-token cap", "output_limit"), ("hard global budget", "global_budget_limit"),
        ("metering", "uncertain_metering"), ("metered", "uncertain_metering"),
        ("unconfirmed", "uncertain_metering"), ("halted", "uncertain_metering"),
        ("reservation stays", "uncertain_metering"), ("discover", "search_error"),
        ("json", "response_schema"), ("tool arguments", "response_schema"),
        ("tool_calls", "response_schema"), ("assistant message", "response_schema")):
        if fragment in text:
            return category
    return "other_run_error"


def nonnull_mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def paired_interval(differences, seed=530, samples=10000):
    if not differences:
        return {"mean_delta": None, "interval_95": None, "n_pairs": 0}
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(samples))
    return {"mean_delta": statistics.mean(differences),
            "interval_95": [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]],
            "n_pairs": n, "bootstrap_samples": samples, "seed": seed}


def stratified_interval(groups, seed=530, samples=10000):
    groups = [values for values in groups if values]
    if not groups:
        return paired_interval([])
    rng = random.Random(seed)
    means = sorted(statistics.mean(statistics.mean(rng.choices(values, k=len(values)))
                                   for values in groups) for _ in range(samples))
    return {"mean_delta": statistics.mean(statistics.mean(values) for values in groups),
            "interval_95": [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]],
            "n_pairs": sum(len(values) for values in groups), "bootstrap_samples": samples,
            "seed": seed, "strata": "question type; equal weight for each type"}


def grouped_intervals(source_groups, seed=530, samples=10000):
    """Resample whole source projects jointly across questions and question types."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    kinds = [kind for kind in QUESTION_TYPES
             if any(values.get(kind) for values in source_groups.values())]
    if not source_groups or not kinds:
        raise ValueError("source groups must contain scored questions")
    aggregates = [tuple((sum(values.get(kind, [])), len(values.get(kind, [])))
                        for kind in kinds) for values in source_groups.values()]
    counts = [sum(row[index][1] for row in aggregates) for index in range(len(kinds))]
    observed = [sum(row[index][0] for row in aggregates) / count
                for index, count in enumerate(counts)]
    draws = [[] for _ in kinds]
    balanced = []
    rejected = 0
    rng = random.Random(seed)
    while len(balanced) < samples:
        selected = rng.choices(aggregates, k=len(aggregates))
        draw_counts = [sum(row[index][1] for row in selected) for index in range(len(kinds))]
        if not all(draw_counts):
            rejected += 1
            if rejected > samples * 100:
                raise ValueError("source resamples cannot retain all question types")
            continue
        means = [sum(row[index][0] for row in selected) / count
                 for index, count in enumerate(draw_counts)]
        for values, mean in zip(draws, means):
            values.append(mean)
        balanced.append(statistics.mean(means))

    def result(values, mean, n_pairs):
        values.sort()
        return {"mean_delta": mean,
                "interval_95": [values[int(samples * 0.025)],
                                values[min(samples - 1, int(samples * 0.975))]],
                "n_pairs": n_pairs, "n_source_projects": len(aggregates),
                "bootstrap_samples": samples, "seed": seed,
                "resampling_unit": "source project; all questions remain together",
                "rejected_empty_type_resamples": rejected}

    intervals = {kind: result(values, mean, count)
                 for kind, values, mean, count in zip(kinds, draws, observed, counts)}
    intervals["balanced_pilot"] = result(balanced, statistics.mean(observed), sum(counts))
    intervals["balanced_pilot"]["strata"] = "question type; equal weight for each type"
    return intervals


def omit_financial_fields(value):
    """Remove financial fields at every level, including fingerprint inputs."""
    if isinstance(value, dict):
        return {key: omit_financial_fields(item) for key, item in value.items()
                if not any(term in key.lower() for term in ("price", "budget", "charge", "cost", "usd"))}
    if isinstance(value, list):
        return [omit_financial_fields(item) for item in value]
    return value


def analyze(bench, evidence, public_report=False):
    tasks = list(live.jsonl(evidence / "tasks.jsonl"))
    manifest = json.loads((evidence / "manifest.private.json").read_text())
    if ev.sha256(evidence / "tasks.jsonl") != manifest["tasks_sha256"]:
        raise ValueError("cohort hash differs from the private manifest")
    mapping = {row["task_id"]: row for row in manifest["queries"]}
    task_ids = {task["id"] for task in tasks}
    if not tasks or len(task_ids) != len(tasks) or len(mapping) != len(manifest["queries"]):
        raise ValueError("cohort contains no tasks or duplicate task IDs")
    if task_ids != set(mapping):
        raise ValueError("manifest differs from the planned task cohort")
    for task in tasks:
        mapped = mapping[task["id"]]
        if task["type"] not in QUESTION_TYPES or task["type"] != mapped["type"] or not mapped["source_id"]:
            raise ValueError("manifest source or question type is invalid")
    raw_report = json.loads((evidence / "live-report.private.json").read_text())
    arms = raw_report["settings"].get("selected_arms", list(ARMS))
    if not isinstance(arms, (list, tuple)) or not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("selected arms are invalid")
    grouped_sources = len({mapping[task["id"]]["source_id"] for task in tasks}) < len(tasks)
    bootstrap_seed = manifest.get("seed", 530)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("manifest seed must be an integer")
    rows = list(live.jsonl(evidence / "live-results.private.jsonl"))
    keyed = {}
    for row in rows:
        key = (row["task_id"], row["arm"])
        if key in keyed:
            raise ValueError("duplicate result row")
        if row["task_id"] not in mapping or row["arm"] not in arms:
            raise ValueError("result is outside the planned cohort")
        keyed[key] = row
    ledger = json.loads((evidence / "budget-ledger.private.json").read_text())
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
    if any(not positive for positive in labels.values()):
        raise ValueError("every planned question must have official positive labels")
    label_audit = {}
    for kind in ("core_query", "subfield_query"):
        ids = [task["id"] for task in tasks if task["type"] == kind]
        future_by_task = {task_id: sum(ev.is_after_cutoff(corpus[doc_id]["published"], doc_id,
            mapping[task_id]["cutoff"]) for doc_id in labels[task_id]) for task_id in ids}
        label_audit[kind] = {"positive_pairs": sum(len(labels[task_id]) for task_id in ids),
                            "future_positive_pairs": sum(future_by_task.values()),
                            "queries_with_future_positives": sum(count > 0 for count in future_by_task.values())}
    exposed = {}
    provider_counts = Counter()
    capture_path = evidence / "captures.private.jsonl"
    request_count = sum(state["requests"] for state in ledger["arms"].values())
    inflight_ids = set(ledger["inflight"])
    terminal = ledger.get("terminal_unconfirmed", {})
    if not isinstance(terminal, dict):
        raise ValueError("terminal unconfirmed requests must be a map")
    terminal_ids = set(terminal)
    if terminal_ids & inflight_ids:
        raise ValueError("unconfirmed request remains both terminal and in flight")
    policy = raw_report["settings"].get("uncertain_request_policy", "strict")
    if terminal_ids and policy != "consume-reservation-and-fail-arm":
        raise ValueError("terminal unconfirmed requests require the declared failure policy")
    request_path = evidence / "requests.private.jsonl"
    request_map = {}
    if request_path.exists():
        for request in live.jsonl(request_path):
            request_id = request["request_id"]
            if request_id in request_map:
                raise ValueError("request evidence contains duplicate request IDs")
            body = request["request"]
            input_bound = len(json.dumps({"messages": body["messages"], "tools": body.get("tools", [])},
                                        ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1024
            request_map[request_id] = {"task_id": request["task_id"], "arm": request["arm"],
                                       "input_bound": input_bound, "max_output": body["max_tokens"],
                                       "sequence": len(request_map)}
    terminal_keys = set()
    for request_id, record in terminal.items():
        key = (record["task_id"], record["arm"])
        if key in terminal_keys:
            raise ValueError("terminal unconfirmed arm contains a repeated failed request")
        terminal_keys.add(key)
        row = keyed.get(key)
        request = request_map.get(request_id)
        if (key[0] not in mapping or key[1] not in arms or not request
                or (request["task_id"], request["arm"]) != key):
            raise ValueError("terminal unconfirmed request does not match its request evidence")
        if any((other["task_id"], other["arm"]) == key and other["sequence"] > request["sequence"]
               for other in request_map.values()):
            raise ValueError("failed unconfirmed arm contains a later request")
        if not row or not (row.get("missing") or row.get("failed")) or not row.get("error"):
            raise ValueError("terminal unconfirmed request requires a failed result row")
        if error_category(row["error"]) != "uncertain_metering":
            raise ValueError("terminal unconfirmed request does not match its failed result error")
        if record["arm"] + "|" + record["task_id"] not in ledger["arms"]:
            raise ValueError("terminal unconfirmed request has no matching arm counters")
        if record.get("id") != request_id or not record.get("error_class"):
            raise ValueError("terminal unconfirmed request has no matching ID or error class")
        if record["input_bound"] != request["input_bound"] or record["max_output"] != request["max_output"]:
            raise ValueError("terminal request bounds differ from its request evidence")
        context_cap = raw_report["settings"].get("endpoint_context_cap")
        if isinstance(context_cap, bool) or not isinstance(context_cap, int) or context_cap < 1:
            raise ValueError("terminal unconfirmed requests require the explicit endpoint context cap")
        if record.get("financial_input_bound") != context_cap:
            raise ValueError("terminal financial input bound differs from the endpoint context cap")
        reserved = (record["financial_input_bound"] * raw_report["settings"]["input_price"]
                    + record["max_output"] * raw_report["settings"]["output_price"]) / 1_000_000
        if not math.isclose(record["budgeted_cost_usd"], reserved, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("terminal unconfirmed request does not consume its full reservation")
    for state_key, state in ledger["arms"].items():
        records = [record for record in terminal.values()
                   if record["arm"] + "|" + record["task_id"] == state_key]
        if (state.get("unconfirmed_input_tokens", 0) != sum(record["financial_input_bound"] for record in records)
                or state.get("unconfirmed_output_tokens", 0) != sum(record["max_output"] for record in records)
                or not math.isclose(state.get("unconfirmed_cost_usd", 0),
                                    sum(record["budgeted_cost_usd"] for record in records),
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("terminal reservations differ from the arm counters")
    all_requests_uncertain = (request_count == len(inflight_ids) + len(terminal_ids)
                             and (not inflight_ids or ledger["halted"]))
    if not capture_path.exists() and request_count and not all_requests_uncertain:
        raise ValueError("model requests have no capture evidence")
    captures = live.jsonl(capture_path) if capture_path.exists() else []
    captured_ids = set()
    for capture in captures:
        key = (capture["task_id"], capture["arm"])
        if key[0] not in mapping or key[1] not in arms:
            raise ValueError("capture is outside the planned cohort")
        request_id = capture["request_id"]
        if request_id in captured_ids or request_id in inflight_ids or request_id in terminal_ids:
            raise ValueError("capture request IDs are duplicate or still uncertain")
        request = request_map.get(request_id)
        if not request or (request["task_id"], request["arm"]) != key:
            raise ValueError("capture request does not match its request evidence")
        captured_ids.add(request_id)
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
    if len(captured_ids) + len(inflight_ids) + len(terminal_ids) != request_count:
        raise ValueError("capture evidence does not account for every model request")
    if set(request_map) - captured_ids - inflight_ids - terminal_ids:
        raise ValueError("request evidence contains an unaccounted request ID")
    if inflight_ids and not ledger["halted"]:
        raise ValueError("uncertain requests have no durable stop state")
    unconfirmed_request_count = len(inflight_ids) + len(terminal_ids)

    def value(task_id, arm, metric):
        row = keyed.get((task_id, arm))
        if not row or row.get("missing") or row.get("failed"):
            return 0.0
        score = row["metrics"].get(metric)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("completed result contains an invalid metric")
        return score

    summaries = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        states = [state for key, state in ledger["arms"].items() if key.startswith(arm + "|")]
        arm_unconfirmed = sum(record["arm"] == arm for record in terminal.values()) + sum(
            record["arm"] == arm for record in ledger["inflight"].values())
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
            scores["n_planned"] = len(ids)
            groups[kind] = scores
        groups["balanced_pilot"] = {metric: nonnull_mean(groups[kind][metric]
            for kind in ("core_query", "subfield_query"))
            for metric in METRICS}
        missing = len(tasks) - sum(not row.get("missing") and not row.get("failed") for row in arm_rows)
        summaries[arm] = {"metrics": groups, "failed_or_missing_rows": missing,
            "errors": dict(Counter(error_category(row.get("error", "missing result"))
                                   for row in arm_rows if row.get("missing") or row.get("failed"))),
            "metered_cost_usd": sum(state.get("metered_cost_usd", 0) for state in states),
            "metered_cost_is_partial": bool(arm_unconfirmed),
            "metering_complete": not arm_unconfirmed,
            "unconfirmed_request_count": arm_unconfirmed,
            "budgeted_cost_usd": sum(state["actual_cost_usd"] + state.get("unconfirmed_cost_usd", 0)
                                     for state in states),
            "prompt_tokens": sum(state["prompt_tokens"] for state in states),
            "completion_tokens": sum(state["completion_tokens"] for state in states),
            "unconfirmed_input_token_bounds": sum(state.get("unconfirmed_input_tokens", 0) for state in states),
            "unconfirmed_output_token_bounds": sum(state.get("unconfirmed_output_tokens", 0) for state in states),
            "model_requests": sum(state["requests"] for state in states),
            "mean_seconds_per_planned_row": sum(row.get("elapsed_seconds", 0) for row in arm_rows) / len(tasks),
            "mean_ranked_slots": sum(len(row.get("ranking_slots", [])) for row in arm_rows) / len(tasks),
            "invalid_selection_attempts": sum(row.get("invalid_slots", 0) for row in arm_rows),
            "unjudged_final_slots": sum(slot.get("doc_id") is None and slot.get("source") != "invalid"
                for row in arm_rows for slot in row.get("ranking_slots", [])),
            "tool_payload_omitted_papers": sum(row.get("tool_payload_truncated", 0) for row in arm_rows)}
    comparisons = {}
    for left, right in (("orx_skill", "plain_tools"), ("plain_tools", "closed_book"), ("orx_skill", "closed_book")):
        if left not in arms or right not in arms:
            continue
        by_type = {}
        if grouped_sources:
            for metric in METRICS:
                source_groups = {}
                for task in tasks:
                    values = source_groups.setdefault(mapping[task["id"]]["source_id"], {})
                    values.setdefault(task["type"], []).append(
                        value(task["id"], left, metric) - value(task["id"], right, metric))
                for kind, interval in grouped_intervals(source_groups, seed=bootstrap_seed).items():
                    by_type.setdefault(kind, {})[metric] = interval
        else:
            for kind in QUESTION_TYPES:
                ids = [task["id"] for task in tasks if task["type"] == kind]
                by_type[kind] = {metric: paired_interval([
                    value(task_id, left, metric) - value(task_id, right, metric)
                    for task_id in ids], seed=bootstrap_seed) for metric in METRICS}
            by_type["balanced_pilot"] = {metric: stratified_interval([
                [value(task["id"], left, metric) - value(task["id"], right, metric)
                 for task in tasks if task["type"] == kind]
                for kind in QUESTION_TYPES], seed=bootstrap_seed) for metric in METRICS}
        comparisons[left + "_minus_" + right] = by_type
    report = {"schema_version": 2, "protocol": "restricted_live_diagnostic", "official_benchmark_result": False,
        "selected_arms": list(arms), "source_project_count": len({row["source_id"] for row in mapping.values()}),
        "model": raw_report["model"], "provider": raw_report["provider"], "settings": raw_report["settings"],
        "data_revision": raw_report["data_revision"], "verified_fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
        "cohort_sha256": manifest["tasks_sha256"], "task_count": len(tasks), "completed_rows": len(rows),
        "complete": (len(rows) == len(tasks) * len(arms) and not ledger["inflight"]
                     and not ledger["halted"] and not ledger.get("budget_exhausted", False)
                     and raw_report.get("complete", True)),
        "metered_cost_usd": sum(state.get("metered_cost_usd", 0) for state in ledger["arms"].values()),
        "metered_cost_is_partial": bool(unconfirmed_request_count),
        "metered_cost_scope": "Confirmed responses only; the total excludes unconfirmed requests.",
        "metering_complete": not unconfirmed_request_count,
        "unconfirmed_request_count": unconfirmed_request_count,
        "budgeted_cost_usd": sum(state["actual_cost_usd"] + state.get("unconfirmed_cost_usd", 0)
                                 for state in ledger["arms"].values()),
        "reserved_cost_usd": sum(state["reserved_cost_usd"] for state in ledger["arms"].values()),
        "provider_response_counts": dict(provider_counts), "arms": summaries, "paired_comparisons": comparisons,
        "label_audit": label_audit, "gold_denominator": "retain official positives, including cutoff conflicts",
        "primary_comparison": {"contrast": "orx_skill_minus_plain_tools", "metric": "recall@5",
            "statistic": "balanced_pilot", "available": "orx_skill_minus_plain_tools" in comparisons,
            "role": "single final primary comparison",
            "result": comparisons.get("orx_skill_minus_plain_tools", {}).get("balanced_pilot", {}).get("recall@5")},
        "secondary_comparisons": "all other contrasts, metrics, and question-type estimates",
        "failure_policy": (
            "An unconfirmed request ends that arm with a zero score. "
            "If no global stop occurs, other planned arms continue. The request is not retried."
            if policy == "consume-reservation-and-fail-arm" else
            "An unconfirmed request stops the run. Failed and remaining planned arms score zero."),
        "uncertainty_method": ("paired percentile bootstrap over source projects; all questions resample jointly; "
                               "equal weight for each question type; resamples with an empty type are rejected"
                               if grouped_sources else
                               "paired stratified percentile bootstrap over distinct source papers; equal weight for each question type")
                              + "; missing and failed final runs score zero",
        "candidate_method": "known-positive papers in tool payloads of confirmed model responses; includes failed final runs",
        "limits": ["Live corpus differs from the official frozen corpus.",
                   "Sparse labels do not establish relevance of unjudged papers.",
                   "The workflow uses alphaXiv discovery only and common rank validation.",
                   "Model knowledge can include the original source projects.",
                   "A pilot gives preliminary evidence."]}
    report["quality_complete"] = report["complete"]
    if unconfirmed_request_count:
        report["limits"].append("Quality estimates include zero scores for unconfirmed requests.")
    if not report["complete"]:
        report["limits"].append(
            "The experiment is incomplete; intervals do not support a quality conclusion.")
    if public_report:
        report = omit_financial_fields(report)
        report["private_accounting_fields_omitted"] = True
        report["fingerprint_scope"] = (
            "The verified hash identifies the complete private configuration. "
            "Public fingerprint inputs omit private accounting fields and cannot reproduce that hash.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--public-report", action="store_true", help="Omit all financial fields from the aggregate report")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new path")
    report = analyze(args.bench_dir, args.evidence_dir, public_report=args.public_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "task_count", "metered_cost_usd") if key in report}, indent=2))


if __name__ == "__main__":
    main()
