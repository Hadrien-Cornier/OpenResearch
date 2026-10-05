"""Mock tests for the paid live diagnostic. No network or model calls run."""

import json
import math
import contextlib
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scholarcatalyst_eval as ev
import scholarcatalyst_live as live


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_fixture(root):
    bench, out = root / "bench", root / "out"
    bench.mkdir()
    corpus = [
        {"id": "arxiv_2301.00001", "title": "Source Paper With Long Name", "published": "2023-01-01"},
        {"id": "arxiv_2202.00002", "title": "Positive Relevant Research Paper", "published": "2022-02-01"},
        {"id": "arxiv_2203.00003", "title": "Other Useful Research Paper", "published": "2022-03-01"},
        {"id": "arxiv_2402.00004", "title": "Future Paper After Cutoff", "published": "2024-02-01"},
        {"id": "local_unknown", "title": "Paper With Unknown Date", "published": ""},
    ]
    queries = [
        {"id": "qcore", "type": "core_query", "question": "How can the research question improve?",
         "paper_id": "arxiv_2301.00001", "paper_published": "2023-01-01",
         "paper_title": "Source Paper With Long Name"},
        {"id": "qsub", "type": "subfield_query", "question": "What prior work helps?",
         "paper_id": "arxiv_2301.00001", "paper_published": "2023-01-01",
         "paper_title": "Source Paper With Long Name"},
    ]
    dump_jsonl(bench / "corpus.jsonl", corpus)
    dump_jsonl(bench / "queries.jsonl", queries)
    dump_jsonl(bench / "rels/core_query.jsonl", [
        {"query_id": "qcore", "positive_docs": ["arxiv_2202.00002"], "hard_negatives": []}])
    dump_jsonl(bench / "rels/subfield_query.jsonl", [
        {"query_id": "qsub", "positive_docs": ["arxiv_2202.00002"], "hard_negatives": []}])
    queries, _ = ev.load_queries(bench / "queries.jsonl")
    corpus_map = ev.load_corpus(bench / "corpus.jsonl")
    tasks = [ev.task_record(q) for q in queries]
    out.mkdir(mode=0o700)
    ev.write_jsonl(out / "tasks.jsonl", tasks)
    manifest = {"tasks_sha256": ev.sha256(out / "tasks.jsonl"), "queries": [
        {"task_id": ev.task_record(q)["id"], "query_id": q["id"], "source_id": q["paper_id"],
         "source_title": q["paper_title"], "source_alias_ids": [q["paper_id"]],
         "type": q["type"], "cutoff": q["cutoff"]} for q in queries]}
    live.write_json(out / "manifest.private.json", manifest)
    return bench, out, tasks, corpus_map


def make_cohort_fixture(root):
    bench = root / "cohort-bench"
    bench.mkdir()
    corpus, queries = [], []
    for index in range(8):
        project = "arxiv_2301.{:05d}".format(index + 1)
        corpus.append({"id": project, "title": "Source Paper {}".format(index),
                       "published": "2023-01-01"})
        queries.append({"id": "core{}".format(index), "type": "core_query",
            "question": "Core question {}?".format(index), "paper_id": project,
            "paper_published": "2023-01-01", "paper_title": "Source Paper {}".format(index)})
        if index < 5:
            queries.append({"id": "sub{}".format(index), "type": "subfield_query",
                "question": "Subfield question {}?".format(index), "paper_id": project,
                "paper_published": "2023-01-01", "paper_title": "Source Paper {}".format(index)})
    dump_jsonl(bench / "corpus.jsonl", corpus)
    dump_jsonl(bench / "queries.jsonl", queries)
    return bench


def config(**overrides):
    value = {"model": "mock/model", "provider": "mock", "quantization": "fp8",
        "input_price": 0.15, "output_price": 0.50, "total_budget": 5.0,
        "arm_input_cap": 80_000, "arm_output_cap": 24_000,
        "max_output": 4096, "max_tool_rounds": 4, "max_discover_calls": 6,
        "reasoning_enabled": True, "reasoning_effort": "low"}
    value.update(overrides)
    return value


class ScholarCatalystLiveTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bench, self.out, self.tasks, self.corpus = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def runner(self, discover_fn=None, request_fn=None, cfg=None):
        return live.LiveRunner(self.bench, self.out, cfg or config(), discover_fn, request_fn)

    def test_arxiv_ids_normalize_versions_urls_and_prefixes(self):
        for value in ("2301.00001v3", "https://arxiv.org/abs/2301.00001v2", "arxiv_2301.00001"):
            self.assertEqual(live.canonical_arxiv(value), "arxiv_2301.00001")

    def test_candidate_filter_excludes_source_future_and_unknown_before_exposure(self):
        runner = self.runner()
        task = runner.task_map[self.tasks[0]["id"]]
        observed = {}
        counts = {"discover_calls": 0}
        records = [
            {"id": "arxiv_2301.00001v2", "title": "Source Paper With Long Name", "publicationDate": "2023-01-01"},
            {"id": "arxiv_2202.00002", "title": "Positive Relevant Research Paper", "publicationDate": "2022-02-01"},
            {"id": "arxiv_2402.00004", "title": "Future Paper After Cutoff", "publicationDate": "2024-02-01"},
            {"id": "unknown-id", "title": "Paper With Unknown Date"},
            {"id": "arxiv:2204.12345v1", "title": "Live Paper Outside Corpus", "publicationDate": "2022-04-01"},
        ]
        runner.discover_fn = lambda primitive, query, cutoff: records
        payload = runner.search("plain_tools", task, "embedding", "semantic query", observed, counts)
        text = json.dumps(payload)
        self.assertNotIn("_doc_id", text)
        self.assertNotIn("arxiv_2202.00002", text)
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(counts["excluded_source"], 1)
        self.assertEqual(counts["excluded_future"], 1)
        self.assertEqual(counts["excluded_unknown_date"], 1)
        self.assertEqual(sum(row.get("_doc_id") is None for row in observed.values()), 1)

    def test_live_future_date_wins_over_older_corpus_date(self):
        row = {"id": "arxiv_2202.00002", "title": "Positive Relevant Research Paper",
               "publicationDate": "2024-02-01"}
        safe, reason = live.normalize_candidate(row, self.corpus, runner_index(self.corpus),
            ev.title_index(self.corpus), {"source_alias_ids": [], "source_title": "", "source_id": "",
                                          "cutoff": "2023-01-01"})
        self.assertIsNone(safe)
        self.assertEqual(reason, "future")

    def test_rank_metrics_keep_holes_and_ignore_duplicate_ids(self):
        positives = {"arxiv_2202.00002"}
        slots = [{"doc_id": None}, {"doc_id": "arxiv_2202.00002"}]
        metrics = live.LiveRunner.metrics(slots, positives, set())
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertAlmostEqual(metrics["ndcg@15"], 1 / math.log2(3))
        duplicate = live.LiveRunner.metrics(
            [{"doc_id": "arxiv_2202.00002"}, {"doc_id": "arxiv_2202.00002"}], positives, set())
        self.assertEqual(duplicate["recall@5"], 1.0)
        self.assertEqual(duplicate["ndcg@15"], 1.0)

    def test_tool_arms_reject_memory_titles_closed_book_accepts_them(self):
        runner = self.runner()
        task = runner.task_map[self.tasks[0]["id"]]
        item = {"title": "Positive Relevant Research Paper", "arxiv_id": "2202.00002"}
        self.assertEqual(runner.resolve_final("closed_book", item, task, {})[0]["doc_id"],
                         "arxiv_2202.00002")
        rejected, reason = runner.resolve_final("plain_tools", item, task, {})
        self.assertIsNone(rejected["doc_id"])
        self.assertEqual(reason, "not_returned_by_tool")

    def test_budget_reservation_settlement_and_config_resume_guard(self):
        runner = self.runner()
        task_id = self.tasks[0]["id"]
        request = {"messages": [{"role": "user", "content": "question"}], "tools": []}
        request_id = runner.reserve("closed_book", task_id, request, 100)
        ledger = json.loads((self.out / "budget-ledger.private.json").read_text())
        self.assertIn(request_id, ledger["inflight"])
        settled = runner.settle(request_id, {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.000001})
        self.assertGreaterEqual(settled["budgeted_cost_usd"], settled["reported_cost_usd"])
        with self.assertRaises(live.LiveError):
            live.LiveRunner(self.bench, self.out, config(output_price=0.51))

    def test_known_over_reservation_charge_saves_response_receipt_before_stop(self):
        def overcharged(_body):
            return {"choices": [{"message": {"role": "assistant", "content": '{"papers": []}'},
                    "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 1.0}}
        runner = self.runner(request_fn=overcharged,
            cfg=config(selected_arms=["closed_book"]))
        report = runner.execute()
        captures = list(live.jsonl(self.out / "captures.private.jsonl"))
        rows = list(live.jsonl(self.out / "live-results.private.jsonl"))
        ledger = json.loads((self.out / "budget-ledger.private.json").read_text())
        self.assertEqual(len(captures), 1)
        self.assertTrue(captures[0]["usage"]["over_reservation"])
        self.assertEqual(captures[0]["usage"]["reported_cost_usd"], 1.0)
        self.assertEqual(len(rows), len(self.tasks))
        self.assertTrue(ledger["halted"])
        self.assertIn("metered cost exceeded", ledger["stop_reason"])
        self.assertFalse(report["complete"])
        self.assertGreaterEqual(ledger["arms"]["closed_book|" + self.tasks[0]["id"]]["actual_cost_usd"], 1.0)

    def test_two_inflight_reservations_obey_global_ceiling(self):
        runner = self.runner(cfg=config(total_budget=0.00025))
        body = {"messages": [{"role": "user", "content": "x"}]}
        runner.reserve("closed_book", self.tasks[0]["id"], body, 100)
        with self.assertRaises(live.LiveError):
            runner.reserve("plain_tools", self.tasks[0]["id"], body, 100)
        ledger = json.loads((self.out / "budget-ledger.private.json").read_text())
        self.assertTrue(ledger["budget_exhausted"])
        self.assertIn("hard global budget", ledger["stop_reason"])

    def test_global_budget_stop_writes_all_planned_rows_and_blocks_resume(self):
        calls = []
        def fake_request(body):
            calls.append(body)
            message = {"role": "assistant", "content": json.dumps({"papers": []})}
            encoded = json.dumps({"messages": body["messages"], "tools": body.get("tools", [])},
                ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return {"choices": [{"message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(encoded) + 1024,
                              "completion_tokens": body["max_tokens"], "cost": 0}}
        cfg = config(total_budget=0.004, max_output=500,
                     selected_arms=["plain_tools", "orx_skill"])
        runner = self.runner(request_fn=fake_request, cfg=cfg)
        report = runner.execute()
        rows = list(live.jsonl(self.out / "live-results.private.jsonl"))
        ledger = json.loads((self.out / "budget-ledger.private.json").read_text())
        self.assertEqual(len(rows), len(self.tasks) * 2)
        self.assertGreater(len(calls), 0)
        self.assertLess(len(calls), len(self.tasks) * 2)
        self.assertTrue(ledger["budget_exhausted"])
        self.assertFalse(report["complete"])
        self.assertTrue(report["budget_exhausted"])
        self.assertEqual(report["planned_rows"], len(self.tasks) * 2)
        self.assertEqual(report["completed_planned_rows"], len(self.tasks) * 2)
        self.assertTrue(report["stop_reason"])
        with self.assertRaisesRegex(live.LiveError, "do not resume or repeat"):
            runner.execute()

    def test_missing_or_nonfinite_usage_halts_and_keeps_reservation(self):
        runner = self.runner()
        request_id = runner.reserve("closed_book", self.tasks[0]["id"],
            {"messages": [{"role": "user", "content": "x"}]}, 100)
        with self.assertRaises(live.LiveError):
            runner.settle(request_id, {"prompt_tokens": 1, "completion_tokens": 1, "cost": float("nan")})
        ledger = json.loads((self.out / "budget-ledger.private.json").read_text())
        self.assertTrue(ledger["halted"])
        self.assertIn(request_id, ledger["inflight"])

    def test_strict_provider_error_log_is_private_allowlisted_and_bounded(self):
        api_key = "sk-test-private-key-93841"
        body = json.dumps({"error": {"message": "private text " + api_key,
            "code": api_key, "provider_name": api_key,
            "metadata": {"limit_source": api_key}}}).encode()
        headers = {"Retry-After": api_key, "X-Generation-Id": api_key,
            "X-Request-Id": api_key, "Authorization": api_key}
        errors = [
            urllib.error.HTTPError("https://openrouter.ai/api/v1/chat/completions", 429,
                "private reason " + api_key, headers, io.BytesIO(body)),
            TimeoutError("private timeout " + api_key),
        ]
        for index, error in enumerate(errors):
            case = self.root / ("provider-error-" + str(index))
            case.mkdir()
            bench, out, _, _ = make_fixture(case)
            def fail_request(_body, raised=error):
                raise raised
            runner = live.LiveRunner(bench, out, config(), request_fn=fail_request)
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": api_key}):
                report = runner.execute()
            log_path = out / "provider-errors.private.jsonl"
            rows = list(live.jsonl(log_path))
            payload = json.dumps(rows)
            ledger = json.loads((out / "budget-ledger.private.json").read_text())
            self.assertEqual(len(rows), 1)
            self.assertNotIn(api_key, payload)
            self.assertNotIn("private reason", payload)
            self.assertNotIn("private text", payload)
            self.assertNotIn("Authorization", payload)
            self.assertTrue(ledger["halted"])
            self.assertEqual(len(ledger["inflight"]), 1)
            self.assertFalse(report["complete"])
            if isinstance(error, urllib.error.HTTPError):
                self.assertEqual(rows[0]["error_class"], "http_429")
                self.assertEqual(rows[0]["exception_chain"][0]["http_status"], 429)

    def test_consume_policy_continues_after_429_without_retry_and_accounts_upper_bound(self):
        api_key = "sk-test-private-key-93841"
        calls = []
        error_body = json.dumps({"error": {"message": "private text " + api_key,
            "code": 429, "provider_name": "Z.AI",
            "metadata": {"limit_source": "openrouter_key_limit"}}}).encode()
        error_headers = {"Retry-After": "0", "X-Generation-Id": "gen-1727282430-aBcDeFgHiJkLmNoPqRsT",
            "X-Request-Id": "req-1727282430-aBcDeFgHiJkLmNoPqRsT", "Authorization": api_key}
        def request(body):
            calls.append(body)
            if len(calls) == 2:
                raise urllib.error.HTTPError("https://openrouter.ai/api/v1/chat/completions",
                    429, "private reason " + api_key, error_headers, io.BytesIO(error_body))
            return {"choices": [{"message": {"role": "assistant", "content": '{"papers": []}'},
                    "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}
        cfg = config(total_budget=0.02, max_output=500, arm_output_cap=500,
            selected_arms=["closed_book", "plain_tools"],
            uncertain_request_policy=live.CONSUME_UNCERTAIN_POLICY, endpoint_context_cap=10_000)
        live.validate_config(cfg)
        runner = self.runner(request_fn=request, cfg=cfg)
        delays = []
        runner.wait_fn = delays.append
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": api_key}):
            report = runner.execute()
        rows = list(live.jsonl(self.out / "live-results.private.jsonl"))
        captures = list(live.jsonl(self.out / "captures.private.jsonl"))
        error_rows = list(live.jsonl(self.out / "provider-errors.private.jsonl"))
        ledger = json.loads((self.out / "budget-ledger.private.json").read_text())
        terminal = list(ledger["terminal_unconfirmed"].values())
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len({row["request_id"] for row in captures + error_rows}), 4)
        self.assertEqual(len(captures), 3)
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["error_class"], "http_429")
        self.assertEqual(terminal[0]["financial_input_bound"], 10_000)
        self.assertEqual(terminal[0]["budgeted_cost_usd"],
            (10_000 * cfg["input_price"] + terminal[0]["max_output"] * cfg["output_price"]) / 1_000_000)
        self.assertEqual(ledger["arms"]["plain_tools|" + self.tasks[0]["id"]]["unconfirmed_output_tokens"], 500)
        with self.assertRaisesRegex(live.LiveError, "output-token cap"):
            runner.reserve("plain_tools", self.tasks[0]["id"],
                {"messages": [{"role": "user", "content": "next request"}]}, 1)
        self.assertTrue(report["complete"])
        self.assertFalse(report["metering_complete"])
        self.assertEqual(report["unconfirmed_request_count"], 1)
        committed = sum(state["actual_cost_usd"] + state["reserved_cost_usd"] +
                        state["unconfirmed_cost_usd"] for state in ledger["arms"].values())
        self.assertLessEqual(committed, cfg["total_budget"])
        observed_charge = sum(state["actual_cost_usd"] for state in ledger["arms"].values())
        self.assertAlmostEqual(observed_charge,
            3 * (cfg["input_price"] + cfg["output_price"]) / 1_000_000)
        safe_error = json.dumps(error_rows)
        self.assertNotIn(api_key, safe_error)
        self.assertNotIn("private reason", safe_error)
        self.assertEqual(error_rows[0]["exception_chain"][0]["response_headers"]["Retry-After"], "0")
        self.assertEqual(error_rows[0]["exception_chain"][0]["response_headers"]["X-Generation-Id"],
                         "gen-1727282430-aBcDeFgHiJkLmNoPqRsT")
        self.assertEqual(error_rows[0]["exception_chain"][0]["provider_error"],
                         {"code": 429, "provider_name": "Z.AI", "limit_source": "openrouter_key_limit"})
        self.assertEqual(delays, [])

    def test_consume_policy_stops_on_auth_failure_and_excessive_cooldown(self):
        cases = ((401, "0", "provider response requires a global stop"),
                 (429, "301", "provider cooldown exceeds the permitted 300 seconds"))
        for status, retry_after, expected_reason in cases:
            case = self.root / ("global-stop-" + str(status) + "-" + retry_after)
            case.mkdir()
            bench, out, _, _ = make_fixture(case)
            calls = []
            def fail_request(body):
                calls.append(body)
                raise urllib.error.HTTPError("https://openrouter.ai/api/v1/chat/completions",
                    status, "private response text", {"Retry-After": retry_after}, io.BytesIO(b"{}"))
            cfg = config(selected_arms=["closed_book", "plain_tools"], max_output=100,
                uncertain_request_policy=live.CONSUME_UNCERTAIN_POLICY, endpoint_context_cap=10_000)
            runner = live.LiveRunner(bench, out, cfg, request_fn=fail_request)
            delays = []
            runner.wait_fn = delays.append
            report = runner.execute()
            rows = list(live.jsonl(out / "live-results.private.jsonl"))
            ledger = json.loads((out / "budget-ledger.private.json").read_text())
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(rows), len(self.tasks) * 2)
            self.assertTrue(ledger["halted"])
            self.assertEqual(ledger["stop_reason"], expected_reason)
            self.assertEqual(len(ledger["terminal_unconfirmed"]), 1)
            self.assertFalse(report["complete"])
            self.assertEqual(delays, [])

    def test_orx_skill_fallback_preserves_unjudged_observation_slot(self):
        calls = {"request": 0}
        records = [
            {"id": "arxiv_2202.00002", "title": "Positive Relevant Research Paper", "publicationDate": "2022-02-01"},
            {"id": "arxiv_2204.12345", "title": "Unjudged Live Paper", "publicationDate": "2022-04-01"},
        ]
        def fake_request(body):
            calls["request"] += 1
            if calls["request"] == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1",
                    "type": "function", "function": {"name": "orx_discover_embedding",
                    "arguments": json.dumps({"query": "question"})}}]}
            else:
                # Invalid source selection triggers the skill's observation-order fallback.
                message = {"role": "assistant", "content": json.dumps({"papers": [
                    {"title": "Source Paper With Long Name", "arxiv_id": "2301.00001"}]})}
            return {"choices": [{"message": message, "finish_reason": "tool_calls" if calls["request"] == 1 else "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}
        runner = self.runner(lambda *args: records, fake_request)
        row = runner.run_task_arm("orx_skill", self.tasks[0])
        self.assertEqual([slot["doc_id"] for slot in row["ranking_slots"]],
                         ["arxiv_2202.00002", None])
        self.assertEqual(row["metrics"]["recall@5"], 1.0)
        capture = (self.out / "captures.private.jsonl").read_text()
        self.assertNotIn("arxiv_2202.00002", capture.splitlines()[0])

    def test_mock_execute_writes_all_six_fixed_cohort_rows(self):
        request_bodies = []
        def fake_request(body):
            request_bodies.append(body)
            message = {"role": "assistant", "content": json.dumps({"papers": [
                {"title": "Positive Relevant Research Paper", "arxiv_id": "2202.00002"}]})}
            return {"choices": [{"message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}
        runner = self.runner(request_fn=fake_request)
        report = runner.execute()
        rows = list(live.jsonl(self.out / "live-results.private.jsonl"))
        self.assertEqual(len(rows), 6)
        self.assertEqual(report["completed_rows"], 6)
        self.assertEqual(report["arms"]["closed_book"]["metrics"]["core_query"]["n_queries"], 1)
        self.assertEqual(report["arms"]["closed_book"]["metrics"]["core_query"]["recall@5"], 1.0)
        self.assertEqual(report["arms"]["plain_tools"]["metrics"]["core_query"]["recall@5"], 0.0)
        self.assertTrue(all(body["provider"]["only"] == ["mock"] for body in request_bodies))
        self.assertTrue(all(body["provider"]["allow_fallbacks"] is False for body in request_bodies))
        self.assertTrue(all("max_price" in body["provider"] and "max_price" not in body for body in request_bodies))

    def test_cohort_counts_exclude_prior_source_projects_and_record_actuals(self):
        bench = make_cohort_fixture(self.root)
        previous = self.root / "previous"
        live.make_cohort(bench, previous, seed=531, core_count=1, subfield_count=1,
                         cohort_mode="paired-projects")
        out = self.root / "expanded"
        result = live.make_cohort(bench, out, seed=532, core_count=2, subfield_count=3,
                                  exclude_manifest=previous / "manifest.private.json")
        manifest = json.loads((out / "manifest.private.json").read_text())
        selected_projects = {row["source_project_id"] for row in manifest["queries"]}
        excluded = set(manifest["excluded_source_projects"])
        self.assertEqual(result["task_count"], 5)
        self.assertEqual(result["core_query"], 2)
        self.assertEqual(result["subfield_query"], 3)
        self.assertEqual(manifest["actual_counts"], {"core_query": 2, "subfield_query": 3})
        self.assertEqual(len(selected_projects), 5)
        self.assertFalse(selected_projects & excluded)
        self.assertEqual(manifest["seed"], 532)

    def test_cohort_modes_and_insufficient_source_disjoint_capacity(self):
        bench = make_cohort_fixture(self.root)
        paired = self.root / "paired"
        result = live.make_cohort(bench, paired, seed=1, core_count=2, subfield_count=2,
                                  cohort_mode="paired-projects")
        manifest = json.loads((paired / "manifest.private.json").read_text())
        self.assertEqual(result["source_projects"], 2)
        self.assertEqual(result["task_count"], 4)
        self.assertEqual(len({row["source_project_id"] for row in manifest["queries"]}), 2)
        with self.assertRaisesRegex(live.LiveError, "distinct source projects"):
            live.make_cohort(bench, self.root / "too-large", seed=1, core_count=6,
                             subfield_count=3)

    def test_all_project_questions_uses_every_remaining_query_and_checks_counts(self):
        bench = make_cohort_fixture(self.root)
        out = self.root / "all"
        result = live.make_cohort(bench, out, seed=1, cohort_mode="all-project-questions")
        manifest = json.loads((out / "manifest.private.json").read_text())
        self.assertEqual(result["task_count"], 13)
        self.assertIsNone(manifest["requested_counts"])
        with self.assertRaisesRegex(live.LiveError, "does not accept custom counts"):
            live.make_cohort(bench, self.root / "all-custom", seed=1, core_count=10,
                             cohort_mode="all-project-questions")

    def test_exclusion_manifest_must_match_benchmark_hashes(self):
        bench = make_cohort_fixture(self.root)
        previous = self.root / "previous-valid"
        live.make_cohort(bench, previous, seed=1, core_count=1, subfield_count=1,
                         cohort_mode="paired-projects")
        (bench / "queries.jsonl").write_text((bench / "queries.jsonl").read_text() + "\n")
        with self.assertRaisesRegex(live.LiveError, "data hashes"):
            live.make_cohort(bench, self.root / "excluded", seed=1, core_count=1,
                subfield_count=1, exclude_manifest=previous / "manifest.private.json")

    def test_exclusion_manifest_rejects_unknown_source_projects(self):
        bench = make_cohort_fixture(self.root)
        previous = self.root / "previous-project"
        live.make_cohort(bench, previous, seed=1, core_count=1, subfield_count=1,
                         cohort_mode="paired-projects")
        manifest_path = previous / "manifest.private.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["queries"][0]["source_project_id"] = "unknown-project"
        live.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(live.LiveError, "unknown or mismatched"):
            live.make_cohort(bench, self.root / "excluded-project", seed=1, core_count=1,
                subfield_count=1, exclude_manifest=manifest_path)

    def test_all_project_questions_rejects_an_empty_remaining_cohort(self):
        bench = make_cohort_fixture(self.root)
        previous = self.root / "all-previous"
        live.make_cohort(bench, previous, seed=1, cohort_mode="all-project-questions")
        with self.assertRaisesRegex(live.LiveError, "no queries after"):
            live.make_cohort(bench, self.root / "all-excluded", seed=1,
                cohort_mode="all-project-questions", exclude_manifest=previous / "manifest.private.json")

    def test_custom_arm_selection_and_larger_hard_budget_are_valid(self):
        args = live.parser().parse_args(["run", "--bench-dir", str(self.bench), "--output-dir",
            str(self.out), "--arms", "plain_tools", "orx_skill", "--total-budget", "12"])
        cfg = live.config_from_args(args)
        live.validate_config(cfg)
        self.assertEqual(cfg["selected_arms"], ["plain_tools", "orx_skill"])
        default_args = live.parser().parse_args(["run", "--bench-dir", str(self.bench),
            "--output-dir", str(self.out)])
        self.assertNotIn("selected_arms", live.config_from_args(default_args))

    def test_dry_run_reports_requested_hard_budget_and_selected_arms(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = live.main(["run", "--bench-dir", str(self.bench), "--output-dir",
                str(self.out), "--total-budget", "12", "--arms", "plain_tools", "orx_skill"])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["hard_budget_usd"], 12.0)
        self.assertEqual(report["arms"], 2)
        self.assertEqual(report["selected_arms"], ["plain_tools", "orx_skill"])

    def test_worker_limit_allows_sixteen_and_rejects_higher_values(self):
        live.validate_config(config(workers=16))
        with self.assertRaisesRegex(live.LiveError, "one and sixteen"):
            live.validate_config(config(workers=17))

    def test_selected_arms_bound_execution_and_report(self):
        def fake_request(body):
            message = {"role": "assistant", "content": json.dumps({"papers": [
                {"title": "Positive Relevant Research Paper", "arxiv_id": "2202.00002"}]})}
            return {"choices": [{"message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}
        selected = ["plain_tools", "orx_skill"]
        runner = self.runner(request_fn=fake_request, cfg=config(selected_arms=selected))
        report = runner.execute()
        rows = list(live.jsonl(self.out / "live-results.private.jsonl"))
        self.assertEqual(len(rows), 4)
        self.assertEqual(set(report["arms"]), set(selected))
        self.assertEqual(report["settings"]["selected_arms"], selected)
        self.assertEqual(report["selected_arms"], selected)

    def test_parallel_queries_preserve_metering_cache_and_complete_rows(self):
        guard = threading.Lock()
        first_wave = threading.Barrier(4)
        active, max_active = 0, 0

        def fake_request(body):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            if not body.get("tools") and "concurrent_" in body["messages"][1]["content"]:
                first_wave.wait(timeout=10)
            time.sleep(0.01)
            if body.get("tools") and not any(message["role"] == "tool" for message in body["messages"]):
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1",
                    "type": "function", "function": {"name": "orx_discover_embedding",
                    "arguments": json.dumps({"query": "shared query"})}}]}
            else:
                message = {"role": "assistant", "content": json.dumps({"papers": [
                    {"title": "Positive Relevant Research Paper", "arxiv_id": "2202.00002"}]})}
            with guard:
                active -= 1
            return {"choices": [{"message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}

        records = [{"id": "2202.00002", "title": "Positive Relevant Research Paper",
                    "publicationDate": "2022-02-01"}]
        runner = self.runner(lambda *args: records, fake_request, config(workers=4))
        for index in range(4):
            task = {**runner.tasks[0], "id": "extra_" + str(index)}
            task["question"] += " concurrent_" + str(index)
            runner.tasks.append(task)
            runner.task_map[task["id"]] = dict(runner.task_map[self.tasks[0]["id"]])
            runner.labels[task["id"]] = set(runner.labels[self.tasks[0]["id"]])
        report = runner.execute()
        rows = list(live.jsonl(self.out / "live-results.private.jsonl"))
        captures = list(live.jsonl(self.out / "captures.private.jsonl"))
        self.assertGreaterEqual(max_active, 2)
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({(row["task_id"], row["arm"]) for row in rows}), 18)
        self.assertEqual(len(captures), 30)
        self.assertEqual(runner.ledger["inflight"], {})
        self.assertEqual(len(json.loads(runner.cache_path.read_text())), 1)
        self.assertAlmostEqual(sum(arm["cost_usd"] for arm in report["arms"].values()), 30 * 0.65 / 1_000_000)
        self.assertTrue(all(arm["successful_rows"] == 6 for arm in report["arms"].values()))

    def test_empty_results_score_zero_on_the_full_planned_cohort(self):
        report = self.runner().make_report([])
        for arm in report["arms"].values():
            for group in arm["metrics"].values():
                self.assertEqual(group["n_queries"], 1)
                self.assertEqual(group["missing_rows"], 1)
                self.assertEqual(group["recall@5"], 0)

    def test_malformed_tool_arguments_fail_each_arm_and_preserve_report(self):
        def fake_request(body):
            if body.get("tools"):
                message = {"role": "assistant", "tool_calls": [{"id": "bad", "function": {
                    "name": "orx_discover_embedding", "arguments": "[]"}}]}
            else:
                message = {"role": "assistant", "content": '{"papers":[]}'}
            return {"choices": [{"message": message}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}
        runner = self.runner(request_fn=fake_request)
        report = runner.execute()
        self.assertEqual(report["completed_rows"], 6)
        self.assertEqual(report["arms"]["plain_tools"]["missing_rows"], 2)
        self.assertEqual(report["arms"]["orx_skill"]["missing_rows"], 2)
        self.assertEqual(runner.ledger["inflight"], {})


def runner_index(corpus):
    out = {}
    for doc_id in corpus:
        canon = live.canonical_arxiv(doc_id)
        if canon:
            out.setdefault(canon, []).append(doc_id)
    return out


if __name__ == "__main__":
    unittest.main()
