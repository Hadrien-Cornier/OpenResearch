"""Mock tests for the paid live diagnostic. No network or model calls run."""

import json
import math
import sys
import tempfile
import threading
import time
import unittest
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

    def test_two_inflight_reservations_obey_global_ceiling(self):
        runner = self.runner(cfg=config(total_budget=0.00025))
        body = {"messages": [{"role": "user", "content": "x"}]}
        runner.reserve("closed_book", self.tasks[0]["id"], body, 100)
        with self.assertRaises(live.LiveError):
            runner.reserve("plain_tools", self.tasks[0]["id"], body, 100)

    def test_missing_or_nonfinite_usage_halts_and_keeps_reservation(self):
        runner = self.runner()
        request_id = runner.reserve("closed_book", self.tasks[0]["id"],
            {"messages": [{"role": "user", "content": "x"}]}, 100)
        with self.assertRaises(live.LiveError):
            runner.settle(request_id, {"prompt_tokens": 1, "completion_tokens": 1, "cost": float("nan")})
        ledger = json.loads((self.out / "budget-ledger.private.json").read_text())
        self.assertTrue(ledger["halted"])
        self.assertIn(request_id, ledger["inflight"])

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

    def test_parallel_queries_preserve_metering_cache_and_complete_rows(self):
        guard = threading.Lock()
        active, max_active = 0, 0

        def fake_request(body):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
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
