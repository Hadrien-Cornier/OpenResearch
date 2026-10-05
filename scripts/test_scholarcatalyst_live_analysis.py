"""Check paired analysis and aggregate privacy with a complete mock run."""

import json
from pathlib import Path
import tempfile
import unittest

import scholarcatalyst_live as live
import scholarcatalyst_live_analysis as analysis
import scholarcatalyst_eval as ev
from test_scholarcatalyst_live import make_fixture, config


class AnalysisTest(unittest.TestCase):
    @staticmethod
    def mock_request(body):
        return {"provider": "mock", "choices": [{"message": {"role": "assistant",
            "content": '{"papers":[{"arxiv_id":"2202.00002"}]}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}

    def test_project_bootstrap_preserves_cross_type_dependence(self):
        groups = {str(index): {"core_query": [effect], "subfield_query": [-effect] * 3}
                  for index, effect in enumerate([-1.0, -0.5, 0.5, 1.0])}
        intervals = analysis.grouped_intervals(groups, samples=1000, seed=531)
        balanced = intervals["balanced_pilot"]
        self.assertEqual(balanced["interval_95"], [0, 0])
        self.assertEqual(balanced["n_pairs"], 16)
        self.assertEqual(balanced["n_source_projects"], 4)
        self.assertLess(intervals["core_query"]["interval_95"][0], 0)
        self.assertGreater(intervals["core_query"]["interval_95"][1], 0)
        independent = analysis.stratified_interval(
            [[value for group in groups.values() for value in group[kind]]
             for kind in analysis.QUESTION_TYPES], samples=1000, seed=531)
        self.assertLess(independent["interval_95"][0], 0)
        self.assertGreater(independent["interval_95"][1], 0)

    def test_balanced_statistic_does_not_weight_types_by_question_count(self):
        intervals = analysis.grouped_intervals({
            "a": {"core_query": [1], "subfield_query": [-1] * 5},
            "b": {"core_query": [1], "subfield_query": [-1] * 2}}, samples=100)
        self.assertEqual(intervals["balanced_pilot"]["mean_delta"], 0)
        self.assertEqual(intervals["balanced_pilot"]["interval_95"], [0, 0])

    def test_two_arm_grouped_run_retains_missing_and_failed_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            bench, out, tasks, _ = make_fixture(Path(directory))
            manifest_path = out / "manifest.private.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["seed"] = 531
            live.write_json(manifest_path, manifest)
            live.LiveRunner(bench, out, config(selected_arms=["plain_tools", "orx_skill"]),
                            request_fn=self.mock_request).execute()
            result_path = out / "live-results.private.jsonl"
            rows = list(live.jsonl(result_path))
            self.assertEqual(len(rows), 4)
            for row in rows:
                row["metrics"] = {metric: float(row["arm"] == "orx_skill") for metric in analysis.METRICS}
            ev.write_jsonl(result_path, rows)
            report = analysis.analyze(bench, out)
            self.assertTrue(report["complete"])
            self.assertEqual(set(report["arms"]), {"plain_tools", "orx_skill"})
            self.assertEqual(set(report["paired_comparisons"]), {"orx_skill_minus_plain_tools"})
            self.assertEqual(report["primary_comparison"]["result"]["mean_delta"], 1)
            self.assertEqual(report["primary_comparison"]["result"]["seed"], 531)
            self.assertEqual(report["source_project_count"], 1)
            rows = [row for row in rows if not (row["arm"] == "orx_skill" and row["task_id"] == tasks[0]["id"])]
            ev.write_jsonl(result_path, rows)
            missing = analysis.analyze(bench, out)
            self.assertFalse(missing["complete"])
            self.assertEqual(missing["primary_comparison"]["result"]["mean_delta"], 0.5)
            self.assertEqual(missing["primary_comparison"]["result"]["n_pairs"], 2)
            for row in rows:
                if row["arm"] == "orx_skill":
                    row["missing"] = True
            ev.write_jsonl(result_path, rows)
            failed = analysis.analyze(bench, out)
            self.assertEqual(failed["primary_comparison"]["result"]["mean_delta"], 0)
            self.assertEqual(failed["arms"]["orx_skill"]["failed_or_missing_rows"], 2)

    def test_public_report_omits_financial_fields_after_fingerprint_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            bench, out, tasks, _ = make_fixture(Path(directory))
            live.LiveRunner(bench, out, config(), request_fn=self.mock_request).execute()
            private = analysis.analyze(bench, out)
            public = analysis.analyze(bench, out, public_report=True)

            def check_keys(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        self.assertFalse(any(term in key.lower() for term in ("price", "budget", "charge", "cost", "usd")))
                        check_keys(item)
                elif isinstance(value, list):
                    for item in value:
                        check_keys(item)

            check_keys(public)
            self.assertIn("input_price", private["settings"])
            self.assertIn("input_price", private["fingerprint_inputs"])
            self.assertEqual(public["verified_fingerprint"], private["verified_fingerprint"])
            self.assertIn("complete private configuration", public["fingerprint_scope"])
            text = json.dumps(public)
            self.assertNotIn(tasks[0]["question"], text)
            self.assertNotIn(tasks[0]["id"], text)
            self.assertNotIn("arxiv_2301.00001", text)
            self.assertEqual(analysis.omit_financial_fields({"nested": [{"charge": 2, "tokens": 3}],
                "unit_price": 1}), {"nested": [{"tokens": 3}]})
            path = out / "live-report.private.json"
            altered = json.loads(path.read_text())
            altered["settings"]["input_price"] = 999
            live.write_json(path, altered)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                analysis.analyze(bench, out, public_report=True)
            altered["settings"] = private["settings"]
            live.write_json(path, altered)
            (out / "captures.private.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "no capture evidence"):
                analysis.analyze(bench, out, public_report=True)

    def test_paired_intervals_preserve_the_effect_and_pair_count(self):
        interval = analysis.paired_interval([1.0] * 25, samples=100)
        self.assertEqual(interval["mean_delta"], 1)
        self.assertEqual(interval["interval_95"], [1, 1])
        self.assertEqual(interval["n_pairs"], 25)
        self.assertIsNone(analysis.paired_interval([])["mean_delta"])
        balanced = analysis.stratified_interval([[-1.0] * 25, [1.0] * 25], samples=100)
        self.assertEqual(balanced["mean_delta"], 0)
        self.assertEqual(balanced["interval_95"], [0, 0])
        self.assertEqual(balanced["n_pairs"], 50)

    def test_complete_mock_run_exports_no_questions_or_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            bench, out, tasks, _ = make_fixture(Path(directory))
            manifest_path = out / "manifest.private.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["queries"][1]["source_id"] = "arxiv_2301.00009"
            live.write_json(manifest_path, manifest)

            def request(body):
                return {"provider": "mock", "choices": [{"message": {"role": "assistant",
                    "content": '{"papers":[{"arxiv_id":"2202.00002"}]}'}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}

            runner = live.LiveRunner(bench, out, config(workers=4), request_fn=request)
            runner.execute()
            report = analysis.analyze(bench, out)
            self.assertTrue(report["complete"])
            self.assertEqual(report["completed_rows"], 6)
            self.assertEqual(report["provider_response_counts"], {"mock": 6})
            delta = report["paired_comparisons"]["plain_tools_minus_closed_book"]["core_query"]["recall@5"]
            self.assertEqual(delta["mean_delta"], -1)
            self.assertEqual(delta["interval_95"], [-1, -1])
            text = json.dumps(report)
            self.assertNotIn("arxiv_2301.00009", text)
            self.assertNotIn(tasks[0]["question"], text)
            self.assertEqual(report["metered_cost_usd"], 0)
            self.assertGreater(report["budgeted_cost_usd"], 0)
            relation = bench / "rels/core_query.jsonl"
            relation.write_text(relation.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                analysis.analyze(bench, out)

    def test_public_errors_use_fixed_categories(self):
        error = "orx discover failed for secret unpublished research question"
        self.assertEqual(analysis.error_category(error), "search_error")
        self.assertEqual(analysis.error_category("secret unknown data path"), "other_run_error")

    def test_global_limit_keeps_planned_zero_rows_but_run_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            bench, out, tasks, _ = make_fixture(Path(directory))

            def request(body):
                self.fail("the exhausted run must not send a model request")

            settings = config(total_budget=0.000001, selected_arms=["plain_tools", "orx_skill"])
            live.validate_config(settings)
            live.LiveRunner(bench, out, settings, request_fn=request).execute()
            ledger = json.loads((out / "budget-ledger.private.json").read_text())
            self.assertTrue(ledger["budget_exhausted"])
            self.assertFalse(ledger["halted"])
            self.assertFalse(ledger["inflight"])
            self.assertFalse((out / "captures.private.jsonl").exists())
            report = analysis.analyze(bench, out)
            self.assertEqual(report["completed_rows"], len(tasks) * 2)
            self.assertFalse(report["complete"])
            self.assertEqual(report["primary_comparison"]["result"]["n_pairs"], len(tasks))
            self.assertEqual(report["primary_comparison"]["result"]["mean_delta"], 0)
            for arm in report["arms"].values():
                self.assertEqual(arm["failed_or_missing_rows"], len(tasks))
                self.assertEqual(arm["model_requests"], 0)
            self.assertFalse(analysis.analyze(bench, out, public_report=True)["complete"])
            self.assertIn("The experiment is incomplete; intervals do not support a quality conclusion.",
                          report["limits"])

    def test_uncertain_first_request_exports_incomplete_aggregate_without_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            bench, out, tasks, _ = make_fixture(Path(directory))
            attempts = []

            def request(body):
                attempts.append(body)
                raise TimeoutError("private unpublished question")

            live.LiveRunner(bench, out, config(selected_arms=["plain_tools", "orx_skill"]),
                            request_fn=request).execute()
            ledger = json.loads((out / "budget-ledger.private.json").read_text())
            self.assertEqual(len(attempts), 1)
            self.assertTrue(ledger["halted"])
            self.assertEqual(len(ledger["inflight"]), 1)
            self.assertEqual(sum(state["requests"] for state in ledger["arms"].values()), 1)
            self.assertFalse((out / "captures.private.jsonl").exists())
            report = analysis.analyze(bench, out, public_report=True)
            self.assertFalse(report["complete"])
            self.assertEqual(report["completed_rows"], len(tasks) * 2)
            self.assertEqual(report["provider_response_counts"], {})
            self.assertEqual(report["primary_comparison"]["result"]["n_pairs"], len(tasks))
            self.assertIn("The experiment is incomplete; intervals do not support a quality conclusion.",
                          report["limits"])
            self.assertNotIn("private unpublished question", json.dumps(report))
            self.assertNotIn(tasks[0]["question"], json.dumps(report))

    def test_bounded_policy_completes_quality_without_inventing_observed_usage(self):
        for failure in ("transport", "missing_usage"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                bench, out, tasks, _ = make_fixture(Path(directory))
                attempts = []

                def request(body):
                    attempts.append(body)
                    if len(attempts) == 1:
                        if failure == "transport":
                            raise TimeoutError("private question")
                        response = self.mock_request(body)
                        response["usage"] = {}
                        return response
                    if any(message.get("role") == "tool" for message in body["messages"]):
                        return self.mock_request(body)
                    return {"provider": "mock", "choices": [{"message": {"role": "assistant",
                        "content": None, "tool_calls": [{"id": "lookup", "type": "function",
                        "function": {"name": "orx_discover_embedding",
                                     "arguments": '{"query":"prior work"}'}}]}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}

                records = [{"id": "arxiv_2202.00002", "title": "Positive Relevant Research Paper",
                            "publicationDate": "2022-02-01"}]
                settings = config(workers=1, selected_arms=["plain_tools", "orx_skill"],
                    uncertain_request_policy="consume-reservation-and-fail-arm", endpoint_context_cap=1048576)
                live.validate_config(settings)
                live.LiveRunner(bench, out, settings, discover_fn=lambda *args: records,
                                request_fn=request).execute()
                ledger = json.loads((out / "budget-ledger.private.json").read_text())
                self.assertFalse(ledger["halted"])
                self.assertFalse(ledger["inflight"])
                self.assertEqual(len(ledger["terminal_unconfirmed"]), 1)
                self.assertEqual(len(attempts), 7)
                rows = list(live.jsonl(out / "live-results.private.jsonl"))
                failed = [row for row in rows if row.get("missing")]
                self.assertEqual(len(failed), 1)
                self.assertEqual(failed[0]["arm"], "plain_tools")
                self.assertEqual(failed[0]["task_id"], tasks[0]["id"])
                self.assertEqual(failed[0]["metrics"]["recall@5"], 0)
                self.assertTrue(all(row["metrics"]["recall@5"] == 1 for row in rows if not row.get("missing")))
                report = analysis.analyze(bench, out)
                self.assertTrue(report["complete"])
                self.assertTrue(report["quality_complete"])
                self.assertFalse(report["metering_complete"])
                self.assertEqual(report["unconfirmed_request_count"], 1)
                self.assertEqual(report["provider_response_counts"], {"mock": 6})
                self.assertEqual(report["metered_cost_usd"], 0)
                self.assertTrue(report["metered_cost_is_partial"])
                self.assertIn("Confirmed responses only", report["metered_cost_scope"])
                self.assertEqual(sum(arm["prompt_tokens"] for arm in report["arms"].values()), 6)
                self.assertEqual(sum(arm["completion_tokens"] for arm in report["arms"].values()), 6)
                terminal = next(iter(ledger["terminal_unconfirmed"].values()))
                self.assertGreater(report["budgeted_cost_usd"], terminal["budgeted_cost_usd"])
                self.assertEqual(report["arms"]["plain_tools"]["unconfirmed_input_token_bounds"], terminal["financial_input_bound"])
                self.assertEqual(report["primary_comparison"]["result"]["mean_delta"], 0.5)
                public = analysis.analyze(bench, out, public_report=True)
                self.assertTrue(public["quality_complete"])
                self.assertFalse(public["metering_complete"])
                self.assertIn("The request is not retried", public["failure_policy"])
                self.assertNotIn("metered_cost_usd", public)
                self.assertNotIn(tasks[0]["question"], json.dumps(public))
                terminal["financial_input_bound"] -= 1
                live.write_json(out / "budget-ledger.private.json", ledger)
                with self.assertRaisesRegex(ValueError, "financial input bound"):
                    analysis.analyze(bench, out)
                terminal["financial_input_bound"] += 1
                live.write_json(out / "budget-ledger.private.json", ledger)
                capture_path = out / "captures.private.jsonl"
                captures = list(live.jsonl(capture_path))
                ev.write_jsonl(capture_path, captures[1:])
                with self.assertRaisesRegex(ValueError, "capture evidence"):
                    analysis.analyze(bench, out)


if __name__ == "__main__":
    unittest.main()
