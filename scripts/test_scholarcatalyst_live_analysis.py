"""Check paired analysis and aggregate privacy with a complete mock run."""

import json
from pathlib import Path
import tempfile
import unittest

import scholarcatalyst_live as live
import scholarcatalyst_live_analysis as analysis
from test_scholarcatalyst_live import make_fixture, config


class AnalysisTest(unittest.TestCase):
    def test_paired_intervals_preserve_the_effect_and_pair_count(self):
        interval = analysis.paired_interval([1.0] * 25, samples=100)
        self.assertEqual(interval["mean_delta"], 1)
        self.assertEqual(interval["interval_95"], [1, 1])
        self.assertEqual(interval["n_pairs"], 25)
        self.assertIsNone(analysis.paired_interval([])["mean_delta"])

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


if __name__ == "__main__":
    unittest.main()
