# ScholarCatalyst live diagnostic

This experiment compares a model without tools, a model with ordinary literature tools, and the same model with the OpenResearch literature skill. It uses actual `orx discover` requests. It measures a restricted literature workflow through the CLI.

Use the offline evaluator for results on the official frozen corpus. Live alphaXiv search returns papers outside that corpus. A live diagnostic measures coverage of known positive papers. An unmatched paper receives zero credit, but that score does not establish that the paper is irrelevant.

## Controls

The pilot selects 50 questions before any model call. It includes 25 core questions and 25 subfield questions. It selects one question per source paper. The task files use opaque identifiers. The model receives research questions and publication cutoffs. Source identifiers, source titles, benchmark labels, and corpus membership remain private.

All three arms use the same model, provider, temperature, and reasoning setting. Both tool arms use the same alphaXiv keyword and embedding tools. Each tool arm permits six searches and four model turns with tools, followed by one final turn. Each final list contains at most 15 papers.

The source paper and papers after the cutoff are removed before exposure. Unknown dates cause exclusion. Tool results preserve live identifiers. Abstracts and snippets have fixed size limits. Only papers in the actual tool payload count as exposed candidates.

Unmatched and invalid selections retain their ranking positions. Duplicate positive papers receive credit once. Missing and failed runs receive zero credit in the planned cohort. The report separates the two question types.

This common scoring rule changes the skill's native final step, which removes unsupported selections. The replay retains those positions in both tool arms. The skill's fallback applies only when no supported selection survives.

## Model choice and request limits

Consult current Artificial Analysis results when you select an open model. Verify the exact model version and endpoint before a run. A general model score does not predict performance on this experiment.

The October 4 pilot selects `z-ai/glm-5.3-flash` through the first-party Z.AI FP8 endpoint. It uses temperature zero and reasoning effort `low`. Provider fallback is disabled.

The runner enforces a configured global request limit across concurrent calls. Missing or uncertain metering stops the run and preserves the reservation. The runner does not repeat an uncertain request.

Each task-arm has a cumulative cap of 80,000 input tokens and 24,000 output tokens. Each request permits 4,096 output tokens, including reasoning tokens. Input reservations use a conservative byte bound. The runner can stop a task before its actual token cap if that bound is too large.

The first pilot stops 34 skill runs at that conservative input bound. One additional skill run has a search failure. The repeat after this correction uses `--arm-input-cap 320000` in all three arms. The byte reservation stays active. Preserve the first run and its failures as a separate experiment.

## Run the diagnostic

Set `OPENROUTER_API_KEY` in the process environment. Keep its value outside logs and files.

Prepare the cohort before the model calls:

```sh
python3 scripts/scholarcatalyst_live.py prepare \
  --bench-dir /path/to/benchmark --output-dir /path/to/private-evidence
```

Inspect the configuration without a paid call:

```sh
python3 scripts/scholarcatalyst_live.py run \
  --bench-dir /path/to/benchmark --output-dir /path/to/private-evidence \
  --arm-input-cap 320000
```

Add `--execute` to run that configuration. Export the aggregate after the run:

```sh
python3 scripts/scholarcatalyst_live_analysis.py \
  --bench-dir /path/to/benchmark --evidence-dir /path/to/private-evidence \
  --output /path/to/aggregate.json
```

The analysis verifies data, labels, cohort, source guards, runner code, skill, and settings against the ledger fingerprint. Use the same runner revision for later analysis. The aggregate contains fixed error categories and summary values. It excludes private questions and paper identifiers.

## Evidence

Keep benchmark data and complete traces outside the source repository. The private evidence includes the cohort manifest, metadata guards, tool cache, requests without credentials, responses, failures, and budget ledger. Publish aggregate results and code versions.

Report Recall@5, Recall@15, nDCG@15, exposed candidate recall, list depth, invalid selections, unmatched papers, failures, and latency. Compare the tool arms on the same questions. Resample source papers for uncertainty estimates.

The live corpus, restricted tools, sparse labels, and possible model knowledge of source projects limit the interpretation. This experiment does not measure the full OpenResearch application or establish an official ScholarCatalyst score.
