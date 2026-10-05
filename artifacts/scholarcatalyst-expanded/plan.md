# Fixed held-out comparison plan

This run compares the OpenResearch literature skill with ordinary alphaXiv tools on 314 questions from 157 unused source papers.

## Cohort and conditions

The pinned dataset contains 207 source papers and 894 questions. The pilot uses 50 distinct papers. This run excludes those papers entirely.

Each remaining paper supplies its core question and one randomly selected subfield question. Seed 531 fixes the selection before any model call.

Both arms answer every question. The run uses eight query workers. This changes execution concurrency from the pilot, so latency is not directly comparable. The complete cohort contains 628 arm results. The model remains `z-ai/glm-5.3-flash`, with first-party `z-ai`, `fp8`, and low reasoning. The model choice uses current [Artificial Analysis results](https://artificialanalysis.ai/models/glm-5-3-flash). The [provider metadata](https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints) confirms the endpoint and supported parameters.

The run preserves the pilot prompts, literature skill, tool limits, date guards, source exclusion, and rank validation. Both arms use alphaXiv discovery.

## Primary result and uncertainty

Recall@5 is the fraction of labeled positive papers in the first five result slots. The primary result is skill Recall@5 minus ordinary-tools Recall@5.

For each source paper, calculate the paired difference for each question. Average its core and subfield differences. Then average across all 157 papers.

The 95% confidence interval uses 10,000 paired bootstrap samples with seed 531. Each sample selects 157 source papers with replacement. Each selected paper retains both questions and both arms.

This method gives equal weight to source papers and question types. It preserves dependence between questions from the same paper.

Recall@15, nDCG@15, and separate question-type results are secondary descriptions. They do not change the primary decision rule.

## Fixed stop and failure policy

Complete the fixed 628 arm results, then perform one final outcome analysis. Progress checks inspect completion, errors, and budget only.

Do not inspect interim score differences. Do not add questions, extend the run, or repeat completed results because of the observed scores.

A failed final result scores zero. Do not exclude failed questions or source papers. Preserve invalid and unjudged result slots.

The hard budget remains a stop condition. If it prevents completion, report the experiment as incomplete. Preserve every planned question in the analysis.

## Expected precision and limits

The pilot suggests a confidence interval half-width of about 2.8 to 3.8 percentage points. This estimate depends on the unknown dependence between question types.

The pilot difference is about 2.2 percentage points. The expanded run can narrow the interval, but it cannot guarantee a statistically significant result.

All 676 unused questions would add 362 subfield questions while the count remains 157 source papers. Pilot-based calculations suggest only a modest further precision gain.

The source paper is the available project identifier. Papers can share authors, topics, or earlier project history. The bootstrap does not remove that dependence.

The live search corpus differs from the frozen benchmark corpus. Sparse labels do not establish relevance for unjudged papers. Model knowledge can include the original projects.

This result applies to the restricted live comparison under these fixed conditions. It is not an official frozen-corpus benchmark result.
