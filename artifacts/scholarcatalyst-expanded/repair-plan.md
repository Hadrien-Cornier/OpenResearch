# One operational repair of the fixed comparison

The first expanded attempt stops after an unconfirmed provider request. It records 36 successful final results, one search failure, one unconfirmed request, and 590 results that the global stop prevents.

No held-out score difference is inspected. The stopped run remains separate and cannot support a quality conclusion.

## Fixed repair contract

Run the same 314 questions from the same 157 held-out source papers. Retain seed 531, source guards, both tool arms, model, provider, reasoning, prompts, tool limits, and rank validation.

Use four query workers. This reduces concurrency from the interrupted attempt. Latency is not directly comparable across attempts.

Start a fresh experiment with separate evidence. Do not import old rankings, model responses, or partial scores. Preserve the interrupted evidence and its unresolved request.

Make one repair attempt. Analyze the fresh fixed cohort once after execution. Do not repeat it because of the observed scores.

## Predetermined request-failure policy

Each request has one attempt. An isolated transport error or an unconfirmed usage record fails that arm for that question. Its final score is zero. Continue with the other planned questions.

Each request reserves the full advertised endpoint context limit and its maximum output. Confirmed usage releases the unused reservation. An unconfirmed request consumes the full reservation. The private ledger keeps those upper bounds separate from observed usage. The finite global limit includes both. This bound assumes that the endpoint honors its advertised limits.

A rate-limit response applies a shared cooldown before future requests. It does not repeat the failed request.

Authentication, configuration, or account errors stop the experiment globally. A known response that exceeds its reservation also stops the experiment. A global stop produces an incomplete result.

Completion of the fixed scoring cohort and completion of usage records are separate states. Report request failures by arm. Preserve all questions in the denominator.

## Primary analysis

The primary result remains balanced Recall@5 for the OpenResearch skill minus ordinary alphaXiv tools. All final failures receive zero credit.

Use 10,000 paired whole-source-paper bootstrap draws with seed 531. Each sampled paper retains both questions and both arms. Do not combine the interrupted attempt with the repair.

The confidence interval measures variation across source papers for this run. It does not measure variation across model seeds. The live corpus, sparse labels, and restricted replay still limit the claim.
