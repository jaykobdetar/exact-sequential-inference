# Historical technical notes

These notes describe the original project, not a verified release bundle. The referenced `evidence/`, validation report, amendment and parent approvals are **absent from this repository**. Claims about historical review are unverified. See [README.md](README.md) for current setup and status.

# G2 · Exact sequential inference

This standard-library Python package supplies two separate formal generators for proposed L1 and L2 experiments: `active_sensing` and `noisy_temporal`. It implements new sequential decisions and noisy Markov inference. Task-8's static terminal calibration and task-6's deterministic revision lookup are read-only references, not dependencies of this package.

Version **1.0.2**; record schema **exact_sequential_inference/1.0**. The source specification and architect-approved `semantic_conventions_v1.1.json` are pinned under `evidence/`. See `reports/VALIDATION_REPORT.md` for checks actually executed, recorded failures, resource evidence and remaining limitations. An independent review found metadata-binding gaps, corrected in v1.0.2; reviewer recheck is pending. See `reports/AMENDMENT_v1.0.2.md`.

## Reproduce locally

Python 3.10+; no third-party runtime dependencies, model APIs, training, GPU, network or credentials. Commands run one process and use no worker pool. Before running checks on a shared computer, follow the current parent-approved resource gate recorded in the validation history; do not assume an earlier threshold still applies. The observed release commands and address-space limit are recorded in the validation history.

```bash
python -m unittest discover -s tests -v
python -m exact_sequential_inference generate --config configs/smoke.json --output data-new
python -m exact_sequential_inference validate data-new --report reports/new-validation.json
python -m exact_sequential_inference export data-new --view model --output model-inputs.jsonl
python -m exact_sequential_inference export data-new --view adaptation --cutoff 31 --output adaptation-inputs.jsonl
python -m exact_sequential_inference solve active_sensing examples/active-input.json
python -m exact_sequential_inference solve noisy_temporal examples/temporal-input.json
```

Generation refuses existing output directories. Reports and exports refuse overwriting files. A byte-cap failure retains its partial output for diagnosis. Use a new directory for a changed source snapshot. The manifest pins exact source/config/schema/spec hashes and data hashes. Do not row-resplit a generated dataset. `validate_record` now checks the fixed primary release profile as well as the exact numerical result: axis values, likelihood/surface regimes and temporal copy/cadence labels must match the formal input. The generic solvers retain their broader bounded domains. Dataset validation additionally requires every configured group slot and its complete paired factorial.

## Active sensing: belief and future loss

The main release has four hypotheses and four binary tests, each usable once. `p_one[test][hypothesis]` is the chance of outcome 1. `loss[guess][hypothesis]` is the nonnegative terminal loss. `initial_horizon` includes tests already in `history`, so remaining opportunities equal initial horizon minus history length. A new test pays its cost, reveals its outcome and consumes one opportunity. Stopping selects a guess. At remaining horizon zero, only guesses are available. Sunk history costs do not enter future action values. Hypotheses use indices 0..H−1 and tests H..H+T−1. The smallest minimizing index is the canonical action; `audit.optimal_actions` retains the complete tied set. The scorer also accepts alternative optimal actions.

Bayes' rule and a posterior-normalized Bellman recurrence compute the label and all reachable decision-node traces. A separate unnormalized-risk recurrence independently cross-checks every emitted four-state label and tied action set. A third tiny oracle constructs all deterministic policy trees before minimizing, with no child-policy pruning. Production supports 2–4 hypotheses and 1–4 tests to permit small exhaustive tests. A full four-test depth-three traversal has 249 ordered-history nodes when every branch has positive probability. Zero-mass histories get `impossible_evidence`, null posterior/risk/action and an empty optimal set; no conditional decision is invented.

Each ordinary canonical table includes two priors, four predeclared ordered histories and a 2×2 cost/loss intervention. The posterior must remain exactly unchanged across all four interventions. The anchor family has an exactly balanced entropy-reduction-maximizing test and another test that perfectly resolves the costly event H=0; its inexpensive-test utility optimum differs from the entropy optimum. Its action and loss values are exact; the entropy-order certificate clears rational joint-mass denominators and compares exact rational products proportional to exp(−M H(H|Y)), with no floating approximation or hand-assigned entropy label. Anchors are outside all training and test splits.

## Noisy temporal inference: three questions

Each released family has 16 independent binary chains over times 0..63. X₀ has the declared prior. Between successive event times, symmetric flip hazard is 1/100 or 2/25. Each declared source has bit error 0 or 1/5, forming four cells. Distinct origins are independent conditional on the trajectory, including independent reports with the same source/entity/event time. Two source identities are used within each cell; their error rates are equal in this primary crossed design. General declared stochastic transition and emission matrices are also supported by the solver and tested on a small asymmetric case.

Reports expose `event_time` and `arrival_time`, with event ≤ arrival. Arrival delays and the reporting schedule are exogenous. Copies preserve origin, entity, event time, source and bit. The first available copy introduces that origin's emission once. Conflicting copies are invalid input. Copies may arrive later, and input array order carries no extra evidence. Each base stream is paired with added copies and the reversed duplicate array. The primary cadence is exactly one original report at each t=0..63, for entity t mod 16: 64 origins and four reports per entity. Independent corroboration is a separate tiny diagnostic test, not extra primary exposure. Dense 1,024-origin streams are an unimplemented extension; no token-fit claim is made without a tokenizer audit.

| Query kind | Target | Conditioning cutoff |
|---|---|---|
| `current_filter` | X_c | reports arriving by c |
| `historical_smoother` | X_t, t ≤ c | reports arriving by c |
| `historical_as_known` | X_t | reports arriving by t; cutoff must equal t |

The forward/backward implementation attaches emissions to event time, using only origins available by cutoff. The independent scalable oracle clamps a target state and sums with a forward transfer recurrence, without a backward pass. A third oracle enumerates all latent paths on chains of at most eight steps. Traces contain exact emission products, alpha and beta arrays for queried entity/cutoff pairs, origin IDs and evidence masses for every entity, and global evidence likelihood. The latter is the product of per-entity likelihoods. Any zero factor makes every query at that cutoff impossible, even if the queried entity alone has positive evidence mass. Contradictory noiseless streams are diagnostic tests only; they are not included in primary possible-stream posterior means.

The generator transiently samples latent states to emit reports but never releases latent states or uses them as posterior labels. All targets are recomputed from declared parameters and public reports. Target-world adaptation uses the explicit source-only export: parameters, cutoff and available reports. Evaluation posteriors and traces remain outside adaptation.

## Splits and safe views

The active split unit is the exact likelihood-table orbit under common hypothesis permutations, arbitrary test permutations and independent binary outcome flips. Priors, histories, costs, loss matrices and renderers never change that group. Canonicalization chooses the minimum column orientation and sorted test order for each hypothesis permutation, so it is exact within the capped domain. It deliberately does not claim to remove every common arithmetic subproblem or every lower-dimensional posterior equivalence.

The temporal split unit is the full stream. A canonical signature removes display names and origin spellings while retaining independent-origin multiplicity, source emission matrices, event times and earliest arrivals. Sorting chain signatures also removes entity renamings. All copy, order and renderer variants remain in that family. The signature does not quotient arbitrary binary-state complements; the declared split requirement concerns full seeded streams and their specified variants.

`train`, `validation` and `test_id` use the same declared parameter supports and standard renderer, with disjoint whole groups. Active `ood_likelihood` reserves probabilities {0,1/3,2/3,1}; ordinary groups use {1/5,2/5,3/5,4/5}. `ood_surface` reserves the `reserved` namespace and a pretty-JSON renderer, with a standard-rendered paired control in the same group. This combines names and renderer shifts; it is not a separately identified name-only effect. Temporal hazard/error cells occur in every temporal split. New cost/prior axes are within-group interventions on held-out tables; this release does not claim a separate cost-OOD training distribution.

A complete record is an evaluation/audit container, **not an allowed model input**. `prompt` contains one approved view per decision/query. Temporal prompts exclude future arrivals separately for each query. The model export exposes no split, internal IDs, targets, proof arrays, latent states, or audit fields. It exports structured payloads in standard JSON; renderer-sensitive experiments must consume the saved `prompt` strings directly, because reserializing payloads removes the reserved formatting intervention. Public `origin_report_id` and `report_id` remain in reports because provenance is part of the task. The adaptation export excludes questions and all evaluation labels. Keep the manifest's whole-group split map alongside exports; never infer split assignment from individual exported rows.

## Experiments this enables—and does not run

For L1, compare posterior error and exact future-loss regret on matched costs/losses; decompose inference versus action errors with oracle-posterior/oracle-policy interfaces. At least one competent validation interface is required before interpreting shifts. Report canonical-action accuracy separately from optimal-set accuracy. An impossible history is a status task, excluded from posterior-error and policy-regret denominators and included in the status denominator. Cheap-test use, high-cost stopping, cost invariance and the entropy-disagreement anchor are mechanical diagnostics.

For L2, compare current and historical posterior errors by hazard/error cell and paired duplicate factor. Preserve timestamps, provenance and the declared model. Evaluate record recall before interpreting failed state inference. The main release has eight queries per stream variant over two of the 16 entities; it does not exhaust all entity/time/cutoff combinations. Baseline audits use fixed latest-report, majority and decay rules without fitting or training. Metadata-sensitive matched examples and exhaustive tests establish semantics; they do not establish that a model avoids shortcuts.

This is a modest smoke release. The proposed 20,000/2,000/2,000 active-table banks and 256/1,024 temporal families per cell have not been generated, trained or evaluated. Resource caps intentionally prevent silently scaling this configuration to those sizes. Freeze larger distributions, competence gates, seed banks and label-information budgets before future experiments. Cross-review, experimental model training and Library packaging belong to the parent workflow.
