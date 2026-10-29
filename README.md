# Exact sequential inference

Exact rational inference and synthetic dataset generation for two small experimental worlds:

- **Active sensing:** Bayesian belief updates and finite-horizon decisions that minimize expected future test and terminal loss.
- **Noisy temporal inference:** binary Markov chains with event/arrival times, delayed reports, origin-aware deduplication, filtering and smoothing.
- Independent reference solvers, exhaustive tiny-case oracles, paired interventions, whole-world splits, validation, scoring and label-free exports.

This is a research prototype, version 1.0.3. It does not run model training, call model APIs, download models or require a GPU. It is not a complete historical experiment reproduction bundle.

## Setup

Python 3.10+; no third-party runtime dependencies. From this checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m exact_sequential_inference --help
.venv/bin/python -m unittest discover -s tests -v
```

Installation may fetch setuptools from PyPI for the build. Alternatively, run the module and tests directly with `python3` from the checkout without installing anything. JSON Schema tests optionally use `jsonschema`; the standard-library semantic validation and solver tests do not require it. Tests and helper scripts should be run from the repository root. The installed module can solve, generate, validate and export from any directory; example/config paths must point to this checkout or your own input files.

## Small example workflow

```bash
python3 -m exact_sequential_inference solve active_sensing examples/active-input.json
python3 -m exact_sequential_inference solve noisy_temporal examples/temporal-input.json
python3 -m exact_sequential_inference generate --config configs/smoke.json --output data-new
python3 -m exact_sequential_inference validate data-new
python3 -m exact_sequential_inference export data-new --view model --output model-inputs.jsonl
python3 -m exact_sequential_inference export data-new --view adaptation --cutoff 31 --output adaptation-inputs.jsonl
python3 scripts/reproduce.py data-new
```

Generation refuses existing output directories, and exports/reports refuse existing files. A generation byte-cap failure retains partial output for diagnosis. Keep generated data outside version control. The smoke configuration caps data at 20 MB; inference uses exact fractions and can be slow on large rational inputs. Do not treat these bounded solvers as an internet-facing service for untrusted inputs.

A complete record contains evaluation labels and traces: **never use it directly as a model input**. Model exports allowlist the formal inputs and filter temporal reports by query cutoff. Adaptation exports omit queries and labels. Keep the manifest's group/split map separately; never row-resplit paired worlds. Structured exports normalize formatting; use saved prompt strings when testing renderer interventions.

Generic active solvers support 2–4 hypotheses, 1–4 tests and horizon at most 3. Generic temporal solvers support up to 16 independent binary chains and 64 steps. Dataset validation enforces the narrower fixed experimental profile in `release_contract.py`. Rational inputs are reduced strings such as `"1/2"`, not floats. Impossible evidence produces null posterior/action fields rather than an invented conditional answer.

## Provenance and limitations

The repository began with source snapshot `9c83ac7` of the user-selected project P0252 on 2026-10-29. The owner described the project as AI-assisted work. Original development history was not imported; the Git author identity is not proof of ownership of every contribution.

The original specification, `semantic_conventions_v1.1.json`, historical validation reports, amendment and release datasets are **not bundled**. They were not recovered or reconstructed during this cleanup. New manifests explicitly mark historical evidence as `not_bundled`; passing current tests does not authenticate historical approval or scientific conclusions. The preserved [technical notes](README.source.md) explain the algorithms and experiments but contain historical claims that remain unverified.

Manifests hash the config, data and source assets available in the running installation. A source checkout includes schemas/tests/scripts in that hash; a wheel installation contains the Python package only. Generate and validate within the same unchanged installation. These hashes detect changes, not authorship or missing historical evidence. `scripts/reproduce.py` checks seeded record regeneration separately.

The external task-8 comparison is optional and skips unless explicitly supplied:

```bash
python3 scripts/task8_anchor_check.py
python3 scripts/task8_anchor_check.py --task8-root /path/to/trusted/task8-checkout
```

The second command imports code from the supplied checkout and reports its source hashes; it is not a historical version pin. This repository does not supply that dependency. `scripts/bounded_run.py` is an optional Linux-only resource wrapper using `/proc`, process limits and one CPU; core functionality is cross-platform Python.

## Release readiness

See [RELEASE_READINESS.md](RELEASE_READINESS.md) for verification and outstanding decisions. No license has been selected: public visibility would not itself grant reuse permission. The owner must confirm rights/provenance and choose an appropriate license before describing this as an open-source release. Repository visibility remains private pending owner review.
