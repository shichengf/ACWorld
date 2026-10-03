# [ACWorld](https://shichengf.github.io/ACWorld/)

[![arXiv](https://img.shields.io/badge/arXiv-2608.02441-b31b1b.svg)](https://arxiv.org/abs/2608.02441)
[![Project Page](https://img.shields.io/badge/Project-Page-12a594.svg)](https://shichengf.github.io/ACWorld/)
[![Benchmark](https://img.shields.io/badge/Benchmark-260_tasks-6d5bd0.svg)](#choose-the-benchmark)
[![License](https://img.shields.io/badge/License-MIT-1677b8.svg)](LICENSE)

**Agentic Commerce World: An Auditable and Verifiable Environment for Vibe Commerce**

ACWorld is a many-to-many environment for training and evaluating Buyer and
Merchant agents. Agents act through the Vibe Commerce Protocol, the Commerce
Intelligence Platform validates each action, and the World records authorized
transaction effects.

The released benchmark has two parts:

- **Capability benchmark:** 200 tasks across ten commerce families and 80
  capabilities.
- **Large-catalog benchmark:** 60 tasks across four families and 18
  capabilities. It searches 785,022 listings derived from 791,431 source
  records and scores decisions against deterministic full-catalog oracles.

The two parts can be run separately or together. They retain separate scores
because they test different distributions.

## Paper and current versions

The paper's 200-task capability-coverage results use the [`v1.0.0` code package](https://github.com/shichengf/ACWorld/tree/v1.0.0) at commit `169602300bb0b9c51c69c0c8afa54c0756a3f7d3`, also named `kdd2027-submission`.

Use `git checkout v1.0.0` before reproducing the paper's task and scorer configuration. The current `v1.1.0` release points to `250f04cf157e4587ca807a75561a996843ea4b90`. It changes the evaluation interface and scoring, so results from the two versions must be reported separately.

The paper's 200-task results and the later large-catalog track are separate evaluations.

## Hidden evaluation

The [Harbor adapter](hidden-evaluation/README.md) runs submitted Agents against 20 private configurations using the v1.0.0 runtime and scorers. Agents use their own model configuration and provider credentials. The evaluation operator keeps the task bundle private and returns aggregate results. This initial set is separate from the 200 public development tasks and is intended to expand in future releases.

## Changes in 1.1.0

Revises the 200-task capability benchmark. The 60-task large-catalog benchmark
is unchanged.

- Task requests read like real Buyer and Merchant messages rather than
  evaluation specifications.
- Internal field and component names no longer appear in what the model sees.
- Responses are read more tolerantly, so ordinary formatting no longer fails a
  valid decision.
- After-sales tasks no longer publish the operations the scorer expects.
- Refusing a request the published policy does not cover is now sometimes the
  correct answer, and is scored as one.
- Credit rests mainly on the disposition reached and the commercial state it
  left behind rather than on naming operations in order.
- Facts a party could look up for itself are no longer pre-filtered into the
  task brief.
- Comparison candidates are closer, and the correct answer is no longer
  restricted to the first few positions.
- `--workers` may be raised above 2, and every run reports how many tasks went
  unscored; see [Raising `--workers`](#raising---workers).

The ten task families and 200 task identifiers are retained. Model-visible task content, parsing, and scoring have changed. Scores are not
comparable with 1.0.0; rerun a model rather than rescoring its saved 1.0.0
outputs.

## Quick start

Requirements:

- Git
- Python 3.11 or newer
- network access
- at least 5 GB of free disk space for the large catalog and outputs
- an OpenRouter API key stored in a text file outside the repository

```bash
git clone https://github.com/shichengf/ACWorld.git
cd ACWorld

./run_benchmark.sh run \
  --tasks 200 \
  --model google/gemini-3.6-flash \
  --api-key-file /absolute/path/to/openrouter-key.txt
```

The launcher uses an installed `uv` or installs a pinned copy locally, installs
the locked dependencies, and resumes from completed task files.

## Download the large catalog

The prepared catalog is distributed with the
[`large-catalog-data-v1`](https://github.com/shichengf/ACWorld/releases/tag/large-catalog-data-v1)
release. It contains the complete normalized SQLite database required by the
60-task benchmark. The first run with `--tasks 60` or `--tasks 260` downloads
and assembles its 65 release parts automatically.

To download and check it before running a model:

```bash
./scripts/run_large_catalog.sh download
./scripts/run_large_catalog.sh validate
```

The files are installed under `output/large-catalog/`. Validation checks the
SQLite database and confirms 791,431 source records, 785,022 searchable
listings, and 60 task definitions.

## Choose the benchmark

Run the 200 capability tasks:

```bash
./run_benchmark.sh run \
  --tasks 200 \
  --model google/gemini-3.6-flash \
  --api-key-file /absolute/path/to/openrouter-key.txt
```

Run the 60 large-catalog tasks:

```bash
./run_benchmark.sh run \
  --tasks 60 \
  --model google/gemini-3.6-flash \
  --api-key-file /absolute/path/to/openrouter-key.txt
```

Run both parts in sequence:

```bash
./run_benchmark.sh run \
  --tasks 260 \
  --model google/gemini-3.6-flash \
  --api-key-file /absolute/path/to/openrouter-key.txt
```

Omitting `--tasks` runs the 200 capability tasks. The first 60-task or 260-task
run downloads the prepared catalog automatically. Pass
`--data-root /absolute/path/to/raw_data` only when rebuilding the catalog from
authorized source CSV files.

Use `--model all` to run the configured model panel under the checked-out code version, or repeat
`--model` to select several models. Print every accepted model ID with:

```bash
./run_benchmark.sh list
```

The paper panel contains:

```text
qwen/qwen3.5-plus-20260420
deepseek/deepseek-v4-pro
mistralai/mistral-medium-3-5
google/gemini-3.5-flash
anthropic/claude-sonnet-5
openai/gpt-5.6-terra
openai/gpt-5.6-sol
openai/gpt-5.6-luna
moonshotai/kimi-k3
google/gemini-3.6-flash
```

`anthropic/claude-opus-4.8` is available for optional 200-task runs but is not
part of the paper panel or the 60-task suite.

## Validation and smoke tests

Run the no-cost Buyer and Merchant smoke for the 200-task benchmark:

```bash
./run_benchmark.sh smoke
```

Run one paid 200-task smoke:

```bash
./run_benchmark.sh live-smoke \
  --model google/gemini-3.6-flash \
  --api-key-file /absolute/path/to/openrouter-key.txt
```

Run the three representative large-catalog canaries:

```bash
./run_benchmark.sh run \
  --tasks 60 \
  --model google/gemini-3.6-flash \
  --api-key-file /absolute/path/to/openrouter-key.txt \
  --canary-only
```

To validate the large-catalog data, tasks, deterministic scorers, error
mutations, and process rewards without model calls:

```bash
./scripts/run_large_catalog.sh download
./scripts/run_large_catalog.sh validate
./scripts/run_large_catalog.sh reference
```

The reference command executes all 60 tasks through the normal ACWorld runtime.
Every reference task must receive full credit. The scorer uses Python
predicates only and makes no model calls.

## Execution behavior

- Provider-default generation settings are used.
- `--workers` sets how many tasks run at once, from 1 to 8 (default 2).
- A transport or provider failure is retried once.
- A model protocol error or incorrect decision is kept as a scored outcome.
- Repeating the same command skips completed task files.
- `--max-cost-usd` sets the reported-cost limit for each large-catalog model
  run.

### Raising `--workers`

Tasks are scored in isolation, so the worker count does not change any task's
score. What it changes is how fast calls reach the provider. If your account's
rate limit is lower than that, throttled tasks fail and are **not scored** — and
because the tasks that take the most calls are the ones most likely to be cut
off, what goes missing is systematically the hardest work rather than a random
sample.

Every run therefore reports what did not get scored:

```json
"scored_run_count": 197,
"unscored_run_count": 3,
"provider_failed_run_count": 3,
"provider_failed_run_keys": ["...", "...", "..."]
```

Check those fields before using a result. Re-running the same command retries
only the missing tasks.

The published numbers were produced at `--workers 2`. A run at a higher worker
count should report its code version and `unscored_run_count`. Use `v1.0.0` for the paper configuration.

The large-catalog tools search the full database with explicit filters,
sorting, and pagination. Tool responses remain bounded, while the deterministic
oracle evaluates the complete matching space. Model actions still pass through
the Agent, VCP, Commerce Intelligence Platform, and World before scoring.

## Outputs

Runs and model responses are generated locally. The paper score archive, `ACWorld-paper-scores-10models-200tasks.tar.gz`, contains all 2,000 task scores for the ten evaluated models, their model identifiers, API configuration, and an aggregation script. Readers can inspect individual results and recompute the 130 overall, role, and family means reported in Tables 4 and 13 for the capability-coverage track without new model API calls. These means have been checked against the paper results. Download the [archive](https://github.com/shichengf/ACWorld/releases/download/paper-artifacts-v1/ACWorld-paper-scores-10models-200tasks.tar.gz) and its [SHA-256 checksum](https://github.com/shichengf/ACWorld/releases/download/paper-artifacts-v1/ACWorld-paper-scores-10models-200tasks.tar.gz.sha256).

- [5×5 shared-market study records](artifacts/shared-market/README.md)
- [Catalog provenance](artifacts/catalog-provenance/README.md)

```text
output/
  benchmark/
    <model>/
      runs/
      results.csv
      summary.json
  large-catalog/
    catalog.sqlite
    catalog-summary.json
    tasks.json
    validation-report.json
    reference/
    <model>/
      LC-*.json
      results.csv
      summary.json
    summary.json
```

For the 200-task benchmark:

```bash
./run_benchmark.sh status --model google/gemini-3.6-flash
./run_benchmark.sh paper-report --model google/gemini-3.6-flash
./run_benchmark.sh paper-analysis --model google/gemini-3.6-flash
```

For the 60-task benchmark, each model directory retains the task-level score,
process rewards, terminal state, model calls, and latency. The combined
`output/large-catalog/summary.json` contains model, role, capability, and stage
means together with full, partial, zero, and protocol-error counts.

## Repository contents

- `src/`: Agent, VCP, Commerce Intelligence Platform, World, benchmark
  construction, scoring, and runners.
- `skills/`: reusable Buyer and Merchant capabilities.
- `scenarios/`: deterministic environment examples and market studies.
- `data/`: controlled catalog fixtures and provenance metadata.
- `src/cwe_examples/`: extension examples.
- `tests/`: focused regression tests for the public benchmark paths.

The 200-task benchmark includes the hardened T4, T6, and T7 tasks and the
clarified Multi-item workflow used by the paper. The 60-task definitions are
rebuilt deterministically from the prepared catalog by the checked-out code, so
the scorer, predicates, prompts, and reference answers stay aligned.

## Data and licensing

The repository does not contain raw merchant pages or private user data. The
controlled fixtures are informed by public product information. The prepared
large catalog contains normalized product records, not raw pages. Researchers
who rebuild from CSV files are responsible for using data they are authorized
to process.

Spring Brand provided the source catalog for the 200-task benchmark, collected by its platform from six retailers' public product listings on 21 May 2026. It authorized research use and publication of the adapted benchmark data, subject to restrictions on further redistribution.

ACWorld source code and code documentation are distributed under the MIT
License in [LICENSE](LICENSE). The MIT License does not apply to the prepared
large-catalog database or the derived catalog fixtures.

The data package may be downloaded and used for research reproduction and
evaluation of ACWorld. No permission is granted to redistribute, mirror,
rehost, sublicense, sell, or republish the data package or its contents.
Contact the repository maintainers before any other use. Citation metadata is
provided in [CITATION.cff](CITATION.cff).
