# ACWorld

ACWorld is an extensible commerce environment for independent Buyer and Merchant
agents. Its Vibe Commerce Protocol (VCP) connects agent proposals to platform
validation before accepted actions change persistent World state. Execution
records link decisions to their evidence and effects, enabling researchers to
inspect agent behavior and reconstruct recorded state.

## Run the environment

With Python 3.11+, install from the repository root and check a bundled scenario
without model calls:

```bash
python -m pip install -e .
acworld-scenario dry-run scenarios/s1_direct_purchase_42.yaml
```

Run that scenario with an OpenRouter model (incurs API charges):

```bash
acworld-run --scenarios scenarios/s1_direct_purchase_42.yaml \
  --api-key-file /absolute/path/to/key.txt --limit 1 --allow-paid-inference
```

Execution records are saved under `reports/`. Use `acworld-replay reports/`
to verify recorded runs and reconstruct their state.

## Reuse and extend modules

Start with a scenario in [`scenarios/`](scenarios/), or create one with
`acworld-scenario init my_scenario.yaml`. Define the products and the buyer's
goal, then specify how merchants respond. Set success conditions that reflect
the intended task and check the setup with
`acworld-scenario dry-run my_scenario.yaml`.

Reuse the included pet-supplies generator:

```bash
acworld-scenario generate --plugin cwe_examples.pet_supplies_domain \
  --generator example.pet-supplies --seed 42 --out output/pet.yaml
acworld-scenario dry-run output/pet.yaml
```

To add a module, follow an example and register it under a unique name through
[`episode.extensions`](src/episode/extensions.py):

- **Product domains:** adapt [pet_supplies_domain.py](src/cwe_examples/pet_supplies_domain.py); select your module and registered name with `generate --plugin ... --generator ...`.
- **Ranking policies:** adapt [round_robin_ranking.py](src/cwe_examples/round_robin_ranking.py); select the registered name in the scenario's `platform_policy.name`.
- **World events and checks:** adapt [flash_restock_event.py](src/cwe_examples/flash_restock_event.py); configure `world_events` and `extension_evaluations` in the [scenario schema](src/episode/scenario.py).

Keep your module importable and load policies or events with
`acworld-run --plugin your_module ...` alongside the run arguments above.
The existing example modules can be loaded in the same way.

## Executable task suite

Within ACWorld, we instantiate 200 capability-coverage tasks and 60 large-catalog
tasks to study how agents carry out commercial goals. These executable scenarios
provide concrete settings for collecting interaction trajectories. Programmatic
checks then connect recorded behavior to task requirements, supporting analysis
of intermediate progress as well as comparisons across models.

The [task-suite runner](run_benchmark.sh) runs on Linux with Bash. The 200
capability-coverage tasks use the bundled task definitions and scenario data. Running
`--tasks 60` or `--tasks 260` additionally requires the large-catalog data package
described below. Existing command names are retained for compatibility:

```bash
./run_benchmark.sh run --help
```

## Data and license

The large-catalog data package is omitted from this anonymous repository and
will be released publicly with supporting materials and data-use terms. No
default download URL is configured in this snapshot. Users who already have
authorized source CSV files can supply `--data-root` to the runner; an available
prepared package can be installed with
`./scripts/run_large_catalog.sh download --asset-url <package-url>`.
The original merchant catalog snapshots are also omitted. Their provenance and
the provider's authorization for research use and release are documented in
[catalog metadata](data/catalog_provenance.json); this metadata describes the
capability-coverage sources, not the separate large catalog.
Code and documentation use the [MIT License](LICENSE).
