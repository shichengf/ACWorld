# [ACWorld](https://shichengf.github.io/ACWorld/)

[Paper](https://arxiv.org/abs/2608.02441) · [Project page](https://shichengf.github.io/ACWorld/) · [MIT License](LICENSE)

ACWorld is an extensible commerce environment for Buyer and Merchant agents.
Agents discover products, negotiate, and complete transactions through the
Vibe Commerce Protocol. The platform validates actions, the World maintains
commercial state, and execution traces record what happened for replay and
scoring.

## Run the environment

With Python 3.11+, install the code and check a bundled scenario without model calls:

```bash
git clone https://github.com/shichengf/ACWorld.git
cd ACWorld
python -m pip install -e .
acworld-scenario dry-run scenarios/s1_direct_purchase_42.yaml
```

Run the scenario with an OpenRouter model (incurs API charges):

```bash
acworld-run --scenarios scenarios/s1_direct_purchase_42.yaml \
  --api-key-file /absolute/path/to/key.txt --limit 1 --allow-paid-inference
```

Records are saved under `reports/`. Run `acworld-replay reports/` to verify them
and reconstruct episode state.

## Reuse and extend modules

Start with [`scenarios/`](scenarios/), or create a scenario with
`acworld-scenario init my_scenario.yaml`. Set its products, buyer goals,
merchant policies, and success conditions; check it with
`acworld-scenario dry-run my_scenario.yaml`.

Reuse the included pet-supplies module:

```bash
acworld-scenario generate --plugin cwe_examples.pet_supplies_domain \
  --generator example.pet-supplies --seed 42 --out output/pet.yaml
acworld-scenario dry-run output/pet.yaml
```

To add a module, adapt an example and register a unique name through
[`episode.extensions`](src/episode/extensions.py):

- **Product domains:** [pet_supplies_domain.py](src/cwe_examples/pet_supplies_domain.py); select your module and registered name with `generate --plugin ... --generator ...`.
- **Ranking policies:** [round_robin_ranking.py](src/cwe_examples/round_robin_ranking.py); select the registered name in the scenario's `platform_policy.name`.
- **World events and checks:** [flash_restock_event.py](src/cwe_examples/flash_restock_event.py); configure `world_events` and `extension_evaluations` in the [scenario schema](src/episode/scenario.py).

Keep modules importable and load policies or events with
`acworld-run --plugin your_module ...` alongside the run arguments above.
The existing example modules work the same way. Buyer and Merchant capabilities
are organized in [`skills/`](skills/).

## Benchmark and research artifacts

The benchmark is a reference application: 200 capability tasks and a separate
60-task large-catalog workload. See the [evaluation guide](docs/benchmark.md)
for commands, code versions, model settings, and published result downloads,
or the [Harbor adapter](hidden-evaluation/README.md) for hidden evaluation.

## Data and license

The prepared catalog is available from the
[large-catalog data release](https://github.com/shichengf/ACWorld/releases/tag/large-catalog-data-v1);
[download instructions](docs/benchmark.md#download-the-large-catalog) cover installation and validation.
Springbrand provided the source catalog for the 200-task benchmark and authorized
research use and publication of the adapted data; see [provenance](artifacts/catalog-provenance/README.md).

Code and documentation use the [MIT License](LICENSE). Catalog data are separate:
they may be used for ACWorld research reproduction and evaluation, but may not
be redistributed, mirrored, rehosted, sublicensed, sold, or republished.
Contact the maintainers for other uses. See [CITATION.cff](CITATION.cff) for citation metadata.
