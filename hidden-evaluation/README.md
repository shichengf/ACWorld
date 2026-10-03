# ACWorld hidden evaluation with Harbor

This adapter runs commerce Agents in Harbor 0.22.0. An Agent reads its assigned role's observation and submits business decisions through a command-line client. A separate ACWorld container executes those decisions using the frozen v1.0.0 runtime and scorers. Harbor then collects the result from that container and validates it in a separate verifier.

The initial hidden set contains 20 configurations, two per task family. It is versioned separately from the paper's 200 public tasks. We plan to expand it in subsequent releases.

## Running an Agent

The evaluation operator keeps the assembled task bundle private. Agents use `python /app/acworld.py observe` and `python /app/acworld.py submit --request-id REQUEST_ID --decision FILE` inside Harbor. Each observation includes the business request and response schema. The operator selects the submitted Agent and model through Harbor's standard Agent interface.

Model credentials belong to the running Agent's provider configuration. The ACWorld service does not call a model or require a provider key. API usage is charged to the configured provider account. Custom Agents can use Harbor's Agent interface and the same commerce client.

Run the commands below from `hidden-evaluation/` in the repository, or from the extracted adapter directory. With Docker and uv installed, the operator can run:

```bash
HARBOR_TELEMETRY=0 uvx --from harbor==0.22.0 harbor run \
  -p /private/hidden-set/harbor-build/tasks -a AGENT_NAME -m MODEL_ID -e docker -n 1 \
  -o /private/hidden-set/harbor-jobs
```

Set the credentials required by the chosen Harbor Agent before running this command.

## Scores and feedback

Harbor records `reward` as complete task success and `capability_score` as the original partial credit. The scorer, predicates, and weights are unchanged. An unfinished execution or infrastructure error remains unscored. The aggregate exporter requires every task to have a verified result before producing suite means.

```bash
python summarize.py --job /private/hidden-set/harbor-jobs/JOB_NAME \
  --build-manifest /private/hidden-set/harbor-build/build-manifest.json \
  --agent-version AGENT_VERSION --output feedback.json
```

Only aggregate feedback is returned to submitters. Task configurations, reference solutions, and detailed execution records stay with the evaluation operator. A person with access to the complete bundle or Docker host can inspect them, so that host must be controlled by the trusted operator.

## Building and checking the private bundle

`build.py` consumes an existing frozen private task set and refuses to overwrite an earlier build. It checks the source and fixture hashes, generates one Harbor task per configuration, and runs each reference policy without using a model API. The resulting bundle includes private reference solutions for Harbor's `oracle` check.

```bash
python build.py --private /private/hidden-set --output /private/hidden-set/harbor-build
HARBOR_TELEMETRY=0 uvx --from harbor==0.22.0 harbor run \
  -p /private/hidden-set/harbor-build/tasks -a oracle -e docker -n 1 \
  -o /private/hidden-set/harbor-jobs
python -m unittest discover -s . -p 'test_*.py'
```

Use Python 3.11 or later with the dependencies in `adapter/requirements.txt` for assembly. Docker images install those pinned dependencies. The generated task bundle and `oracle` solutions are private.

The Agent container receives the commerce client and role-visible requests. The independent verifier reads the trusted service artifact after Agent execution ends. Agent-written reward files do not determine the result.
