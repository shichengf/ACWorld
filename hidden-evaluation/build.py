"""Assemble private Harbor tasks from the existing frozen ACWorld hidden set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(task_id, destination):
    from hidden_tasks import installed
    from episode.capability_runtime_registry import runtime_bundle_v2
    from episode.scenario import population_for_scenario
    from experiments.benchmark_executor import BenchmarkExecutor
    from experiments.benchmark_plan import RunSpecV2, LOCAL_REFERENCE_MODEL_V2
    rows = []
    with installed():
        bundle = runtime_bundle_v2(task_id)
        population = population_for_scenario(bundle.scenario)
        ideal = bundle.ideal_channel()

        class Spec(RunSpecV2):
            def __post_init__(self):
                if self.task_id != task_id:
                    raise ValueError('Unexpected reference task')

        class Channel:
            supports_business_decisions = True
            supports_decision_evidence_context = True

            def complete_business_decision(self, **kwargs):
                reply = ideal.complete_business_decision(**kwargs)
                rows.append({'decision': json.loads(reply.content),
                             'system_sha256': hashlib.sha256(kwargs['system_prompt'].encode()).hexdigest(),
                             'user_sha256': hashlib.sha256(kwargs['user_prompt'].encode()).hexdigest()})
                return reply

        run = Spec(suite='main', model_id=LOCAL_REFERENCE_MODEL_V2, task_id=task_id,
                   evaluated_role=bundle.task.evaluated_role,
                   buyers=len(population.buyers), merchants=len(population.merchants))
        destination.mkdir(parents=True)
        result = BenchmarkExecutor(model_channels=lambda *_: Channel(), max_model_calls_per_run=48)(run, destination)
        if result['metrics']['capability_score'] != 1.0 or result['metrics']['strict_success'] is not True:
            raise ValueError('Reference did not satisfy the original scorer: ' + task_id)
        save(destination / 'result.json', result)
    return rows


def build(private, output):
    private, output = private.resolve(), output.resolve()
    if output.parent != private:
        raise ValueError('Use a new directory directly inside private, outside the frozen inputs')
    if output.exists():
        raise ValueError('Refusing to overwrite an existing build')
    sys.path.insert(0, str(private))
    from prepare import check_frozen
    manifest = check_frozen()
    if len(manifest['tasks']) != 20 or len({r['task_id'] for r in manifest['tasks']}) != 20:
        raise ValueError('This pilot adapter expects 20 unique frozen tasks')
    manifest_sha = digest(private / 'manifest.json')
    output.mkdir(parents=True)
    shutil.copyfile(private / 'manifest.json', output / 'task-manifest.json')
    task_root = output / 'tasks'
    task_root.mkdir()
    inventory = []
    for index, row in enumerate(manifest['tasks'], start=1):
        # Directory names need not disclose family, role, or generation parameters.
        task = task_root / f'acworld-hidden-{index:02d}'
        shutil.copytree(HERE / 'templates', task,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        service = task / 'environment' / 'acworld-server'
        hidden = service / 'private'
        hidden.mkdir()
        for name in ('baseline-lock.json', 'bootstrap.py', 'hidden_tasks.py', 'prepare.py', 'manifest.json'):
            shutil.copyfile(private / name, hidden / name)
        for name in ('baseline', 'fixtures'):
            shutil.copytree(private / name, hidden / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        shutil.copytree(HERE / 'adapter', service / 'adapter',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        expected = {'task_id': row['task_id'], 'task_manifest_sha256': manifest_sha,
                    'task_set_version': manifest['version'], 'baseline_commit': manifest['baseline_commit'],
                    'scorer_version': manifest['scorer_version']}
        save(service / 'case.json', expected)
        save(task / 'tests' / 'expected.json', expected)
        replies = reference(row['task_id'], output / 'reference-evidence' / row['task_id'])
        solution = task / 'solution'
        solution.mkdir()
        save(solution / 'decisions.json', replies)
        shutil.copyfile(HERE / 'reference_replay.py', solution / 'replay.py')
        (solution / 'solve.sh').write_text('#!/bin/sh\nset -eu\npython /solution/replay.py /solution/decisions.json\n')
        inventory.append({'directory': task.name, **expected, 'reference_decisions': len(replies)})
        print(json.dumps({'built': task.name, 'reference_score': 1.0}), flush=True)
    check_frozen()
    save(output / 'build-manifest.json', {
        'schema': 'acworld.harbor-build.v1', 'harbor_version': '0.22.0',
        'task_set_version': manifest['version'], 'task_manifest_sha256': manifest_sha,
        'baseline_commit': manifest['baseline_commit'], 'tasks': inventory,
        'adapter_hashes': {p.relative_to(HERE).as_posix(): digest(p) for p in sorted(HERE.rglob('*'))
                           if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'},
        'usage': 'private evaluator package, includes hidden configurations and reference solutions'})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.private, args.output)


if __name__ == '__main__':
    main()
