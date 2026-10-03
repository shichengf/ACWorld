"""Export aggregate feedback from a complete, version-bound Harbor job."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_COUNT = 20
spec = importlib.util.spec_from_file_location('acworld_verifier', HERE / 'templates' / 'tests' / 'verify.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def invalid_constant(_value):
    raise ValueError('JSON numbers must be finite')


def object_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON field')
        value[key] = item
    return value


def read_object(path):
    value = json.loads(path.read_bytes(), parse_constant=invalid_constant,
                       object_pairs_hook=object_pairs)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: ' + path.name)
    return value


def load_inventory(build_manifest):
    manifest = read_object(build_manifest)
    if manifest.get('schema') != 'acworld.harbor-build.v1' or manifest.get('harbor_version') != '0.22.0':
        raise ValueError('Unexpected Harbor build manifest or version')
    frozen_path = build_manifest.parent / 'task-manifest.json'
    frozen = read_object(frozen_path)
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != manifest.get('task_manifest_sha256'):
        raise ValueError('Frozen task manifest hash differs from build manifest')
    if (manifest.get('baseline_commit') != verifier.BASELINE_COMMIT
            or frozen.get('baseline_commit') != manifest['baseline_commit']
            or frozen.get('version') != manifest.get('task_set_version')):
        raise ValueError('Build and frozen task manifest bindings differ')
    scorer_version = frozen.get('scorer_version')
    if not isinstance(scorer_version, str) or not scorer_version:
        raise ValueError('Frozen task manifest has no scorer version')
    original = frozen.get('tasks')
    rows = manifest.get('tasks')
    if not isinstance(original, list) or not isinstance(rows, list) or len(original) != TASK_COUNT or len(rows) != TASK_COUNT:
        raise ValueError('A complete build must contain exactly 20 tasks')
    original_ids = [row.get('task_id') if isinstance(row, dict) else None for row in original]
    if any(not isinstance(task_id, str) or not task_id for task_id in original_ids) or len(set(original_ids)) != TASK_COUNT:
        raise ValueError('Frozen task IDs must be 20 unique strings')
    expected = {}
    seen_ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid task inventory entry')
        directory = row.get('directory')
        task_id = row.get('task_id')
        if (not isinstance(directory, str) or not directory or directory in ('.', '..')
                or '/' in directory or chr(92) in directory or directory in expected
                or not isinstance(task_id, str) or task_id in seen_ids):
            raise ValueError('Task inventory directories and IDs must be unique')
        for key in ('task_manifest_sha256', 'task_set_version', 'baseline_commit'):
            if row.get(key) != manifest.get(key):
                raise ValueError('Task inventory binding differs: ' + key)
        if row.get('scorer_version') != scorer_version:
            raise ValueError('Task inventory scorer differs from frozen manifest')
        expected[directory] = row
        seen_ids.add(task_id)
    if seen_ids != set(original_ids):
        raise ValueError('Build task IDs differ from the frozen task manifest')
    return manifest, expected, scorer_version


def agent_identity(trial):
    config = trial.get('config')
    agent = config.get('agent') if isinstance(config, dict) else None
    if not isinstance(agent, dict):
        raise ValueError('Trial result has no complete Harbor Agent configuration')
    declared = agent.get('name') or agent.get('import_path')
    model_id = agent.get('model_name')
    if not isinstance(declared, str) or not declared or (model_id is not None and not isinstance(model_id, str)):
        raise ValueError('Invalid Harbor Agent or model identity')
    info = trial.get('agent_info')
    actual_name, actual_version, actual_model = None, None, None
    if info is not None:
        if not isinstance(info, dict) or not isinstance(info.get('name'), str) or not info['name']:
            raise ValueError('Invalid Harbor Agent information')
        actual_name, actual_version = info['name'], info.get('version')
        model_info = info.get('model_info')
        if model_info is not None:
            if not isinstance(model_info, dict) or not isinstance(model_info.get('name'), str):
                raise ValueError('Invalid Harbor model information')
            actual_model = (model_info.get('provider'), model_info['name'])
            if model_id is None:
                model_id = '/'.join(part for part in actual_model if part)
        special = {'oracle', 'nop'}
        if (declared in special or actual_name in special) and declared != actual_name:
            raise ValueError('Declared and actual Harbor Agent identities disagree')
    return (declared, agent.get('import_path'), model_id), (actual_name, actual_version, actual_model)


def checked_reward(value):
    if not isinstance(value, dict) or set(value) != {'reward', 'capability_score'}:
        raise ValueError('Unexpected Harbor reward fields')
    for score in value.values():
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError('Invalid Harbor reward value')
    if value['reward'] not in (0, 1):
        raise ValueError('Strict success reward must be binary')
    return value


def summarize(job, build_manifest, agent_version):
    job, build_manifest = Path(job), Path(build_manifest)
    manifest, expected, scorer_version = load_inventory(build_manifest)
    scores, seen, identities, actual_identities = [], set(), set(), set()
    failures = 0
    trial_roots = sorted({path.parent for pattern in ('*/config.json', '*/result.json') for path in job.glob(pattern)})
    for root in trial_roots:
        config = read_object(root / 'config.json')
        task_config = config.get('task')
        task_path = task_config.get('path') if isinstance(task_config, dict) else None
        if not isinstance(task_path, str) or not task_path:
            raise ValueError('Trial has no local task path')
        task = Path(task_path).name
        if task not in expected or task in seen:
            raise ValueError('Job contains an unexpected task or multiple attempts')
        seen.add(task)
        result_path = root / 'result.json'
        if not result_path.exists():
            failures += 1
            continue
        trial = read_object(result_path)
        identity, actual_identity = agent_identity(trial)
        identities.add(identity)
        if actual_identity[0] is not None:
            actual_identities.add(actual_identity)
        if len(identities) > 1 or len(actual_identities) > 1:
            raise ValueError('Job mixes Harbor Agents, Agent versions, or models')
        try:
            bound_path = trial['config']['task']['path']
            if not isinstance(bound_path, str) or Path(bound_path).name != task:
                raise ValueError('Harbor trial task binding differs')
            if not trial.get('finished_at') or trial.get('exception_info') is not None:
                raise ValueError('Harbor trial did not finish successfully')
            result = read_object(root / 'artifacts' / 'acworld-evidence' / 'result.json')
            reward = verifier.validate_result(result, expected[task])
            recorded = checked_reward(read_object(root / 'verifier' / 'reward.json'))
            harbor_reward = checked_reward(trial['verifier_result']['rewards'])
            if recorded != reward or harbor_reward != reward:
                raise ValueError('Harbor reward differs from trusted original score')
            scores.append(reward)
        except (OSError, ValueError, TypeError, KeyError):
            failures += 1
    if not seen:
        raise ValueError('No matching Harbor trials')
    job_result = read_object(job / 'result.json') if (job / 'result.json').exists() else {}
    stats = job_result.get('stats', {})
    if not isinstance(stats, dict):
        raise ValueError('Invalid Harbor job statistics')
    terminal = (bool(job_result.get('finished_at')) and job_result.get('n_total_trials') == TASK_COUNT
                and stats.get('n_completed_trials') == TASK_COUNT
                and all(stats.get(key) == 0 for key in
                        ('n_errored_trials', 'n_running_trials', 'n_pending_trials', 'n_cancelled_trials')))
    complete = terminal and len(seen) == TASK_COUNT and failures == 0 and len(scores) == TASK_COUNT
    declared, model_id, actual_name, actual_version = None, None, None, None
    if identities:
        declared, _import_path, model_id = next(iter(identities))
    if actual_identities:
        actual_name, actual_version, _actual_model = next(iter(actual_identities))
    if declared == 'oracle':
        execution_kind, model_id = 'reference_validation', None
    elif declared == 'nop':
        execution_kind, model_id = 'control_validation', None
    else:
        execution_kind = 'agent_evaluation' if declared is not None else None
    return {
        'schema': 'acworld.harbor-summary.v1', 'agent_version': agent_version,
        'execution_kind': execution_kind, 'agent_name': actual_name or declared,
        'harbor_agent_version': actual_version, 'model_id': model_id,
        'task_set_version': manifest['task_set_version'],
        'task_manifest_sha256': manifest['task_manifest_sha256'],
        'baseline_commit': manifest['baseline_commit'], 'scorer_version': scorer_version,
        'harbor_version': manifest['harbor_version'],
        'status': 'complete' if complete else 'incomplete', 'task_count': TASK_COUNT,
        'scored_task_count': len(scores), 'observed_task_count': len(seen),
        'failed_task_count': failures, 'job_finished': bool(job_result.get('finished_at')),
        'strict_success_rate': sum(row['reward'] for row in scores) / TASK_COUNT if complete else None,
        'mean_partial_credit': sum(row['capability_score'] for row in scores) / TASK_COUNT if complete else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--build-manifest', type=Path, required=True)
    parser.add_argument('--agent-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.job, args.build_manifest, args.agent_version)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'scored_task_count': result['scored_task_count']}))


if __name__ == '__main__':
    main()
