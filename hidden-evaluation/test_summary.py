"""Offline summary tests using synthetic task IDs and scores."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('harbor_summary_contract', HERE / 'summarize.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    save(path, value)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='acworld-summary-contract-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root / 'build'
        self.job = self.root / 'job'
        self.manifest_path = self.build / 'build-manifest.json'
        self.frozen_path = self.build / 'task-manifest.json'
        scorer = 'ACWorld v1.0.0 @ ' + summary.verifier.BASELINE_COMMIT
        frozen = {
            'version': 'contract-v1', 'baseline_commit': summary.verifier.BASELINE_COMMIT,
            'scorer_version': scorer,
            'tasks': [{'task_id': f'synthetic-{index:02d}'} for index in range(20)],
        }
        save(self.frozen_path, frozen)
        digest = hashlib.sha256(self.frozen_path.read_bytes()).hexdigest()
        self.rows = []
        for index, row in enumerate(frozen['tasks']):
            expected = {
                'directory': f'case-{index:02d}', 'task_id': row['task_id'],
                'task_set_version': frozen['version'], 'task_manifest_sha256': digest,
                'baseline_commit': frozen['baseline_commit'], 'scorer_version': scorer,
            }
            self.rows.append(expected)
            trial = self.trial(index)
            task = {'path': str(self.build / 'tasks' / expected['directory'])}
            save(trial / 'config.json', {'task': task, 'trial_name': trial.name})
            score = 1.0 if index % 2 == 0 else 0.5
            strict = index % 2 == 0
            reward = {'reward': int(strict), 'capability_score': score}
            result = {
                **expected, 'schema': summary.verifier.SCHEMA, 'finalized': True,
                'status': 'scored', 'capability_score': score, 'strict_success': strict,
                'submitted_decisions': 2, 'interruption_reason': None,
            }
            save(trial / 'artifacts/acworld-evidence/result.json', result)
            save(trial / 'verifier/reward.json', reward)
            save(trial / 'result.json', {
                'config': {'task': task, 'agent': {'name': 'oracle', 'import_path': None, 'model_name': None}},
                'agent_info': {'name': 'oracle', 'version': '1.0.0', 'model_info': None},
                'finished_at': '2026-01-01T00:00:00Z', 'exception_info': None,
                'verifier_result': {'rewards': reward},
            })
        save(self.manifest_path, {
            'schema': 'acworld.harbor-build.v1', 'harbor_version': '0.22.0',
            'task_set_version': frozen['version'], 'task_manifest_sha256': digest,
            'baseline_commit': frozen['baseline_commit'], 'tasks': self.rows,
        })
        save(self.job / 'result.json', {
            'finished_at': '2026-01-01T00:01:00Z', 'n_total_trials': 20,
            'stats': {'n_completed_trials': 20, 'n_errored_trials': 0,
                      'n_running_trials': 0, 'n_pending_trials': 0, 'n_cancelled_trials': 0},
        })

    def trial(self, index):
        return self.job / f'case-{index:02d}__attempt'

    def export(self):
        return summary.summarize(self.job, self.manifest_path, 'submitted-agent-version')

    def assert_no_means(self, result):
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['strict_success_rate'])
        self.assertIsNone(result['mean_partial_credit'])

    def set_agent(self, name, model=None):
        for index in range(20):
            def change(value):
                value['config']['agent']['name'] = name
                value['config']['agent']['model_name'] = model
                value['agent_info']['name'] = name
            mutate(self.trial(index) / 'result.json', change)

    def test_complete_twenty_trials_have_exact_averages(self):
        result = self.export()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['task_count'], 20)
        self.assertEqual(result['scored_task_count'], 20)
        self.assertEqual(result['strict_success_rate'], 0.5)
        self.assertEqual(result['mean_partial_credit'], 0.75)

    def test_oracle_is_reference_validation_not_model_evaluation(self):
        self.set_agent('oracle', 'unused/model-label')
        result = self.export()
        self.assertEqual(result['execution_kind'], 'reference_validation')
        self.assertEqual(result['agent_name'], 'oracle')
        self.assertIsNone(result['model_id'])
        self.assertEqual(result['agent_version'], 'submitted-agent-version')

    def test_nop_is_control_validation(self):
        self.set_agent('nop')
        result = self.export()
        self.assertEqual(result['execution_kind'], 'control_validation')
        self.assertIsNone(result['model_id'])

    def test_regular_agent_is_agent_evaluation_with_selected_model(self):
        self.set_agent('test-agent', 'provider/model-a')
        result = self.export()
        self.assertEqual(result['execution_kind'], 'agent_evaluation')
        self.assertEqual(result['model_id'], 'provider/model-a')

    def test_custom_agent_may_have_no_model_id(self):
        self.set_agent('test-agent')
        result = self.export()
        self.assertEqual(result['execution_kind'], 'agent_evaluation')
        self.assertIsNone(result['model_id'])

    def test_missing_task_produces_no_means(self):
        shutil.rmtree(self.trial(19))
        result = self.export()
        self.assert_no_means(result)
        self.assertEqual(result['scored_task_count'], 19)

    def test_missing_trial_result_produces_no_means(self):
        (self.trial(19) / 'result.json').unlink()
        result = self.export()
        self.assert_no_means(result)
        self.assertEqual(result['failed_task_count'], 1)

    def test_unfinished_job_produces_no_means_even_with_twenty_scores(self):
        mutate(self.job / 'result.json', lambda value: value.update(finished_at=None))
        result = self.export()
        self.assert_no_means(result)
        self.assertEqual(result['scored_task_count'], 20)

    def test_duplicate_attempt_is_rejected(self):
        shutil.copytree(self.trial(0), self.job / 'extra-attempt')
        with self.assertRaisesRegex(ValueError, 'multiple attempts'):
            self.export()

    def test_unexpected_task_is_rejected(self):
        mutate(self.trial(0) / 'config.json', lambda value: value['task'].update(path='/other/task'))
        with self.assertRaisesRegex(ValueError, 'unexpected task'):
            self.export()

    def test_mixed_agents_are_rejected(self):
        def change(value):
            value['config']['agent']['name'] = 'test-agent'
            value['agent_info']['name'] = 'test-agent'
        mutate(self.trial(0) / 'result.json', change)
        with self.assertRaisesRegex(ValueError, 'mixes Harbor'):
            self.export()

    def test_mixed_models_are_rejected(self):
        self.set_agent('test-agent', 'provider/model-a')
        mutate(self.trial(19) / 'result.json', lambda value: value['config']['agent'].update(model_name='provider/model-b'))
        with self.assertRaisesRegex(ValueError, 'mixes Harbor'):
            self.export()

    def test_mixed_actual_versions_are_rejected(self):
        mutate(self.trial(19) / 'result.json', lambda value: value['agent_info'].update(version='different'))
        with self.assertRaisesRegex(ValueError, 'mixes Harbor'):
            self.export()

    def test_forged_reward_file_cannot_raise_suite_average(self):
        save(self.trial(1) / 'verifier/reward.json', {'reward': 1, 'capability_score': 1.0})
        self.assert_no_means(self.export())

    def test_boolean_reward_is_rejected_despite_numeric_equality(self):
        save(self.trial(0) / 'verifier/reward.json', {'reward': True, 'capability_score': 1.0})
        self.assert_no_means(self.export())

    def test_harbor_result_reward_must_match_too(self):
        mutate(self.trial(1) / 'result.json', lambda value: value['verifier_result'].update(
            rewards={'reward': 1, 'capability_score': 1.0}))
        self.assert_no_means(self.export())

    def test_trial_exception_prevents_means_even_if_reward_exists(self):
        mutate(self.trial(0) / 'result.json', lambda value: value.update(exception_info={'exception_type': 'TestError'}))
        self.assert_no_means(self.export())

    def test_incomplete_service_evidence_prevents_means(self):
        mutate(self.trial(0) / 'artifacts/acworld-evidence/result.json', lambda value: value.update(
            status='incomplete', capability_score=None, strict_success=None))
        self.assert_no_means(self.export())

    def test_truncated_manifest_is_rejected(self):
        mutate(self.manifest_path, lambda value: value.update(tasks=value['tasks'][:1]))
        with self.assertRaisesRegex(ValueError, 'exactly 20'):
            self.export()

    def test_duplicate_manifest_directory_is_rejected(self):
        mutate(self.manifest_path, lambda value: value['tasks'][1].update(directory=value['tasks'][0]['directory']))
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.export()

    def test_duplicate_manifest_task_id_is_rejected(self):
        mutate(self.manifest_path, lambda value: value['tasks'][1].update(task_id=value['tasks'][0]['task_id']))
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.export()

    def test_build_inventory_must_match_original_task_ids(self):
        mutate(self.manifest_path, lambda value: value['tasks'][0].update(task_id='unknown-task'))
        with self.assertRaisesRegex(ValueError, 'task IDs differ'):
            self.export()

    def test_case_bindings_must_match_top_level_manifest(self):
        mutate(self.manifest_path, lambda value: value['tasks'][0].update(task_set_version='other'))
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            self.export()

    def test_original_manifest_bytes_are_hash_verified(self):
        self.frozen_path.write_text(self.frozen_path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.export()


if __name__ == '__main__':
    unittest.main(verbosity=2)
