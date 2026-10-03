"""Trusted transport into the unchanged ACWorld scorer. No model API calls."""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

MAX_BYTES = 1024 * 1024


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def invalid_constant(value):
    raise ValueError('JSON numbers must be finite')


class Session:
    """Only actor-visible requests cross the container boundary."""

    def __init__(self, private: Path, case: Path, evidence: Path):
        sys.path.insert(0, str(private))
        from prepare import check_frozen
        self.manifest = check_frozen()
        self.private = private
        self.manifest_sha = hashlib.sha256((private / 'manifest.json').read_bytes()).hexdigest()
        self.case = json.loads(case.read_text())
        if self.case.get('task_manifest_sha256') != self.manifest_sha:
            raise ValueError('Task set does not match its build manifest')
        if self.case.get('task_id') not in {r['task_id'] for r in self.manifest['tasks']}:
            raise ValueError('Unknown frozen task')
        self.evidence = evidence
        evidence.mkdir(parents=True, exist_ok=True)
        if any(evidence.iterdir()):
            raise ValueError('An episode needs a fresh evidence directory')
        self.condition = threading.Condition()
        self.request = None
        self.response = None
        self.closed = False
        self.done = False
        self.submitted = 0
        self.outcome = None
        self.worker = threading.Thread(target=self.execute, daemon=True)
        self.worker.start()
        threading.Thread(target=self.finalize_when_requested, daemon=True).start()

    def exchange(self, *, system_prompt, user_prompt, decision_id=None):
        from agents.inference import BusinessDecisionResponseV1, ChannelTransportError
        with self.condition:
            if self.closed or (self.evidence / 'stop-requested').exists():
                raise ChannelTransportError('Agent session ended before a decision was submitted')
            self.request = {
                'request_id': uuid.uuid4().hex,
                'business_request': {'system_prompt': system_prompt, 'user_prompt': user_prompt},
            }
            # decision_id is an opaque Runtime identifier, not a scoring answer.
            if decision_id is not None:
                self.request['business_request']['decision_id'] = decision_id
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.response is not None or self.closed)
            if self.response is None:
                raise ChannelTransportError('Agent session ended before a decision was submitted')
            content = self.response
            self.response = None
            self.request = None
            self.condition.notify_all()
        return BusinessDecisionResponseV1(content=content, response_chars=len(content),
                                         response_sha256=hashlib.sha256(content.encode()).hexdigest())

    def observe(self):
        with self.condition:
            self.condition.wait_for(lambda: self.request is not None or self.done or self.closed, timeout=10)
            if self.done or self.closed:
                return {'status': 'finished'}
            if self.request is None or self.response is not None:
                return {'status': 'running'}
            return {'status': 'awaiting_decision', **self.request}

    def submit(self, request_id, decision):
        with self.condition:
            if (self.closed or (self.evidence / 'stop-requested').exists()
                    or self.done or self.request is None or self.response is not None
                    or request_id != self.request['request_id']):
                return False
            self.response = json.dumps(decision, ensure_ascii=False, allow_nan=False)
            self.submitted += 1
            self.condition.notify_all()
            return True

    def execute(self):
        try:
            self.execute_runtime()
        except Exception as exc:
            (self.evidence / 'internal-error.txt').write_text(traceback.format_exc())
            with self.condition:
                self.outcome = {'status': 'error', 'capability_score': None, 'strict_success': None,
                                'interruption_reason': type(exc).__name__}
                self.done = True
                self.condition.notify_all()

    def execute_runtime(self):
        from hidden_tasks import installed
        from episode.capability_runtime_registry import runtime_bundle_v2
        from episode.scenario import population_for_scenario
        from experiments.benchmark_plan import RunSpecV2, LOCAL_REFERENCE_MODEL_V2
        from experiments.benchmark_executor import BenchmarkExecutor

        session = self

        class Channel:
            supports_business_decisions = True
            supports_decision_evidence_context = True
            complete_business_decision = staticmethod(session.exchange)

        class HiddenRunSpec(RunSpecV2):
            def __post_init__(self):
                bundle = runtime_bundle_v2(self.task_id)
                population = population_for_scenario(bundle.scenario)
                if (self.suite != 'main' or self.task_id != session.case['task_id']
                        or self.evaluated_role != bundle.task.evaluated_role
                        or (self.buyers, self.merchants) != (len(population.buyers), len(population.merchants))):
                    raise ValueError('Run does not match frozen task')

        try:
            with installed():
                bundle = runtime_bundle_v2(self.case['task_id'])
                population = population_for_scenario(bundle.scenario)
                # This identifier selects a local injected channel, never a provider.
                run = HiddenRunSpec(suite='main', model_id=LOCAL_REFERENCE_MODEL_V2,
                                    task_id=self.case['task_id'], evaluated_role=bundle.task.evaluated_role,
                                    buyers=len(population.buyers), merchants=len(population.merchants))
                target = self.evidence / 'execution'
                target.mkdir()
                payload = BenchmarkExecutor(model_channels=lambda *_: Channel(),
                                            max_model_calls_per_run=48)(run, target)
                save(target / 'result.json', payload)
                metrics = payload['metrics']
                outcome = {'status': 'scored', 'capability_score': metrics['capability_score'],
                           'strict_success': metrics['strict_success'], 'interruption_reason': None}
        except Exception as exc:
            (self.evidence / 'internal-error.txt').write_text(traceback.format_exc())
            outcome = {'status': 'incomplete' if self.closed else 'error',
                       'capability_score': None, 'strict_success': None,
                       'interruption_reason': 'agent_stopped_before_completion' if self.closed
                       else type(exc).__name__}
        with self.condition:
            self.outcome = outcome
            self.done = True
            self.condition.notify_all()

    def finalize_when_requested(self):
        while not (self.evidence / 'stop-requested').exists():
            time.sleep(0.1)
        # Seal submissions even when Harbor could not stop the Agent container.
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.worker.join(timeout=60)
        outcome = self.outcome if self.done else {
            'status': 'error', 'capability_score': None, 'strict_success': None,
            'interruption_reason': 'executor_did_not_stop'}
        try:
            from prepare import check_frozen
            if not self.done:
                raise RuntimeError('Executor is still active')
            check_frozen()
            if hashlib.sha256((self.private / 'manifest.json').read_bytes()).hexdigest() != self.manifest_sha:
                raise RuntimeError('Frozen manifest changed during execution')
        except Exception:
            outcome = {'status': 'error', 'capability_score': None, 'strict_success': None,
                       'interruption_reason': 'final_integrity_check_failed'}
        save(self.evidence / 'result.json', {
            'schema': 'acworld.harbor-result.v1', 'finalized': True,
            'task_id': self.case['task_id'], 'task_manifest_sha256': self.manifest_sha,
            'task_set_version': self.manifest['version'],
            'baseline_commit': self.manifest['baseline_commit'],
            'scorer_version': self.manifest['scorer_version'],
            'submitted_decisions': self.submitted, **outcome})


def handler(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, code, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.send_json(200, {'status': 'ok'})
            elif self.path == '/observe':
                self.send_json(200, session.observe())
            else:
                self.send_json(404, {'error': 'unknown_endpoint'})

        def do_POST(self):
            if self.path != '/submit':
                self.send_json(404, {'error': 'unknown_endpoint'})
                return
            try:
                self.connection.settimeout(15)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BYTES or self.headers.get('Transfer-Encoding'):
                    raise ValueError('Unsupported request size or encoding')
                raw = self.rfile.read(size)
                decision = json.loads(raw, parse_constant=invalid_constant)
                if not isinstance(decision, dict):
                    raise ValueError('A decision must be an object')
                json.dumps(decision, ensure_ascii=False, allow_nan=False).encode('utf-8')
            except (ValueError, OSError, RecursionError):
                self.send_json(400, {'error': 'invalid_decision_json'})
                return
            if session.submit(self.headers.get('X-ACWorld-Request-ID'), decision):
                self.send_json(200, {'status': 'accepted'})
            else:
                self.send_json(409, {'error': 'request_not_current'})
    return Handler


def main():
    os.umask(0o077)
    session = Session(Path(os.environ.get('ACWORLD_PRIVATE_ROOT', '/private')),
                      Path(os.environ.get('ACWORLD_CASE_PATH', '/case.json')),
                      Path(os.environ.get('ACWORLD_EVIDENCE_DIR', '/acworld-evidence')))
    server = ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('ACWORLD_PORT', '8765'))), handler(session))
    server.serve_forever()


if __name__ == '__main__':
    main()
