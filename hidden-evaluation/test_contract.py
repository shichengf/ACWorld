"""Offline contract checks for the Harbor adapter. No model or network calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load_module("acworld_harbor_server_contract", HERE / "adapter/server.py")
verify = load_module("acworld_harbor_verify_contract", HERE / "templates/tests/verify.py")


def session_with_request():
    """Construct transport state without importing the frozen task package."""
    session = server.Session.__new__(server.Session)
    session.condition = threading.Condition()
    session.closed = False
    session.done = False
    session.submitted = 0
    session.outcome = None
    session.response = None
    session.request = {
        "request_id": "request-current",
        "business_request": {"system_prompt": "system", "user_prompt": "user"},
    }
    # A nonexistent marker path supports the sidecar's filesystem seal check.
    session.evidence = HERE / "__contract_nonexistent_evidence__"
    return session


class MemorySocket:
    """Exercise BaseHTTPRequestHandler's actual HTTP parser without listening."""

    def __init__(self, request: bytes):
        self.input = io.BytesIO(request)
        self.output = bytearray()

    def makefile(self, *_args, **_kwargs):
        return self.input

    def sendall(self, data):
        self.output.extend(data)

    def settimeout(self, _timeout):
        pass


def http_request(session, method="GET", path="/observe", body=b"", headers=None):
    fields = {"Host": "acworld", "Content-Length": str(len(body)), **(headers or {})}
    head = f"{method} {path} HTTP/1.1\r\n"
    head += "".join(f"{key}: {value}\r\n" for key, value in fields.items())
    sock = MemorySocket(head.encode("ascii") + b"\r\n" + body)
    server.handler(session)(sock, ("127.0.0.1", 12345), SimpleNamespace())
    raw_head, raw_body = bytes(sock.output).split(b"\r\n\r\n", 1)
    status_line, raw_headers = raw_head.split(b"\r\n", 1)
    response_headers = BytesParser().parsebytes(raw_headers + b"\r\n\r\n")
    if response_headers.get_content_type() != "application/json":
        raise AssertionError("Adapter response must be JSON")
    return int(status_line.split()[1]), json.loads(raw_body)


class SessionContractTests(unittest.TestCase):
    def test_stale_missing_and_duplicate_ids(self):
        session = session_with_request()
        self.assertFalse(session.submit(None, {}))
        self.assertFalse(session.submit("request-old", {}))
        self.assertEqual(session.submitted, 0)
        self.assertTrue(session.submit("request-current", {"actions": []}))
        self.assertFalse(session.submit("request-current", {"actions": []}))
        self.assertEqual(session.submitted, 1)
        self.assertEqual(json.loads(session.response), {"actions": []})

    def test_concurrent_duplicates_are_accepted_once(self):
        session = session_with_request()
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(
                lambda _: session.submit("request-current", {"actions": []}), range(40)
            ))
        self.assertEqual(sum(accepted), 1)
        self.assertEqual(session.submitted, 1)

    def test_closed_done_and_no_request_reject_submissions(self):
        for state in ("closed", "done", "no_request"):
            with self.subTest(state=state):
                session = session_with_request()
                if state == "no_request":
                    session.request = None
                else:
                    setattr(session, state, True)
                self.assertFalse(session.submit("request-current", {}))
                self.assertEqual(session.submitted, 0)

    def test_consumed_id_does_not_match_next_request(self):
        session = session_with_request()
        self.assertTrue(session.submit("request-current", {}))
        session.response = None
        session.request = {**session.request, "request_id": "request-next"}
        self.assertFalse(session.submit("request-current", {}))
        self.assertTrue(session.submit("request-next", {}))
        self.assertEqual(session.submitted, 2)

    def test_stop_marker_rejects_submit_before_polling_sets_closed(self):
        session = session_with_request()
        with mock.patch.object(Path, "exists", return_value=True):
            self.assertFalse(session.submit("request-current", {}))
        self.assertFalse(session.closed)
        self.assertEqual(session.submitted, 0)

    def test_stop_marker_prevents_a_new_exchange(self):
        session = session_with_request()
        inference = ModuleType("agents.inference")
        inference.ChannelTransportError = type("ChannelTransportError", (RuntimeError,), {})
        inference.BusinessDecisionResponseV1 = lambda **kwargs: SimpleNamespace(**kwargs)
        agents = ModuleType("agents")
        agents.__path__ = []
        agents.inference = inference
        with mock.patch.dict(sys.modules, {"agents": agents, "agents.inference": inference}), \
                mock.patch.object(Path, "exists", return_value=True), \
                self.assertRaises(inference.ChannelTransportError):
            session.exchange(system_prompt="system", user_prompt="user")
        self.assertEqual(session.request["request_id"], "request-current")

    def test_worker_import_failure_sets_done_without_score(self):
        session = session_with_request()
        with mock.patch.object(session, "execute_runtime", side_effect=ImportError("dependency")), \
                mock.patch.object(Path, "write_text"):
            session.execute()
        self.assertTrue(session.done)
        self.assertEqual(session.outcome["status"], "error")
        self.assertIsNone(session.outcome["capability_score"])
        self.assertIsNone(session.outcome["strict_success"])
        self.assertEqual(session.observe(), {"status": "finished"})

    def test_observe_exposes_only_current_business_request(self):
        session = session_with_request()
        self.assertEqual(session.observe(), {"status": "awaiting_decision", **session.request})
        session.submit("request-current", {})
        self.assertEqual(session.observe(), {"status": "running"})
        session.closed = True
        self.assertEqual(session.observe(), {"status": "finished"})

    def _exchange(self, submit_response):
        session = session_with_request()
        session.request = None
        inference = ModuleType("agents.inference")
        inference.ChannelTransportError = type("ChannelTransportError", (RuntimeError,), {})
        inference.BusinessDecisionResponseV1 = lambda **kwargs: SimpleNamespace(**kwargs)
        agents = ModuleType("agents")
        agents.__path__ = []
        agents.inference = inference
        result = {}

        def worker():
            try:
                result["response"] = session.exchange(system_prompt="system", user_prompt="user")
            except Exception as exc:
                result["error"] = exc

        with mock.patch.dict(sys.modules, {"agents": agents, "agents.inference": inference}):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            with session.condition:
                ready = session.condition.wait_for(lambda: session.request is not None, timeout=2)
            self.assertTrue(ready)
            if submit_response:
                self.assertTrue(session.submit(session.request["request_id"], {"actions": []}))
            else:
                with session.condition:
                    session.closed = True
                    session.condition.notify_all()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        return session, result, inference.ChannelTransportError

    def test_exchange_consumes_one_response(self):
        session, result, _error = self._exchange(True)
        self.assertNotIn("error", result)
        self.assertEqual(json.loads(result["response"].content), {"actions": []})
        self.assertIsNone(session.request)
        self.assertIsNone(session.response)

    def test_sealing_without_response_is_transport_failure(self):
        session, result, error = self._exchange(False)
        self.assertIsInstance(result.get("error"), error)
        self.assertNotIn("response", result)
        self.assertEqual(session.submitted, 0)


class HttpContractTests(unittest.TestCase):
    def test_observe_and_matching_request_id(self):
        session = session_with_request()
        status, observation = http_request(session)
        self.assertEqual(status, 200)
        request_id = observation["request_id"]
        body = b'{"actions": []}'
        for header in ({}, {"X-ACWorld-Request-ID": "stale"}):
            status, _ = http_request(session, "POST", "/submit", body, header)
            self.assertEqual(status, 409)
        headers = {"X-ACWorld-Request-ID": request_id}
        self.assertEqual(http_request(session, "POST", "/submit", body, headers),
                         (200, {"status": "accepted"}))
        self.assertEqual(http_request(session, "POST", "/submit", body, headers)[0], 409)
        self.assertEqual(session.submitted, 1)

    def test_sealed_http_submission_is_rejected(self):
        session = session_with_request()
        session.closed = True
        status, response = http_request(session, "POST", "/submit", b"{}",
                                        {"X-ACWorld-Request-ID": "request-current"})
        self.assertEqual((status, response), (409, {"error": "request_not_current"}))
        self.assertEqual(session.submitted, 0)

    def test_stop_marker_rejects_http_submission_before_seal_poll(self):
        session = session_with_request()
        with mock.patch.object(Path, "exists", return_value=True):
            status, response = http_request(session, "POST", "/submit", b"{}",
                                            {"X-ACWorld-Request-ID": "request-current"})
        self.assertEqual((status, response), (409, {"error": "request_not_current"}))
        self.assertEqual(session.submitted, 0)

    def test_private_scoring_and_finalization_are_not_routes(self):
        for method in ("GET", "POST"):
            for path in ("/scoring", "/score", "/finalize", "/private",
                         "/private/manifest.json", "/acworld-evidence/result.json",
                         "/../private/manifest.json", "/observe?private=true"):
                with self.subTest(method=method, path=path):
                    status, response = http_request(session_with_request(), method, path, b"{}")
                    self.assertEqual((status, response), (404, {"error": "unknown_endpoint"}))

    def test_malformed_payloads_are_rejected_without_submission(self):
        bad_bodies = (b"", b"{", b"[]", b'"text"', b"null", b'{"x": NaN}',
                      b'{"x": Infinity}', b'{"x": 1e999}', b'{"x": "\\ud800"}', b"\xff")
        for body in bad_bodies:
            with self.subTest(body=body):
                session = session_with_request()
                status, response = http_request(session, "POST", "/submit", body,
                                                {"X-ACWorld-Request-ID": "request-current"})
                self.assertEqual((status, response), (400, {"error": "invalid_decision_json"}))
                self.assertEqual(session.submitted, 0)

    def test_unsupported_request_lengths_and_encoding_are_rejected(self):
        for headers in ({"Content-Length": "-1"}, {"Content-Length": "not-an-int"},
                        {"Content-Length": str(server.MAX_BYTES + 1)},
                        {"Transfer-Encoding": "chunked"}):
            with self.subTest(headers=headers):
                session = session_with_request()
                self.assertEqual(http_request(session, "POST", "/submit", b"{}", headers)[0], 400)
                self.assertEqual(session.submitted, 0)


class FinalizationContractTests(unittest.TestCase):
    def _finalize(self, *, changed_manifest=False, worker_done=True):
        session = session_with_request()
        session.private = HERE / "__contract_nonexistent_private__"
        session.case = {"task_id": "contract-case"}
        session.manifest = {"version": "contract-v1", "baseline_commit": verify.BASELINE_COMMIT,
                            "scorer_version": "ACWorld v1.0.0 @ " + verify.BASELINE_COMMIT}
        original = b'{"fixture":"original"}'
        session.manifest_sha = hashlib.sha256(original).hexdigest()
        session.worker = mock.Mock()
        session.done = worker_done
        session.outcome = {"status": "scored", "capability_score": 1.0,
                           "strict_success": True, "interruption_reason": None}
        prepare = ModuleType("prepare")
        prepare.check_frozen = mock.Mock(return_value=session.manifest)
        final_bytes = b'{"fixture":"changed"}' if changed_manifest else original
        with mock.patch.dict(sys.modules, {"prepare": prepare}), \
                mock.patch.object(Path, "exists", return_value=True), \
                mock.patch.object(Path, "read_bytes", return_value=final_bytes), \
                mock.patch.object(server, "save") as save:
            session.finalize_when_requested()
        self.assertTrue(session.closed)
        session.worker.join.assert_called_once()
        save.assert_called_once()
        return save.call_args.args[1]

    def test_completed_original_result_is_sealed(self):
        result = self._finalize()
        self.assertTrue(result["finalized"])
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["capability_score"], 1.0)

    def test_manifest_change_disables_scoring(self):
        result = self._finalize(changed_manifest=True)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["interruption_reason"], "final_integrity_check_failed")
        self.assertIsNone(result["capability_score"])
        self.assertIsNone(result["strict_success"])

    def test_worker_still_running_cannot_export_a_score(self):
        result = self._finalize(worker_done=False)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["capability_score"])
        self.assertIsNone(result["strict_success"])


class VerifierContractTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "task_id": "contract-case", "task_set_version": "contract-v1",
            "task_manifest_sha256": "a" * 64, "baseline_commit": verify.BASELINE_COMMIT,
        }
        self.result = {
            **self.expected, "schema": verify.SCHEMA, "finalized": True, "status": "scored",
            "scorer_version": "ACWorld v1.0.0 @ " + verify.BASELINE_COMMIT,
            "capability_score": 0.75, "strict_success": False,
            "submitted_decisions": 2, "interruption_reason": None,
        }

    def test_original_scores_are_projected_without_recalculation(self):
        self.assertEqual(verify.validate_result(self.result, self.expected),
                         {"reward": 0, "capability_score": 0.75})
        self.result.update(strict_success=True, capability_score=1.0)
        self.assertEqual(verify.validate_result(self.result, self.expected),
                         {"reward": 1, "capability_score": 1.0})

    def test_incomplete_or_malformed_evidence_cannot_produce_reward(self):
        patches = (
            {"schema": "other"}, {"finalized": False}, {"finalized": 1},
            {"status": "incomplete"}, {"status": "error"},
            {"capability_score": None}, {"capability_score": float("nan")},
            {"capability_score": float("inf")}, {"capability_score": True},
            {"capability_score": -0.1}, {"capability_score": 1.1},
            {"strict_success": 1}, {"strict_success": None},
            {"submitted_decisions": -1}, {"submitted_decisions": True},
            {"interruption_reason": "interrupted"}, {"scorer_version": ""},
        )
        for patch in patches:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                verify.validate_result({**self.result, **patch}, self.expected)

    def test_result_bindings_must_match_expected(self):
        for field in ("task_id", "task_set_version", "baseline_commit", "task_manifest_sha256"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify.validate_result({**self.result, field: "different"}, self.expected)

    def test_expected_binding_must_be_complete_and_valid(self):
        for field in ("task_id", "task_set_version", "baseline_commit", "task_manifest_sha256"):
            expected = dict(self.expected)
            del expected[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                verify.validate_result(self.result, expected)
        for bad_hash in ("", "a" * 63, "a" * 65, "z" * 64, "A" * 64):
            with self.subTest(bad_hash=bad_hash), self.assertRaises(ValueError):
                verify.validate_result({**self.result, "task_manifest_sha256": bad_hash},
                                       {**self.expected, "task_manifest_sha256": bad_hash})
        with self.assertRaises(ValueError):
            verify.validate_result(self.result, {**self.expected, "baseline_commit": "different"})

    def test_expected_scorer_version_is_enforced_when_present(self):
        with self.assertRaises(ValueError):
            verify.validate_result(self.result, {**self.expected, "scorer_version": "different"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
