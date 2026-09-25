import json
import tempfile
import unittest
from pathlib import Path

from summary.luna_v1.batch import BatchClient, extract_one_completed, valid_batch_id
from summary.luna_v1.ledger import Ledger


class _Response:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, _cap):
        return b'{"id":"batch_abc123","status":"validating"}'


class _Opener:
    def __init__(self):
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        return _Response()


class LedgerTests(unittest.TestCase):
    def test_remote_batch_id_accepts_live_and_documented_forms(self):
        self.assertTrue(valid_batch_id("batch-1790364440-Gq1lAnlrV1xmZGmroHeB"))
        self.assertTrue(valid_batch_id("batch_abc123"))
        self.assertFalse(valid_batch_id("batch-../../other"))
        self.assertFalse(valid_batch_id("batch_"))

    def test_single_flight_restart_and_unknown_charge_hold_weekly_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = dict(semantic_key="a" * 64, source_sha256="b" * 64,
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1", max_cost_microusd=90_000)
            one = Ledger(root)
            first = one.reserve(output_dir=root / "meeting-one", **args)
            self.assertEqual(first.kind, "new")
            second = one.reserve(output_dir=root / "meeting-two", **args)
            self.assertEqual(second.kind, "pending")
            self.assertEqual(second.job_id, first.job_id)
            self.assertEqual(len(one.consumers("a" * 64)), 2)
            self.assertTrue(one.mark_submitting(first.job_id))
            self.assertFalse(one.mark_submitting(first.job_id))
            one.submission_result(first.job_id, remote_id=None, error_code="transport_unknown")
            one.close()
            restarted = Ledger(root)
            self.assertEqual(restarted.get(first.job_id)["status"], "submission_unknown")
            self.assertFalse(restarted.mark_submitting(first.job_id))
            # Eleven such unknown submissions would exceed the one shared USD,
            # even if each uses another key and output directory.
            for index in range(1, 11):
                decision = restarted.reserve(
                    semantic_key=f"{index:064x}", source_sha256="b" * 64,
                    output_dir=root / f"meeting-{index}", credential_id=f"key-{index}",
                    credential_version=1, workspace_id="workspace-1",
                    max_cost_microusd=90_000,
                )
                self.assertEqual(decision.kind, "new")
            blocked = restarted.reserve(
                semantic_key="f" * 64, source_sha256="b" * 64,
                output_dir=root / "over", credential_id="key-12", credential_version=1,
                workspace_id="workspace-1", max_cost_microusd=90_000,
            )
            self.assertEqual(blocked.reason, "weekly_budget_exceeded")

    def test_exact_batch_shape_and_custom_id_result_mapping(self):
        opener = _Opener()
        reply = BatchClient("fictional-test-token", opener=opener).submit("summary-001", {"messages": [{"role": "user", "content": "тест"}]})
        self.assertEqual(reply.status_code, 202)
        request = opener.calls[0][0]
        payload = json.loads(request.data)
        self.assertEqual(list(payload)[:4], ["endpoint", "model", "provider", "completion_window"])
        self.assertEqual(payload["model"], "openai/gpt-6-luna:batch")
        self.assertEqual(payload["provider"], {"only": ["openai"]})
        self.assertEqual(payload["requests"][0]["custom_id"], "summary-001")
        batch = {"status": "completed", "usage": {"cost": 0.001}, "results": [
            {"custom_id": "someone-else", "response": {"status_code": 200, "body": {"bad": True}}},
            {"custom_id": "summary-001", "response": {"status_code": 200, "body": {"choices": []}}},
        ]}
        body, usage = extract_one_completed(batch, "summary-001")
        self.assertEqual(body, {"choices": []})
        self.assertEqual(usage["cost"], 0.001)


if __name__ == "__main__":
    unittest.main()
