"""Diagnostic downloads expose saved error evidence through the local Dashboard."""
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from quoridor_lab.dashboard import create_dashboard
from tests.test_backlog_dashboard import OfflineExpert, saved_run


class CollectionErrorDownloadTests(unittest.TestCase):
    def test_reconsidered_events_do_not_double_count_unverified_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run_id,folder=saved_run(root)
            quality={'category':'review_e2','flags':['no_distance_progress']}
            records=[{'status':'accepted','quality':quality},
                     {'status':'replaced','quality':quality},
                     {'status':'confirmed','quality':quality},
                     {'status':'accepted','quality':{**quality,'independent_confirmation':True}}]
            (folder/'e2-events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            app=create_dashboard(root,data_dir=root,replay_base='http://localhost:4173',expert_factory=OfflineExpert)
            with TestClient(app) as client:
                health=client.get(f'/api/runs/{run_id}').json()['e2_health']
                self.assertEqual(health['unverified_reviews'],1)
                self.assertEqual(health['by_status'],{'accepted':2,'replaced':1,'confirmed':1})

    def test_existing_diagnostics_are_listed_and_downloaded_without_losing_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, folder = saved_run(root)
            diagnostics = {"provider_response": {"ok": True, "move": "e2", "correlationId": "request-1"},
                           "request": {"moves": "e2,e8,e3", "difficulty": "master", "correlationId": "request-1"}}
            payloads = {
                "rejected-moves.jsonl": json.dumps({"attempt": 1, "diagnostics": diagnostics}) + "\n",
                "deferred-games.jsonl": json.dumps({"episode_id": "example", "diagnostics": diagnostics}) + "\n",
                "last-error.json": json.dumps({"kind": "rate_limited", "diagnostics": {
                    "provider_response": {"error": "rate_limited", "retryAfterMs": 300000}}}),
                "e2-events.jsonl": json.dumps({'status':'quarantined','quality':{'category':'known_e2_fault_signature'},
                                               'diagnostics':diagnostics})+'\n',
            }
            for filename, content in payloads.items():
                (folder / filename).write_text(content, encoding="utf-8")
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost:4173",
                                   expert_factory=OfflineExpert)
            with TestClient(app) as client:
                detail = client.get(f"/api/runs/{run_id}").json()
                self.assertEqual(detail["diagnostic_files"], list(payloads))
                self.assertEqual(detail['e2_health']['by_status'], {'quarantined':1})
                self.assertEqual(detail['e2_health']['scope'], 'events_since_guard_deployment')
                self.assertEqual(detail['e2_health']['quality_version'],3)
                for filename, content in payloads.items():
                    with self.subTest(filename=filename):
                        response = client.get(f"/api/runs/{run_id}/files/{filename}")
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(json.loads(response.text), json.loads(content))
                        mime = "application/json" if filename.endswith(".json") else "application/x-ndjson"
                        self.assertIn(mime, response.headers["content-type"])

    def test_missing_diagnostics_are_not_listed_and_return_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, folder = saved_run(root)
            # A directory with a diagnostic filename must not count as a saved file.
            (folder / "rejected-moves.jsonl").mkdir()
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost:4173",
                                   expert_factory=OfflineExpert)
            with TestClient(app) as client:
                detail = client.get(f"/api/runs/{run_id}").json()
                self.assertEqual(detail["diagnostic_files"], [])
                for filename in ("rejected-moves.jsonl", "deferred-games.jsonl", "last-error.json"):
                    with self.subTest(filename=filename):
                        response = client.get(f"/api/runs/{run_id}/files/{filename}")
                        self.assertEqual(response.status_code, 404)

    def test_download_remains_limited_to_known_filenames_and_valid_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, folder = saved_run(root)
            (folder / "private-notes.json").write_text('{"private": true}', encoding="utf-8")
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost:4173",
                                   expert_factory=OfflineExpert)
            with TestClient(app) as client:
                self.assertEqual(client.get(f"/api/runs/{run_id}/files/private-notes.json").status_code, 422)
                self.assertEqual(client.get(f"/api/runs/{run_id}/files/%2E%2E%2Frun.json").status_code, 404)
                self.assertEqual(client.get("/api/runs/not-a-run/files/last-error.json").status_code, 404)
                self.assertEqual(client.get(f"/api/runs/{'b' * 32}/files/last-error.json").status_code, 404)


if __name__ == "__main__":
    unittest.main()
