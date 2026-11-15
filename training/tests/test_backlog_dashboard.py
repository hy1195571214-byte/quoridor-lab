"""Exercise backlog mode through the Dashboard API without provider connections."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from quoridor_lab.dashboard import CampaignConfig, ResumeConfig, RunConfig, campaign_runs, create_dashboard


class OfflineExpert:
    cache_hits = 0
    network_requests = 0

    def __init__(self, **options):
        self.request_gate = options["request_gate"]
        self.cache = {}

    def close(self):
        pass


def saved_run(root, run_id="a" * 32):
    folder = root / run_id
    folder.mkdir()
    (folder / "pending").mkdir()
    run = {"id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "status": "paused",
           "config": RunConfig(red="master", blue="master", games=3, concurrent_games=2,
                               games_per_connection=1, use_cache=False).model_dump()}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run_id, folder


class BacklogDashboardTests(unittest.TestCase):
    def test_only_formal_campaigns_enable_drain_by_default(self):
        self.assertFalse(RunConfig().drain_pending)
        self.assertIsNone(ResumeConfig().drain_pending)
        self.assertEqual(ResumeConfig(drain_pending=False).model_dump(exclude_none=True),
                         {"drain_pending": False})
        campaigns = list(campaign_runs(CampaignConfig(games_per_matchup=2)))
        self.assertEqual(len(campaigns), 25)
        self.assertTrue(all(config["drain_pending"] for _, config in campaigns))

    def test_resume_enables_drain_persists_setting_and_passes_it_to_collector(self):
        called = threading.Event()
        captured = {}

        def capture_collection(expert, **settings):
            captured.update(settings)
            called.set()
            return {"network_requests": 0, "time_limit_reached": False,
                    "error": {"kind": "stopped", "message": "End offline test"}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, folder = saved_run(root)
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost",
                                   expert_factory=OfflineExpert)
            with patch("quoridor_lab.dashboard.collect", side_effect=capture_collection), TestClient(app) as client:
                response = client.post(f"/api/runs/{run_id}/resume", json={"drain_pending": True})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(called.wait(2))
                deadline = time.monotonic() + 2
                while (detail := client.get(f"/api/runs/{run_id}").json())["status"] == "running":
                    self.assertLess(time.monotonic(), deadline, "Offline collector failed to finish")
                    time.sleep(0.01)
                self.assertEqual(detail["status"], "paused")
                self.assertTrue(detail["config"]["drain_pending"])
                self.assertTrue(captured["resume"])
                self.assertTrue(captured["drain_pending"])
                self.assertTrue(captured["defer_illegal_responses"])
                self.assertEqual(len(captured["request_lanes"]), 2)
                self.assertEqual(captured["output"], folder)
                self.assertTrue(json.loads((folder / "run.json").read_text(encoding="utf-8"))["config"]["drain_pending"])

    def test_details_show_only_the_retry_state_for_the_current_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, folder = saved_run(root)
            retry = {"ply": 1, "count": 3, "retry_at": "2026-11-16T00:00:00+00:00", "delay_seconds": 120}
            for number, retry_ply in ((1, 1), (2, 0)):
                item = {"episode_id": f"checkpoint-{number}", "game_number": number, "moves": ["e2"],
                        "config": {"remote_sides": ["master", "master"]},
                        "deferred_error": {"kind": "illegal_response"},
                        "retry_state": {**retry, "ply": retry_ply}}
                (folder / "pending" / f"{number}.json").write_text(json.dumps(item), encoding="utf-8")
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost",
                                   expert_factory=OfflineExpert)
            with TestClient(app) as client:
                response = client.get(f"/api/runs/{run_id}")
                self.assertEqual(response.status_code, 200)
                active = response.json()["active"]
                self.assertEqual(active[0]["retry_state"], retry)
                self.assertNotIn("retry_state", active[1])


if __name__ == "__main__":
    unittest.main()
