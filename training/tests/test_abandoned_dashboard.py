"""Check abandoned records remain visible without consuming the game target."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from quoridor_lab.dashboard import create_dashboard
from tests.test_abandoned_collection import archived_record
from tests.test_backlog_collection import NEAR_WIN, VirtualClock, checkpoint
from tests.test_backlog_dashboard import OfflineExpert, saved_run


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AbandonedDashboardTests(unittest.TestCase):
    def test_abandoned_games_do_not_count_toward_target_or_duplicate_pending_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, folder = saved_run(root)
            clock = VirtualClock()
            completed_path = checkpoint(folder, clock, "completed", moves=NEAR_WIN + ["e9"], number=1)
            completed = archived_record(json.loads(completed_path.read_text(encoding="utf-8")), clock)
            completed.update(status="completed", winner=1, finished_reason="goal")
            write_rows(folder / "episodes.jsonl", [completed])
            completed_path.unlink()
            abandoned_path = checkpoint(folder, clock, "abandoned", number=2)
            abandoned = archived_record(json.loads(abandoned_path.read_text(encoding="utf-8")), clock)
            write_rows(folder / "abandoned-games.jsonl", [abandoned])
            # The archived game's stale checkpoint simulates a crash before unlink.
            checkpoint(folder, clock, "live", moves=NEAR_WIN[:2], number=3)
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost:4173",
                                   expert_factory=OfflineExpert)
            with TestClient(app) as client:
                response = client.get(f"/api/runs/{run_id}")
                self.assertEqual(response.status_code, 200)
                detail = response.json()
                self.assertEqual(detail["config"]["games"], 3)
                self.assertEqual(detail["games_saved"], 1)
                self.assertEqual(detail["completed"], 1)
                self.assertEqual(detail["truncated"], 0)
                self.assertEqual(detail["abandoned"], 1)
                self.assertEqual(detail["abandoned_samples"], 14)
                self.assertEqual(detail["samples_saved"], 15 + 14 + 2)
                self.assertEqual([g["game_number"] for g in detail["active"]], [3])
                self.assertEqual([g["episode_id"] for g in detail["games"]], ["completed"])

    def test_abandoned_pagination_replay_and_download_preserve_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, folder = saved_run(root)
            clock = VirtualClock()
            abandoned = []
            for number, moves in ((3, NEAR_WIN), (1, NEAR_WIN[:2]), (2, NEAR_WIN[:4])):
                path = checkpoint(folder, clock, f"abandoned-{number}", moves=moves, number=number)
                abandoned.append(archived_record(json.loads(path.read_text(encoding="utf-8")), clock))
                path.unlink()
            write_rows(folder / "abandoned-games.jsonl", abandoned)
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost:4173",
                                   expert_factory=OfflineExpert)
            with TestClient(app) as client:
                response = client.get(f"/api/runs/{run_id}/abandoned", params={"offset": 1, "limit": 1})
                self.assertEqual(response.status_code, 200)
                page = response.json()
                self.assertEqual(page["total"], 3)
                [game] = page["games"]
                self.assertEqual(game["game_number"], 2)
                self.assertEqual(game["status"], "abandoned")
                self.assertEqual(game["moves"], NEAR_WIN[:4])
                query = parse_qs(urlparse(game["replay"]).query)
                self.assertEqual(query["moves"], [",".join(NEAR_WIN[:4])])
                self.assertEqual(query["ai"], ["master,master"])
                self.assertEqual(json.loads(query["result"][0]),
                                 {"winner": None, "reason": "illegal_response_retry_limit"})
                download = client.get(f"/api/runs/{run_id}/files/abandoned-games.jsonl")
                self.assertEqual(download.status_code, 200)
                self.assertEqual([json.loads(line) for line in download.text.splitlines()], abandoned)
                self.assertIn("application/x-ndjson", download.headers["content-type"])

    def test_resume_omitted_threshold_preserves_it_and_explicit_null_disables_it(self):
        for body, expected in (({}, 4), ({"abandon_after_retries": None}, None)):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_id, folder = saved_run(root)
                run_path = folder / "run.json"
                run = json.loads(run_path.read_text(encoding="utf-8"))
                run["config"].update(drain_pending=True, abandon_after_retries=4)
                run_path.write_text(json.dumps(run), encoding="utf-8")
                called = threading.Event()
                captured = {}

                def capture_collection(expert, **settings):
                    captured.update(settings)
                    called.set()
                    return {"network_requests": 0, "time_limit_reached": False,
                            "error": {"kind": "stopped", "message": "Offline test finished"}}

                app = create_dashboard(root, data_dir=root, replay_base="http://localhost:4173",
                                       expert_factory=OfflineExpert)
                with patch("quoridor_lab.dashboard.collect", side_effect=capture_collection), TestClient(app) as client:
                    response = client.post(f"/api/runs/{run_id}/resume", json=body)
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(called.wait(2))
                    deadline = time.monotonic() + 2
                    while (detail := client.get(f"/api/runs/{run_id}").json())["status"] == "running":
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.01)
                    self.assertEqual(detail["status"], "paused")
                    self.assertEqual(detail["config"]["abandon_after_retries"], expected)
                    self.assertEqual(captured["abandon_after_retries"], expected)
                    self.assertEqual(json.loads(run_path.read_text(encoding="utf-8"))["config"]["abandon_after_retries"],
                                     expected)


if __name__ == "__main__":
    unittest.main()
