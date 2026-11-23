"""A natural-completion target must survive cutoffs, resumes and concurrency."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from quoridor_lab.collect import collect
from quoridor_lab.dashboard import CampaignConfig, ResumeConfig, RunConfig, campaign_runs, create_dashboard
from quoridor_lab.opponents import RemoteError
from quoridor_lab.rules import State
from tests.test_backlog_collection import (
    NEAR_WIN, ScriptedExpert, VirtualClock, checkpoint, read_rows, virtual_scheduler,
)
from tests.test_backlog_dashboard import OfflineExpert, saved_run


def capped_checkpoint(root, clock, episode_id="will-truncate", *, number=1):
    path = checkpoint(root, clock, episode_id, moves=NEAR_WIN[:13], number=number)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["config"]["max_plies"] = 14
    path.write_text(json.dumps(saved), encoding="utf-8")
    return path


def terminal_record(path, status):
    saved = json.loads(path.read_text(encoding="utf-8"))
    state = State.from_moves(saved["moves"])
    return {"schema_version": 1, "episode_id": saved["episode_id"], "game_number": saved["game_number"],
            "config": saved["config"], "started_at": saved["started_at"], "finished_at": saved["started_at"],
            "status": status, "winner": state.winner,
            "finished_reason": "goal" if status == "completed" else "max_plies",
            "ply_count": len(saved["moves"]), "moves": saved["moves"], "final": state.to_dict()}


def run_completed(root, expert, *, use_lanes, **overrides):
    settings = dict(output=root, games=1, max_plies=32, resume=True,
                    remote_sides=("master", "master"), concurrent_games=1,
                    completed_only=True, progress=lambda _: None)
    if use_lanes:
        settings.update(request_lanes=[expert], games_per_connection=1,
                        defer_illegal_responses=True, drain_pending=True)
    settings.update(overrides)
    return collect(expert, **settings)


class CompletedTargetTests(unittest.TestCase):
    def assert_unique_trace(self, root):
        trace = read_rows(root / "transitions.jsonl")
        keys = [(row["episode_id"], row["ply"]) for row in trace]
        self.assertEqual(len(keys), len(set(keys)))
        return trace

    def test_truncated_record_is_preserved_and_replaced_until_natural_target_is_met(self):
        for use_lanes in (False, True):
            with self.subTest(use_lanes=use_lanes), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                clock = VirtualClock()
                with virtual_scheduler(clock):
                    path = capped_checkpoint(root, clock)
                    expert = ScriptedExpert(clock)
                    result = run_completed(root, expert, use_lanes=use_lanes)
                self.assertIsNone(result["error"])
                self.assertEqual(result["completed_games"], 1)
                self.assertEqual(result["target_games_finished"], 1)
                self.assertEqual(result["finished_games"], 2)
                self.assertEqual([g["status"] for g in result["episodes"]], ["truncated", "completed"])
                cutoff, complete = result["episodes"]
                self.assertEqual(cutoff["episode_id"], "will-truncate")
                self.assertEqual(cutoff["finished_reason"], "max_plies")
                self.assertEqual(cutoff["ply_count"], 14)
                self.assertIsNone(cutoff["winner"])
                self.assertNotEqual(complete["game_number"], cutoff["game_number"])
                self.assertFalse(path.exists())
                self.assertEqual(len(read_rows(root / "episodes.jsonl")), 2)
                self.assertEqual(json.loads((root / "episode-will-truncate.json").read_text(encoding="utf-8")), cutoff)
                trace = self.assert_unique_trace(root)
                for game in result["episodes"]:
                    steps = [r for r in trace if r["episode_id"] == game["episode_id"]]
                    self.assertEqual([r["notation"] for r in steps], game["moves"])

    def test_real_parallel_cutoff_releases_one_slot_without_excess_natural_completions(self):
        clock = VirtualClock()
        barrier = threading.Barrier(2, timeout=3)
        stopped = threading.Event()
        worker_ids = set()
        worker_lock = threading.Lock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capped_checkpoint(root, clock, number=1)
            checkpoint(root, clock, "old-winner", number=2)

            def before_choose(state):
                if state.history in (tuple(NEAR_WIN), tuple(NEAR_WIN[:13])):
                    with worker_lock:
                        worker_ids.add(threading.get_ident())
                    barrier.wait()

            lanes = [ScriptedExpert(clock, before_choose=before_choose) for _ in range(2)]
            for lane in lanes:
                lane.request_gate = SimpleNamespace(stopped=stopped)
            result = run_completed(root, lanes[0], use_lanes=True, games=2,
                                   concurrent_games=2, request_lanes=lanes)
            self.assertIsNone(result["error"])
            self.assertEqual(len(worker_ids), 2)
            self.assertEqual(result["completed_games"], 2)
            self.assertEqual(result["target_games_finished"], 2)
            records = read_rows(root / "episodes.jsonl")
            self.assertEqual(len(records), 3)
            self.assertEqual(sum(g["status"] == "completed" for g in records), 2)
            self.assertEqual(sum(g["status"] == "truncated" for g in records), 1)
            self.assertEqual(len({g["game_number"] for g in records}), 3)
            self.assertEqual(list((root / "pending").glob("*.json")), [])
            self.assert_unique_trace(root)

    def test_resume_keeps_old_truncation_and_uses_a_new_game_number_for_remaining_target(self):
        for use_lanes in (False, True):
            with self.subTest(use_lanes=use_lanes), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                clock = VirtualClock()
                records = []
                for episode_id, number, moves, status in (
                    ("already-completed", 1, NEAR_WIN + ["e9"], "completed"),
                    ("old-truncation", 2, NEAR_WIN, "truncated"),
                ):
                    path = checkpoint(root, clock, episode_id, moves=moves, number=number)
                    records.append(terminal_record(path, status))
                    path.unlink()
                (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
                with virtual_scheduler(clock):
                    expert = ScriptedExpert(clock)
                    result = run_completed(root, expert, use_lanes=use_lanes)
                self.assertIsNone(result["error"])
                self.assertEqual(result["completed_games"], 1)
                self.assertEqual(result["target_games_finished"], 1)
                self.assertEqual(expert.calls[0][1], ())
                all_records = read_rows(root / "episodes.jsonl")
                self.assertEqual(all_records[:2], records)
                self.assertEqual(len(all_records), 3)
                self.assertEqual(len({g["game_number"] for g in all_records}), 3)
                self.assertEqual(sum(g["status"] == "completed" for g in all_records), 2)
                self.assert_unique_trace(root)

    def test_pause_and_rate_limit_stop_even_when_the_natural_target_is_unfilled(self):
        for use_lanes in (False, True):
            for kind in ("stopped", "rate_limited"):
                with self.subTest(use_lanes=use_lanes, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    clock = VirtualClock()
                    with virtual_scheduler(clock):
                        capped_checkpoint(root, clock)

                        def before_choose(state):
                            if not state.history:
                                expert.request_gate.stopped.set()
                                raise RemoteError("Collection must stop", kind=kind, retryable=True)

                        expert = ScriptedExpert(clock, before_choose=before_choose)
                        result = run_completed(root, expert, use_lanes=use_lanes)
                    self.assertEqual(result["error"]["kind"], kind)
                    self.assertEqual(result["completed_games"], 0)
                    self.assertEqual(result["target_games_finished"], 0)
                    self.assertEqual(len(expert.calls), 2)
                    self.assertEqual([g["status"] for g in read_rows(root / "episodes.jsonl")], ["truncated"])
                    [pending] = (root / "pending").glob("*.json")
                    self.assertEqual(json.loads(pending.read_text(encoding="utf-8"))["moves"], [])
                    self.assert_unique_trace(root)

    def test_time_limit_does_not_open_more_games_to_make_up_natural_completions(self):
        for use_lanes in (False, True):
            with self.subTest(use_lanes=use_lanes), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                clock = VirtualClock()
                with virtual_scheduler(clock):
                    capped_checkpoint(root, clock)

                    def before_choose(state):
                        if not state.history:
                            raise RemoteError("Deadline reached", kind="time_limit")

                    expert = ScriptedExpert(clock, before_choose=before_choose)
                    result = run_completed(root, expert, use_lanes=use_lanes)
                self.assertIsNone(result["error"])
                self.assertTrue(result["time_limit_reached"])
                self.assertEqual(result["completed_games"], 0)
                self.assertEqual(result["target_games_finished"], 0)
                self.assertEqual(len(expert.calls), 2)
                self.assertEqual([g["finished_reason"] for g in read_rows(root / "episodes.jsonl")],
                                 ["max_plies", "time_limit"])

    def test_natural_target_is_opt_in_for_experiments_and_enabled_for_campaigns(self):
        self.assertFalse(RunConfig().completed_only)
        self.assertIsNone(ResumeConfig().completed_only)
        self.assertEqual(ResumeConfig(completed_only=False).model_dump(exclude_none=True),
                         {"completed_only": False})
        planned = list(campaign_runs(CampaignConfig(games_per_matchup=2)))
        self.assertTrue(all(config["completed_only"] for _, config in planned))

    def test_dashboard_resume_counts_only_natural_games_and_ignores_old_cutoffs(self):
        for campaign in (False, True):
            with self.subTest(campaign=campaign), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_id, folder = saved_run(root)
                run_path = folder / "run.json"
                run = json.loads(run_path.read_text(encoding="utf-8"))
                run["config"]["completed_only"] = False
                if campaign:
                    run.update(campaign="formal-campaign", campaign_order=1)
                run_path.write_text(json.dumps(run), encoding="utf-8")
                clock = VirtualClock()
                records = []
                for number, status, moves in ((1, "completed", NEAR_WIN + ["e9"]),
                                               (2, "truncated", NEAR_WIN),
                                               (3, "truncated", NEAR_WIN[:13])):
                    path = checkpoint(folder, clock, f"saved-{number}", moves=moves, number=number)
                    records.append(terminal_record(path, status))
                    path.unlink()
                (folder / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
                called = threading.Event()
                captured = {}

                def capture_collection(expert, **settings):
                    captured.update(settings)
                    called.set()
                    return {"network_requests": 0, "time_limit_reached": False,
                            "error": {"kind": "stopped", "message": "Offline target test finished"}}

                app = create_dashboard(root, data_dir=root, replay_base="http://localhost:4173",
                                       expert_factory=OfflineExpert)
                with patch("quoridor_lab.dashboard.collect", side_effect=capture_collection), TestClient(app) as client:
                    before = client.get(f"/api/runs/{run_id}").json()
                    self.assertEqual(before["games_saved"], 3)
                    self.assertEqual(before["completed"], 1)
                    self.assertEqual(before["target_games_saved"], 1 if campaign else 3)
                    # Formal campaigns remain natural-only even with an explicit false override.
                    response = client.post(f"/api/runs/{run_id}/resume", json={"completed_only": not campaign})
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(called.wait(2))
                    deadline = time.monotonic() + 2
                    while (detail := client.get(f"/api/runs/{run_id}").json())["status"] == "running":
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.01)
                    self.assertEqual(detail["status"], "paused")
                    self.assertTrue(captured["completed_only"])
                    self.assertEqual(captured["games"], 2)
                    self.assertTrue(detail["completed_only"])
                    self.assertEqual(detail["target_games_saved"], 1)
                    self.assertEqual(detail["games_saved"], 3)
                    self.assertEqual(detail["truncated"], 2)
                    self.assertTrue(json.loads(run_path.read_text(encoding="utf-8"))["config"]["completed_only"])


if __name__ == "__main__":
    unittest.main()
