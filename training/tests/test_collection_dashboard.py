from collections import Counter
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from quoridor_lab.collect import _append, _atomic, collect
from fastapi.testclient import TestClient

from quoridor_lab.dashboard import CampaignConfig, LEVELS, campaign_runs, create_dashboard, read, replay_link, rows, sample_stats
from quoridor_lab.opponents import Decision, LocalPolicy, RemoteError, RemoteExpert
from quoridor_lab.rules import State


class FakeExpert:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.policy = LocalPolicy(seed=0)

    def choose(self, state, *, difficulty):
        self.calls.append((state.turn, difficulty))
        if self.fail and len(self.calls) == 2:
            raise RemoteError("busy", kind="rate_limited")
        return Decision(self.policy.choose(state).action, f"barricade:{difficulty}",
                        metadata={"difficulty": difficulty})


class DashboardCollectionTests(unittest.TestCase):
    def test_retries_illegal_move_at_the_same_position_without_saving_it(self):
        class OneInvalidReply(FakeExpert):
            def choose(self, state, *, difficulty):
                if not self.calls:
                    self.calls.append((state.turn, difficulty))
                    raise RemoteError("Illegal expert response: e9", kind="illegal_response")
                return super().choose(state, difficulty=difficulty)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expert = OneInvalidReply()
            result = collect(expert, output=root, games=1, max_plies=2,
                             remote_sides=("expert", "easy"), illegal_move_retries=2,
                             progress=lambda _: None)
            self.assertIsNone(result["error"])
            self.assertEqual(expert.calls[:2], [(1, "expert"), (1, "expert")])
            [retry] = list(rows(root / "rejected-moves.jsonl"))
            self.assertEqual(retry["attempt"], 1)
            self.assertEqual([row["ply"] for row in rows(root / "transitions.jsonl")], [0, 1])

    def test_formal_campaign_covers_all_distances_and_balances_colors(self):
        batches = list(campaign_runs(CampaignConfig(games_per_matchup=5000, request_interval=0.05,
                                                  illegal_move_retries=2)))
        by_matchup = Counter()
        directions = {}
        for key, config in batches:
            by_matchup[key] += config["games"]
            directions.setdefault(key, set()).add((config["red"], config["blue"]))
            self.assertFalse(config["use_cache"])
            self.assertIsNone(config["time_limit_seconds"])
            self.assertEqual(config["request_interval"], 0.05)
            self.assertEqual(config["illegal_move_retries"], 2)
        self.assertEqual(len(batches), 25)
        self.assertEqual([key for key, _ in batches[:3]], ["master-master", "expert-master", "expert-master"])
        self.assertEqual(batches[-1][0], "easy-easy")
        self.assertEqual(len(by_matchup), 15)
        self.assertEqual(set(by_matchup.values()), {5000})
        for index, weaker in enumerate(LEVELS):
            for stronger in LEVELS[index:]:
                self.assertEqual(directions[f"{weaker}-{stronger}"],
                                 {(weaker, stronger), (stronger, weaker)})

    def test_campaign_retries_one_transient_request_but_never_retries_rate_limit(self):
        class FlakyExpert(FakeExpert):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind
                self.seen = []

            def choose(self, state, *, difficulty):
                self.seen.append(state.history)
                if len(self.seen) == 1:
                    raise RemoteError(self.kind, kind=self.kind, retryable=True)
                return super().choose(state, difficulty=difficulty)

        for kind in ("transport", "rate_limited"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                expert = FlakyExpert(kind)
                result = collect(expert, output=root, games=1, max_plies=2,
                                 remote_sides=("master", "master"), retry_transient=True,
                                 progress=lambda _: None)
                if kind == "transport":
                    self.assertIsNone(result["error"])
                    self.assertEqual(expert.seen[:2], [(), ()])
                    [entry] = rows(root / "transport-retries.jsonl")
                    self.assertEqual(entry["kind"], "transport")
                    self.assertEqual(len(list(rows(root / "transitions.jsonl"))), 2)
                else:
                    self.assertEqual(result["error"]["kind"], "rate_limited")
                    self.assertEqual(expert.seen, [()])
                    self.assertFalse((root / "transport-retries.jsonl").exists())

    def test_campaign_promotes_all_queued_batches_and_pages_the_saved_games(self):
        class OfflineExpert(FakeExpert):
            cache_hits = 0

            def __init__(self, **options):
                super().__init__()
                self.request_gate = options["request_gate"]

            @property
            def network_requests(self):
                return len(self.calls)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost", expert_factory=OfflineExpert)
            with TestClient(app) as client:
                request = client.post("/api/campaigns", json={"games_per_matchup": 2, "connections": 1})
                self.assertEqual(request.status_code, 201)
                campaign = request.json()
                self.assertEqual(campaign["games"], 30)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    jobs = client.get("/api/runs").json()
                    if all(job["status"] in {"finished", "paused", "failed"} for job in jobs):
                        break
                    time.sleep(0.05)
                self.assertEqual(Counter(job["status"] for job in jobs), {"finished": 25})
                first = client.get(f"/api/runs/{campaign['id']}", params={"offset": 1, "limit": 1}).json()
                self.assertEqual(first["games_saved"], 2)
                self.assertEqual(first["completed"], 2)
                self.assertEqual([game["game_number"] for game in first["games"]], [2])
                for job in jobs:
                    games = list(rows(root / job["id"] / "episodes.jsonl"))
                    self.assertEqual(len(games), job["config"]["games"])
                    self.assertTrue(all((root / job["id"] / f"episode-{g['episode_id']}.json").exists() for g in games))

    def test_checkpoint_write_retries_a_temporary_windows_file_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            replace = Path.replace
            attempts = []

            def retry_once(source, target):
                attempts.append(target)
                if len(attempts) == 1:
                    raise PermissionError("Checkpoint is temporarily in use")
                return replace(source, target)

            with patch.object(type(path), "replace", retry_once):
                _atomic(path, {"moves": ["e2"]})

            self.assertEqual(len(attempts), 2)
            self.assertEqual(read(path), {"moves": ["e2"]})

    def test_log_append_retries_a_temporary_open_lock_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rejected-moves.jsonl"
            open_path = Path.open
            attempts = []

            def open_with_lock(source, mode="r", *args, **kwargs):
                if source == path and mode == "a":
                    attempts.append(source)
                    if len(attempts) == 1:
                        raise PermissionError("Log file is temporarily in use")
                return open_path(source, mode, *args, **kwargs)

            with patch.object(type(path), "open", open_with_lock):
                _append(path, {"move": "e2"})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(list(rows(path)), [{"move": "e2"}])

    def test_finished_game_retries_a_temporary_checkpoint_delete_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unlink = Path.unlink
            attempts = []

            def unlink_with_lock(path, *args, **kwargs):
                if path.parent == root / "pending":
                    attempts.append(path)
                    if len(attempts) == 1:
                        raise PermissionError("Checkpoint is temporarily locked")
                return unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", unlink_with_lock):
                result = collect(FakeExpert(), output=root, games=1,
                                 remote_sides=("master", "master"), progress=lambda _: None)

            self.assertIsNone(result["error"])
            [game] = list(rows(root / "episodes.jsonl"))
            self.assertEqual(game["status"], "completed")
            self.assertEqual(attempts, [root / "pending" / f"{game['episode_id']}.json"] * 2)
            self.assertFalse(attempts[0].exists())
            self.assertEqual(read(root / f"episode-{game['episode_id']}.json"), game)

    def test_dashboard_read_retries_while_a_checkpoint_is_being_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            _atomic(path, {"moves": ["e2"]})
            read_text = Path.read_text
            with patch.object(type(path), "read_text", side_effect=[
                PermissionError("Checkpoint is temporarily locked"), read_text(path, encoding="utf-8"),
            ]) as reader:
                self.assertEqual(read(path), {"moves": ["e2"]})
            self.assertEqual(reader.call_count, 2)

    def test_dashboard_details_can_poll_while_one_checkpoint_is_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "a" * 32
            folder = root / run_id
            pending = folder / "pending"
            pending.mkdir(parents=True)
            _atomic(folder / "run.json", {"status": "paused", "config": {"opening": []}})
            locked = pending / "locked.json"
            _atomic(locked, {"episode_id": "locked", "game_number": 1, "moves": []})
            _atomic(pending / "visible.json", {"episode_id": "visible", "game_number": 2, "moves": []})
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost")
            read_text = Path.read_text

            def read_with_lock(path, *args, **kwargs):
                if path == locked:
                    raise PermissionError("Checkpoint is temporarily locked")
                return read_text(path, *args, **kwargs)

            with TestClient(app) as client:
                url = f"/api/runs/{run_id}"
                with patch.object(Path, "read_text", read_with_lock):
                    response = client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertEqual([game["game_number"] for game in response.json()["active"]], [2])
                self.assertEqual([game["game_number"] for game in client.get(url).json()["active"]], [1, 2])

    def test_resume_changes_request_interval_without_restarting_saved_game(self):
        created = []

        class OfflineExpert(FakeExpert):
            cache_hits = 0
            network_requests = 0

            def __init__(self, **options):
                super().__init__()
                self.fail = not created
                created.append(options)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost", expert_factory=OfflineExpert)
            with TestClient(app) as client:
                result = client.post("/api/runs", json={"games": 1, "max_plies": 2}).json()
                url = f"/api/runs/{result['id']}"
                deadline = time.monotonic() + 3
                while client.get(url).json()["status"] == "running" and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(client.get(url).json()["error_kind"], "rate_limited")
                [checkpoint] = (root / result["id"] / "pending").glob("*.json")
                self.assertEqual(client.post(url + "/resume", json={"concurrent_games": 2}).status_code, 400)
                response = client.post(url + "/resume", json={"concurrent_games": 1, "request_interval": 2})
                self.assertEqual(response.status_code, 200)
                deadline = time.monotonic() + 3
                while (detail := client.get(url).json())["status"] == "running" and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(detail["status"], "finished")
                self.assertEqual(detail["games"][0]["episode_id"], checkpoint.stem)
                self.assertEqual(created[-1]["interval"], 2)

    def test_pause_keeps_live_game_checkpoint(self):
        waiting = threading.Event()

        class OfflineExpert(FakeExpert):
            cache_hits = 0
            network_requests = 0

            def __init__(self, *, request_gate, **options):
                super().__init__()
                self.gate = request_gate

            def choose(self, state, *, difficulty):
                if state.history:
                    waiting.set()
                    self.gate.stopped.wait(3)
                self.gate.wait(None)
                self.gate.done()
                return super().choose(state, difficulty=difficulty)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_dashboard(root, data_dir=root, replay_base="http://localhost", expert_factory=OfflineExpert)
            with TestClient(app) as client:
                result = client.post("/api/runs", json={"games": 1, "max_plies": 2}).json()
                url = f"/api/runs/{result['id']}"
                self.assertTrue(waiting.wait(2))
                self.assertEqual(client.post(url + "/pause").status_code, 202)
                deadline = time.monotonic() + 3
                while (detail := client.get(url).json())["status"] == "running" and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(detail["status"], "paused")
                self.assertEqual(detail["samples_saved"], 1)
                self.assertEqual(detail["active"][0]["ply"], 1)

    def test_independent_connections_can_wait_for_replies_at_the_same_time(self):
        barrier = threading.Barrier(2, timeout=4)

        class Lane(FakeExpert):
            cache_hits = 0

            @property
            def network_requests(self):
                return len(self.calls)

            def choose(self, state, *, difficulty):
                barrier.wait()
                return super().choose(state, difficulty=difficulty)

        with tempfile.TemporaryDirectory() as tmp:
            lanes = [Lane(), Lane()]
            root = Path(tmp)
            result = collect(lanes[0], output=root, games=6, concurrent_games=6,
                             max_plies=2, remote_sides=("expert", "easy"),
                             request_lanes=lanes, games_per_connection=3, progress=lambda _: None)
            self.assertEqual([lane.network_requests for lane in lanes], [6, 6])
            self.assertEqual(result["finished_games"], 6)
            self.assertEqual(result["network_requests"], 12)
            samples = list(rows(root / "transitions.jsonl"))
            for game in result["episodes"]:
                trace = [row for row in samples if row["episode_id"] == game["episode_id"]]
                self.assertEqual([row["history"] for row in trace], [[], game["moves"][:1]])
                self.assertEqual([row["notation"] for row in trace], game["moves"])

    def test_campaign_defers_illegal_game_and_finishes_the_other_games(self):
        class Lane(FakeExpert):
            cache_hits = 0
            network_requests = 0

            def choose(self, state, *, difficulty):
                if self.fail:
                    raise RemoteError("Illegal expert response: e2", kind="illegal_response")
                return super().choose(state, difficulty=difficulty)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = Lane(), Lane()
            first.fail = True
            options = dict(output=root, max_plies=2, remote_sides=("easy", "easy"),
                           concurrent_games=2, request_lanes=[first, second], games_per_connection=1,
                           illegal_move_retries=2, defer_illegal_responses=True, progress=lambda _: None)
            result = collect(first, games=2, **options)
            self.assertEqual(result["finished_games"], 1)
            self.assertEqual(result["error"]["kind"], "illegal_response")
            [checkpoint] = (root / "pending").glob("*.json")
            saved = read(checkpoint)
            self.assertEqual(saved["moves"], [])
            self.assertEqual(len(list(rows(root / "rejected-moves.jsonl"))), 2)
            first.fail = False
            resumed = collect(first, games=1, resume=True, **options)
            self.assertIsNone(resumed["error"])
            self.assertEqual(resumed["episodes"][0]["episode_id"], saved["episode_id"])
            self.assertFalse(checkpoint.exists())

    def test_campaign_automatically_retries_the_same_game_after_deferral(self):
        class Lane(FakeExpert):
            cache_hits = 0

            @property
            def network_requests(self):
                return len(self.calls)

            def choose(self, state, *, difficulty):
                if len(self.calls) == 1:
                    self.calls.append((state.turn, difficulty))
                    raise RemoteError("Illegal expert response: e2", kind="illegal_response")
                return super().choose(state, difficulty=difficulty)

        with tempfile.TemporaryDirectory() as tmp, patch("quoridor_lab.collect.DEFERRED_RETRY_DELAY", 0):
            root = Path(tmp)
            lane = Lane()
            result = collect(lane, output=root, games=1, max_plies=2, remote_sides=("master", "master"),
                             request_lanes=[lane], games_per_connection=1, defer_illegal_responses=True,
                             deferred_retry_limit=1, progress=lambda _: None)
            self.assertIsNone(result["error"])
            [deferred] = rows(root / "deferred-games.jsonl")
            [game] = result["episodes"]
            self.assertEqual(game["episode_id"], deferred["episode_id"])
            self.assertEqual([row["ply"] for row in rows(root / "transitions.jsonl")], [0, 1])
            self.assertEqual(lane.calls, [(1, "master"), (2, "master"), (2, "master")])
            self.assertEqual(list((root / "pending").glob("*.json")), [])

    def test_remote_limit_expires_before_next_allowed_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            expert = RemoteExpert(data_dir=tmp, interval=1, cache=False)
            expert._last_request = 10
            clock = 10.5

            def wait(seconds):
                nonlocal clock
                clock += seconds

            with patch("quoridor_lab.opponents.time.monotonic", side_effect=lambda: clock), \
                 patch("quoridor_lab.opponents.time.sleep", side_effect=wait), \
                 patch.object(expert, "_connect"):
                with self.assertRaises(RemoteError) as stopped:
                    expert.choose(State(), deadline=10.75)
            self.assertEqual(stopped.exception.kind, "time_limit")
            self.assertEqual(expert.network_requests, 0)
            self.assertEqual(clock, 10.75)

    def test_time_limit_saves_every_active_game_without_a_winner(self):
        class DeadlineExpert(FakeExpert):
            def choose(self, state, *, difficulty, deadline):
                self.assert_deadline = deadline
                if len(self.calls) == 1:
                    raise RemoteError("expired", kind="time_limit")
                return super().choose(state, difficulty=difficulty)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expert = DeadlineExpert()
            deadline = time.monotonic() + 10
            result = collect(expert, output=root, games=3, concurrent_games=3,
                             remote_sides=("expert", "easy"), deadline=deadline,
                             progress=lambda _: None)
            self.assertEqual(expert.assert_deadline, deadline)
            self.assertTrue(result["time_limit_reached"])
            self.assertIsNone(result["error"])
            games = list(rows(root / "episodes.jsonl"))
            self.assertEqual(len(games), 3)
            self.assertEqual(sorted(game["ply_count"] for game in games), [0, 0, 1])
            for game in games:
                self.assertEqual(game["status"], "truncated")
                self.assertEqual(game["finished_reason"], "time_limit")
                self.assertIsNone(game["winner"])
                self.assertEqual(State.from_moves(game["moves"]).to_dict(), game["final"])
            self.assertFalse(list((root / "pending").glob("*.json")))

    def test_concurrent_games_take_turns_and_resume_after_shared_error(self):
        expert = FakeExpert()
        expert.fail = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"output": root, "remote_sides": ("expert", "easy"),
                      "games": 3, "concurrent_games": 3, "max_plies": 4,
                      "progress": lambda _: None}
            paused = collect(expert, **config)
            self.assertEqual(paused["error"]["kind"], "rate_limited")
            self.assertEqual(len(list((root / "pending").glob("*.json"))), 3)
            self.assertEqual(expert.calls, [(1, "expert"), (1, "expert")])
            expert.fail = False
            result = collect(expert, resume=True, **config)
            self.assertIsNone(result["error"])
            self.assertEqual({game["game_number"] for game in result["episodes"]}, {1, 2, 3})
            samples = list(rows(root / "transitions.jsonl"))
            self.assertEqual(len({row["episode_id"] for row in samples if row["ply"] == 0}), 3)
            for game in result["episodes"]:
                trace = [row for row in samples if row["episode_id"] == game["episode_id"]]
                self.assertEqual([row["ply"] for row in trace], [0, 1, 2, 3])
                for row in trace:
                    self.assertEqual(row["history"], game["moves"][:row["ply"]])
                    self.assertEqual(row["notation"], game["moves"][row["ply"]])
                self.assertEqual(State.from_moves(game["moves"]).to_dict(), game["final"])
            self.assertFalse(list((root / "pending").glob("*.json")))

    def test_dual_remote_opening_resume_and_replay(self):
        expert = FakeExpert()
        expert.fail = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"output": root, "remote_sides": ("master", "easy"), "max_plies": 4,
                      "opening": ("e2",), "progress": lambda _: None}
            run = collect(expert, **config)
            self.assertEqual(run["error"]["kind"], "rate_limited")
            self.assertEqual(expert.calls, [(2, "easy"), (1, "master")])
            expert.fail = False
            run = collect(expert, resume=True, **config)
            [game] = run["episodes"]
            samples = list(rows(root / "transitions.jsonl"))
            self.assertEqual([row["ply"] for row in samples], [1, 2, 3])
            self.assertTrue(all(row["is_teacher"] for row in samples))
            self.assertEqual([row["source"] for row in samples],
                             ["barricade:easy", "barricade:master", "barricade:easy"])
            self.assertEqual(State.from_moves(game["moves"]).to_dict(), game["final"])
            self.assertEqual(game["status"], "truncated")
            query = parse_qs(urlparse(replay_link("http://127.0.0.1:4173", game)).query)
            self.assertEqual(query["moves"], [",".join(game["moves"])])
            self.assertEqual(query["ai"], ["master,easy"])

    def test_duplicate_demonstrations_use_position_difficulty_and_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = collect(FakeExpert(), output=root, games=2, max_plies=2,
                          remote_sides=("expert", "expert"), concurrent_games=2,
                          progress=lambda _: None)
            trace = list(rows(root / "transitions.jsonl"))
            self.assertEqual([row["ply"] for row in trace], [0, 0, 1, 1])
            self.assertEqual([row["episode_id"] for row in trace[:2]],
                             [row["episode_id"] for row in trace[2:]])
            stats = sample_stats(root / "transitions.jsonl", run["episodes"])
            self.assertEqual(stats["samples"], 4)
            self.assertEqual(stats["repeated_positions"], 2)
            self.assertEqual(stats["repeated_demonstrations"], 2)
            self.assertEqual(stats["repeated_request_histories"], 2)
            self.assertEqual(stats["changed_request_histories"], 0)

    def test_remote_cache_distinguishes_difficulty_on_one_connection(self):
        class Socket:
            connected = True

            def call(self, event, payload, **kwargs):
                return {"ok": True, "move": "e8"}

        with tempfile.TemporaryDirectory() as tmp:
            expert = RemoteExpert(data_dir=tmp, interval=.25)
            expert.client = Socket()
            state = State.from_moves(["e2"])
            self.assertFalse(expert.choose(state, difficulty="easy").cached)
            self.assertFalse(expert.choose(state, difficulty="master").cached)
            self.assertEqual(expert.choose(state, difficulty="easy").source, "barricade:easy")
            self.assertEqual(expert.network_requests, 2)
            self.assertEqual(expert.cache_hits, 1)
