"""Regression checks for reproducible, replayable search experiments."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from quoridor_lab.rules import State
from quoridor_lab.search import SearchAgent

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "run_search_baseline.py"
spec = importlib.util.spec_from_file_location("search_baseline", SCRIPT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class SearchBaselineTests(unittest.TestCase):
    def test_seeds_and_openings_survive_process_hash_randomization(self):
        code = (
            "import run_search_baseline as b; import json; "
            "seeds=[b.game_seed(23, pair, i) for pair in b.MATCHUPS for i in range(3)]; "
            "print(json.dumps([(s, b.make_opening(s, 4)) for s in seeds]))"
        )
        outputs = []
        for hash_seed in ("1", "2"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed}
            outputs.append(subprocess.check_output(
                [sys.executable, "-c", code], cwd=SCRIPT.parent, env=env, text=True
            ))
        self.assertEqual(outputs[0], outputs[1])
        openings = [tuple(moves) for _, moves in json.loads(outputs[0])]
        self.assertGreater(len(set(openings)), 1)
        for moves in openings:
            self.assertEqual(State.from_moves(moves).history, moves)

    def test_cutoff_is_truncated_and_history_can_be_replayed(self):
        result = baseline.play_game(SearchAgent("easy"), SearchAgent("easy"),
                                    seed=0, opening=("e2",), max_plies=2)
        self.assertEqual(result["result"], "truncated")
        self.assertTrue(result["truncated"])
        moves = json.loads(result["moves"])
        self.assertEqual(len(moves), 2)
        self.assertIsNone(State.from_moves(moves).winner)
        self.assertEqual(result["_steps"][0]["notation"], moves[-1])
        self.assertEqual(result["_steps"][0]["ply"], 2)

    def test_truncated_games_are_excluded_from_completed_win_rate(self):
        rows = [
            {"matchup": "easy_vs_hard", "red": "easy", "blue": "hard", "result": result}
            for result in ("blue", "truncated")
        ]
        summary = baseline.matchup_summary(rows, "easy", "hard")
        self.assertEqual(summary["completed_games"], 1)
        self.assertEqual(summary["truncated_games"], 1)
        self.assertEqual(summary["hard_win_rate"], 1.0)
        rows[0]["result"] = "truncated"
        summary = baseline.matchup_summary(rows, "easy", "hard")
        self.assertIsNone(summary["hard_win_rate"])
        self.assertIn("无完成对局", baseline.build_conclusion([summary]))

    def test_cli_exports_paired_openings_full_histories_and_step_timings(self):
        # Use the real game/export path with cheap policies; search correctness
        # is covered independently in test_search_ai.
        class CheapAgent:
            def __init__(self, *, seed, config):
                self.policy = SearchAgent("easy", seed=seed)

            def choose(self, state):
                return self.policy.choose(state)

        with tempfile.TemporaryDirectory() as folder:
            argv = [str(SCRIPT), "--games", "2", "--max-plies", "6",
                    "--opening-plies", "4", "--output", folder, "--seed", "23",
                    "--hard-time-limit-seconds", "0"]
            with patch.object(sys, "argv", argv), patch.object(baseline, "SearchAgent", CheapAgent):
                baseline.main()
            summary = json.loads(next(Path(folder).glob("*.json")).read_text())
            self.assertEqual(summary["seed"], 23)
            self.assertIsNone(summary["difficulty_config"]["hard"]["time_limit_seconds"])
            with (Path(folder) / summary["csv"]).open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 12)
            for first, second in zip(rows[::2], rows[1::2]):
                self.assertEqual(first["opening_moves"], second["opening_moves"])
                self.assertEqual(first["red"], second["blue"])
                for row in (first, second):
                    state = State.from_moves(json.loads(row["moves"]))
                    self.assertEqual(len(state.history), int(row["plies"]))
            steps = [json.loads(line) for line in (Path(folder) / summary["steps_jsonl"]).read_text().splitlines()]
            self.assertEqual(len(steps), 24)
            self.assertTrue(all(s["latency_seconds"] >= 0 for s in steps))


if __name__ == "__main__":
    unittest.main()
