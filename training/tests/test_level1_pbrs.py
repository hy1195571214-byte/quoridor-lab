import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Install learning dependencies for Level 1 PPO tests")
class Level1PBRSTests(unittest.TestCase):
    def test_adjusted_history_truncation_keeps_the_search_limit(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from quoridor_lab.level1_arena import passed
        from quoridor_lab.phase1_curriculum import LEVEL_GATES, gate_policy
        from quoridor_lab.training_monitor import atomic_json

        thresholds = {"history": .65, "level3": .005}
        scores = {"history": {"win_rate": .72, "truncation_rate": .18, "illegal": 0},
                  "level3": {"win_rate": .01, "truncation_rate": .05, "illegal": 0}}
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            atomic_json(folder / "gate-adjustment.json", {
                "reason": "Audited history-pool truncation",
                "adjusted": {"wins": thresholds, "truncation_limit": .05, "truncation_limits": {"history": .2}},
            })
            original, adjusted, _ = gate_policy(folder, LEVEL_GATES[5])
            self.assertFalse(passed(scores, adjusted["wins"], adjusted["truncation_limit"],
                                    truncation_limits=adjusted["truncation_limits"]))
            scores["level3"]["truncation_rate"] = .045
            self.assertTrue(passed(scores, adjusted["wins"], adjusted["truncation_limit"],
                                   truncation_limits=adjusted["truncation_limits"]))
            self.assertFalse(passed(scores, original["wins"], original["truncation_limit"]))
            self.assertEqual(gate_policy(folder, LEVEL_GATES[5], original_only=True),
                             (original, original, None))

    def test_reference_kl_uses_only_legal_moves_and_keeps_the_reference_frozen(self):
        import math
        from copy import deepcopy
        from quoridor_lab.train_level1 import HYPER, update_policy
        from quoridor_lab.train_phase1 import ActorCritic

        model = ActorCritic("mlp")
        reference = deepcopy(model).eval()
        with torch.no_grad():
            reference.actor.bias[0] += 2
        boards = torch.zeros(8, 8, 9, 9)
        masks = torch.zeros(8, 209, dtype=torch.bool)
        masks[:, :2] = True
        actions = torch.arange(8) % 2
        with torch.no_grad():
            old_log_probs = torch.distributions.Categorical(logits=model(boards, masks)[0]).log_prob(actions)
        data = (boards, masks, actions, old_log_probs, torch.linspace(-1., 1., 8), torch.ones(8))
        options = {**HYPER, "update_epochs": 2, "minibatch": 4, "reference_kl_coef": 1.}
        optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"])
        diagnostics = update_policy(model, optimizer, data, options, "cpu", reference=reference)
        self.assertTrue(all(math.isfinite(value) for value in diagnostics.values()))
        self.assertGreater(diagnostics["reference_kl"], 0.)
        self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))
        self.assertFalse(torch.equal(model.actor.weight, reference.actor.weight))

    def test_policy_recovery_preserves_stage_progress_and_only_applies_once(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from quoridor_lab.phase1_curriculum import recover_policy
        from quoridor_lab.train_level1 import sha256
        from quoridor_lab.training_monitor import atomic_json

        config = {"architecture": "cnn32", "phase1_level": 3}
        optimizer = {"param_groups": [{"lr": .0003}]}
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "checkpoint-00008.pt"
            torch.save({"config": config, "state": {"update": 8}, "model": {"weight": torch.tensor([8])},
                        "optimizer": optimizer}, source)
            atomic_json(folder / "policy-recovery.json", {
                "checkpoint_update": 8, "checkpoint_sha256": sha256(source), "reason": "regression",
            })
            current = {"config": config, "optimizer": optimizer, "model": {"weight": torch.tensor([71])},
                       "state": {"update": 71, "qualified": False, "hyper": {"lr": .0003}, "streak": 1,
                                 "game_units": {"completed": 6916, "next_evaluation": 7000},
                                 "evaluation_count": 13, "rollout": {"started": 7000}}}
            recovered, event = recover_policy(folder, current)
            self.assertEqual(recovered["model"]["weight"].item(), 8)
            self.assertEqual(event["from_update"], 72)
            self.assertEqual(recovered["state"]["streak"], 0)
            self.assertEqual(recovered["state"]["update"], 71)
            self.assertEqual(recovered["state"]["evaluation_count"], 13)
            self.assertEqual(recovered["state"]["game_units"], {"completed": 6916, "next_evaluation": 7000})
            self.assertEqual(recovered["state"]["rollout"], {"started": 7000})
            self.assertEqual(recover_policy(folder, recovered), (recovered, None))

    def test_local_ppo_recovery_requires_a_finished_unit_from_the_same_run(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from quoridor_lab.phase1_curriculum import recover_policy
        from quoridor_lab.train_level1 import sha256
        from quoridor_lab.training_monitor import atomic_json

        config = {"architecture": "mlp", "phase1_level": 3}
        hyper = {"lr": .0003}
        optimizer = {"param_groups": [{"lr": .0003}]}
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "checkpoint-00008.pt"
            torch.save({"config": config, "state": {"update": 8}}, source)
            current = {"config": config, "optimizer": optimizer, "model": {"weight": torch.tensor([71])},
                       "state": {"update": 71, "qualified": False, "hyper": hyper, "streak": 1,
                                 "game_units": {"completed": 6916, "next_evaluation": 7000}}}
            candidate_config = {"source_sha256": sha256(source), "hyperparameters": hyper}

            def request(completed, lineage):
                candidate = folder / "candidate.pt"
                torch.save({"config": {**candidate_config, "source_sha256": lineage},
                            "state": {"phase": "ppo", "update": 4, "completed": completed},
                            "model": {"weight": torch.tensor([92])}, "optimizer": optimizer}, candidate)
                digest = sha256(candidate)
                target = candidate.with_name(f"local-ppo-recovery-{digest}.pt")
                candidate.rename(target)
                atomic_json(folder / "policy-recovery.json", {
                    "checkpoint_update": 8, "checkpoint_sha256": digest, "reason": "Independent evaluation",
                    "local_ppo_candidate": target.name,
                })

            for completed, lineage in ((499, sha256(source)), (500, "a" * 64)):
                with self.subTest(completed=completed, lineage=lineage):
                    request(completed, lineage)
                    with self.assertRaises(ValueError):
                        recover_policy(folder, current)
            request(503, sha256(source))
            recovered, event = recover_policy(folder, current)
            self.assertEqual(recovered["model"]["weight"].item(), 92)
            self.assertEqual(recovered["state"]["game_units"]["completed"], 6916)
            self.assertEqual(event["continuation_games"], 503)
            self.assertEqual(event["from_update"], 72)

    def test_phase1_best_uses_one_gate_for_evaluations_before_and_after_adjustment(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from quoridor_lab.phase1_course import best_graded_checkpoint

        first = {"level2": {"win_rate": .36, "illegal": 0, "truncation_rate": 0.}}
        second = {"level2": {"win_rate": .31, "illegal": 0, "truncation_rate": 0.}}
        gate = {"wins": {"level2": .30}, "truncation_limit": .05}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{"update": 248, "formal_eval": first, "gate_passed": False, "elo": 1250.},
                    {"update": 419, "formal_eval": second, "gate_passed": True, "elo": 1250.}]
            (root / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch("quoridor_lab.phase1_course.checkpoint_info",
                       return_value={"update": 248, "scores": first}) as load:
                result = best_graded_checkpoint(root, 2, {"gate_policy": gate})
            load.assert_called_once_with(root / "checkpoint-00248.pt", 2)
            self.assertEqual(result["scores"], first)

    def test_final_arena_replays_require_the_frozen_opening_both_seats_and_real_result(self):
        from quoridor_lab.level1_arena import evaluation_games, play_evaluation, score_games
        from quoridor_lab.phase1_final_arena import ARENA_SEED, verify_match_replays

        openings = [("e2", "e8"), ("f1", "d9")]
        rows = play_evaluation(evaluation_games(openings, "bfs", ARENA_SEED), "cpu", bot="bfs")
        self.assertEqual(verify_match_replays(rows, openings, "bfs"), score_games(rows))
        reversed_result = "losses" if rows[0]["result"] == "wins" else "wins"
        failures = (rows[:-1], [rows[0], *rows], [{**rows[0], "result": reversed_result}, *rows[1:]])
        for replay in failures:
            with self.assertRaises(ValueError):
                verify_match_replays(replay, openings, "bfs")

    def test_bitset_path_filter_preserves_wall_order_and_search_replies(self):
        from quoridor_lab.level1_arena import DEEP_SEARCH, SearchPosition, bot_action
        from quoridor_lab.rules import State
        from quoridor_lab.search import SearchAgent, candidate_actions

        state = State()
        for number in range(30):
            if state.winner:
                break
            if number % 5 == 0:
                position = SearchPosition(state.pawns, state.walls, state.remaining, state.turn, state.winner)
                for name, preset in DEEP_SEARCH.items():
                    with self.subTest(ply=number, opponent=name):
                        self.assertEqual(candidate_actions(position, preset), candidate_actions(state, preset))
                        self.assertEqual(bot_action(state, name, 0), SearchAgent(config=preset).choose(state).action.id)
            state = state.apply(bot_action(state, "random", 127))

    def test_evaluation_emits_same_complete_records_with_batched_replies(self):
        from quoridor_lab.level1_arena import evaluation_games, play_evaluation, score_games
        from quoridor_lab.phase1_curriculum import Opponents

        openings = [("e2", "e8"), ("f1", "d9")]
        with patch("quoridor_lab.level1_arena.MAX_PLIES", 12):
            independent = play_evaluation(evaluation_games(openings, "random", 0), "cpu", bot="bfs")
            resolver = Opponents({})
            games = evaluation_games(openings, "random", 0, resolver)
            batched = play_evaluation(games, "cpu", bot="bfs", resolve=resolver)
        self.assertEqual(independent, batched)
        self.assertEqual(score_games(batched)["games"], 4)

    def test_batched_replies_match_individual_game_steps(self):
        from copy import deepcopy
        from quoridor_lab.level1_arena import Game, bot_action, step_batch

        games = [Game.start(("e2", "e8"), 1, name, seed) for seed, name in enumerate(("bfs", "random", "level2"))]
        independent = deepcopy(games)
        learner = [bot_action(game.state, "bfs", game.seed) for game in games]
        expected = [game.step(action) for game, action in zip(independent, learner)]
        actual = step_batch(games, learner, lambda requests: [bot_action(*request) for request in requests])
        self.assertEqual(actual, expected)
        self.assertEqual(games, independent)

    def test_random_comparison_pairs_and_restores_opponent_schedule(self):
        from collections import Counter
        from quoridor_lab.train_level1 import Experience, RANDOM_TRAINING_POOL

        openings = [("e2", "e8"), ("f1", "d9")]
        experience = Experience(openings, 4, 0, pool=RANDOM_TRAINING_POOL)
        games = [experience.next_game() for _ in range(13)]
        restored = Experience(openings, 4, 0, experience.snapshot(), pool=RANDOM_TRAINING_POOL)
        for _ in range(37):
            original = experience.next_game()
            self.assertEqual(restored.next_game(), original)
            games.append(original)
        counts = {"level1": 14, "bfs": 4, "randomized_search": 2, "random": 5}
        for player in (1, 2):
            self.assertEqual(Counter(game.opponent for game in games if game.player == player), counts)
        for first, second in zip(games[::2], games[1::2]):
            self.assertEqual((first.opening, first.opponent, first.seed),
                             (second.opening, second.opponent, second.seed))

    def test_potential_telescopes_at_natural_end_and_at_the_ply_cap(self):
        from quoridor_lab.level1_arena import Game, bot_action
        from quoridor_lab.train_level1 import bfs_goal_potential

        gamma, scale = .99, .02
        for max_plies, endings in ((256, {"wins", "losses"}), (4, {"truncated"})):
            for distance_margin in (False, True):
                with self.subTest(max_plies=max_plies, distance_margin=distance_margin), patch(
                    "quoridor_lab.level1_arena.MAX_PLIES", max_plies
                ):
                    game = Game.start((), 1, "bfs", 0)
                    initial = bfs_goal_potential(game.state, 1, scale, distance_margin=distance_margin)
                    discounted_bonus = 0.
                    for index in range(max_plies):
                        before = bfs_goal_potential(game.state, 1, scale, distance_margin=distance_margin)
                        other_distance = game.state.distance(2) if distance_margin else 0
                        self.assertAlmostEqual(before, scale * (other_distance - game.state.distance(1)))
                        result = game.step(bot_action(game.state, "bfs", 10))
                        after = 0. if result else bfs_goal_potential(game.state, 1, scale,
                                                                    distance_margin=distance_margin)
                        discounted_bonus += gamma ** index * (gamma * after - before)
                        if result:
                            break
                    self.assertIn(result, endings)
                    self.assertAlmostEqual(discounted_bonus, -initial)

    def test_shaping_reaches_ppo_targets_without_changing_sampled_actions(self):
        from quoridor_lab.train_level1 import Experience, HYPER
        from quoridor_lab.train_phase1 import ActorCritic

        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        torch.manual_seed(11)
        model = ActorCritic("mlp")
        hyper = {**HYPER, "rollout_actions": 16}
        torch.manual_seed(31)
        raw, results, *_ = Experience([("e2", "e8")], 4, 0).collect(model, hyper, "cpu")
        for potential in ("own_distance", "distance_margin"):
            with self.subTest(potential=potential):
                torch.manual_seed(31)
                shaped, shaped_results, opponents, *_ = Experience([("e2", "e8")], 4, 0).collect(
                    model, {**hyper, "pbrs_scale": .02, "pbrs_potential": potential}, "cpu"
                )
                self.assertEqual(shaped_results, results)
                self.assertTrue(all(torch.equal(before, after) for before, after in zip(raw[:4], shaped[:4])))
                self.assertFalse(torch.allclose(raw[-1], shaped[-1]))
                self.assertGreater(sum(row.get("pbrs_nonzero", 0) for row in opponents.values()), 0)

    def test_ppo_logs_settled_games_by_opening_source(self):
        from quoridor_lab.train_level1 import Experience, HYPER
        from quoridor_lab.train_phase1 import ActorCritic

        model = ActorCritic("mlp")
        for fraction, source in ((0., "standard"), (1., "curriculum")):
            hyper = {**HYPER, "rollout_actions": 8, "curriculum_fraction": fraction}
            experience = Experience([["e2", "e8"]], 4, 0, curriculum=[["f1", "d9"]], pool=("bfs",))
            with self.subTest(source=source), patch("quoridor_lab.level1_arena.MAX_PLIES", 4):
                _, outcomes, opponents, *_ = experience.collect(model, hyper, "cpu")
                self.assertEqual(outcomes["truncated"], 8)
                self.assertEqual(opponents["bfs"][f"truncated_{source}"], 8)

    def test_repeat_temperature_is_used_by_sampling_and_ppo_ratio(self):
        from quoridor_lab.level1_arena import Game
        from quoridor_lab.train_level1 import Experience, HYPER, repeats_pawn_moves, update_policy
        from quoridor_lab.train_phase1 import ActorCritic

        repeated = ("e2", "e8", "e1", "e9") * 2
        self.assertTrue(repeats_pawn_moves(repeated))
        self.assertFalse(repeats_pawn_moves(repeated[:-1]))
        experiences = Experience([("e2", "e8")], 2, 0, pool=("bfs",))
        experiences.games = [Game.start(repeated, 1, "bfs", 0), Game.start(("e2", "e8"), 1, "bfs", 1)]
        model = ActorCritic("mlp")
        hyper = {**HYPER, "rollout_actions": 2, "minibatch": 2, "update_epochs": 1, "repeat_temperature": 2.5}
        batch, _, statistics, *_ = experiences.collect(model, hyper, "cpu")
        boards, masks, actions, old_log_prob, _, _, temperatures = batch
        self.assertEqual(temperatures.tolist(), [2.5, 1.])
        self.assertEqual(statistics["bfs"]["repeat_exploration_actions"], 1)
        with torch.no_grad():
            logits, _ = model(boards, masks)
            expected = torch.distributions.Categorical(logits=logits / temperatures[:, None]).log_prob(actions)
        self.assertTrue(torch.allclose(old_log_prob, expected))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.)
        scores = update_policy(model, optimizer, batch, hyper, "cpu")
        self.assertAlmostEqual(scores["approx_kl"], 0., places=6)

    def test_mirrored_rollout_uses_reflected_mask_but_plays_physical_move(self):
        from random import Random
        from quoridor_lab.fast_observation import training_observation
        from quoridor_lab.level1_arena import Game
        from quoridor_lab.rules import Action, State
        from quoridor_lab.train_level1 import Experience, HORIZONTAL_ACTIONS, HYPER, rollout_observations
        from quoridor_lab.train_phase1 import ActorCritic

        state = State().apply("e2").apply("e8").apply("hc6").apply("vg5")
        seed = next(n for n in range(10) if Random(5_430_000 + n).getrandbits(1))
        game = Game(state, state.turn, "bfs", seed, state.history)
        boards, masks, mirrored = rollout_observations([game], True)
        opposite_view = State(tuple((r, 8 - c) for r, c in state.pawns),
                              tuple(sorted((k, r, 7 - c) for k, r, c in state.walls)),
                              state.remaining, state.turn, state.winner, state.history)
        expected = training_observation(opposite_view)
        self.assertTrue(mirrored.item())
        self.assertTrue(torch.equal(boards[0], torch.from_numpy(expected["board"])))
        self.assertTrue(torch.equal(masks[0], torch.from_numpy(expected["action_mask"]).bool()))
        experience = Experience([state.history], 1, 0, pool=("bfs",))
        experience.games = [game]
        hyper = {**HYPER, "rollout_actions": 1, "mirror_rollouts": True}
        (x, legal, actions, old_log_prob, _, _), _, counts, *_ = experience.collect(ActorCritic("mlp"), hyper, "cpu")
        self.assertTrue(torch.equal(x, boards))
        self.assertTrue(torch.equal(legal, masks))
        self.assertTrue(torch.isfinite(old_log_prob).all())
        physical = Action.from_id(HORIZONTAL_ACTIONS[actions[0]].item())
        self.assertEqual(game.state.history[len(state.history)], physical.notation)
        self.assertEqual(counts["bfs"]["mirrored_actions"], 1)

    def test_practice_replaces_half_the_random_slots_and_preserves_incomplete_pair(self):
        from collections import Counter
        from quoridor_lab.level1_arena import bot_action
        from quoridor_lab.phase1_curriculum import LEVEL3_PRACTICE, Opponents, resolve_practice
        from quoridor_lab.train_level1 import Experience

        openings = [("e2", "e8")]
        pool = ("level3",) * 4 + ("random",)
        resolve = lambda state, name, seed: bot_action(state, "bfs", seed)
        experience = Experience(openings, 1, 0, pool=pool, resolve=resolve)
        experience.pool_replacements = {"random": LEVEL3_PRACTICE}
        games = [experience.next_game() for _ in range(20)]
        self.assertEqual(Counter(game.opponent for game in games), {"level3": 16, "random": 2, LEVEL3_PRACTICE: 2})
        for left, right in zip(games[::2], games[1::2]):
            self.assertEqual((left.opponent, left.opening, left.seed), (right.opponent, right.opening, right.seed))
        experience = Experience(openings, 1, 0, pool=pool, resolve=resolve)
        experience.pool_replacements = {"random": LEVEL3_PRACTICE}
        for _ in range(5):
            first = experience.next_game()
            if first.opponent == LEVEL3_PRACTICE:
                break
            experience.next_game()
        self.assertEqual(first.opponent, LEVEL3_PRACTICE)
        resumed = Experience(openings, 1, 0, experience.snapshot(), pool=pool, resolve=resolve)
        second = resumed.next_game()
        self.assertEqual((first.opponent, first.opening, first.seed), (second.opponent, second.opening, second.seed))
        request = (second.state, LEVEL3_PRACTICE, second.seed)
        self.assertEqual(Opponents({}).batch([request]), [bot_action(*resolve_practice(*request))])

    def test_revisiting_earlier_opponents_keeps_pairs_on_ordinary_openings(self):
        from collections import Counter
        from quoridor_lab.level1_arena import bot_action
        from quoridor_lab.train_level1 import Experience

        standard = ("e2", "e8")
        curriculum = ("d1", "d9")
        pool = ("level3",) * 4 + ("random",)
        resolve = lambda state, name, seed: bot_action(state, "bfs", seed)
        experience = Experience([standard], 1, 0, curriculum=[curriculum], pool=pool, resolve=resolve)
        experience.pool_replacements = {"random": "bfs"}
        experience.alternate_pool_replacements = {"random": "level1"}
        experience.standard_only_opponents = {"bfs", "level1"}
        games = [experience.next_game(curriculum_fraction=1.) for _ in range(20)]
        self.assertEqual(Counter((game.opponent, game.opening) for game in games),
                         {("level3", curriculum): 16, ("bfs", standard): 2, ("level1", standard): 2})
        for first, second in zip(games[::2], games[1::2]):
            self.assertEqual((first.opponent, first.opening, first.seed), (second.opponent, second.opening, second.seed))


if __name__ == "__main__":
    unittest.main()
