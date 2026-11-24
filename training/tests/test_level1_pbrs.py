import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Install learning dependencies for Level 1 PPO tests")
class Level1PBRSTests(unittest.TestCase):
    def test_potential_telescopes_at_natural_end_and_at_the_ply_cap(self):
        from quoridor_lab.level1_arena import Game, bot_action
        from quoridor_lab.train_level1 import bfs_goal_potential

        gamma, scale = .99, .02
        for max_plies, endings in ((256, {"wins", "losses"}), (4, {"truncated"})):
            with self.subTest(max_plies=max_plies), patch("quoridor_lab.level1_arena.MAX_PLIES", max_plies):
                game = Game.start((), 1, "bfs", 0)
                initial = bfs_goal_potential(game.state, 1, scale)
                discounted_bonus = 0.
                for index in range(max_plies):
                    before = bfs_goal_potential(game.state, 1, scale)
                    self.assertAlmostEqual(before, -scale * game.state.distance(1))
                    result = game.step(bot_action(game.state, "bfs", 10))
                    after = 0. if result else bfs_goal_potential(game.state, 1, scale)
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
        torch.manual_seed(31)
        shaped, shaped_results, opponents, *_ = Experience([("e2", "e8")], 4, 0).collect(
            model, {**hyper, "pbrs_scale": .02}, "cpu"
        )
        self.assertEqual(shaped_results, results)
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(raw[:4], shaped[:4])))
        self.assertFalse(torch.allclose(raw[-1], shaped[-1]))
        self.assertGreater(sum(row.get("pbrs_nonzero", 0) for row in opponents.values()), 0)


if __name__ == "__main__":
    unittest.main()
