"""Small REINFORCE integration example; use local opponents before remote training."""
import argparse
import json
from pathlib import Path

import torch
from torch.distributions import Categorical

from quoridor_lab.env import QuoridorEnv
from quoridor_lab.opponents import LocalPolicy, RemoteExpert
from train_imitation import network


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--remote", action="store_true")
    p.add_argument("--warm-start", help="Optional checkpoint produced by train_imitation.py")
    p.add_argument("--output", default="data/reinforce.pt")
    p.add_argument("--max-plies", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    model = network()
    if args.warm_start:
        model.load_state_dict(torch.load(args.warm_start, weights_only=True)["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    opponent = RemoteExpert() if args.remote else LocalPolicy("shortest", args.seed)
    env = QuoridorEnv(opponent=opponent, max_plies=args.max_plies)
    metrics = []
    try:
        for episode in range(args.episodes):
            env.learner_player = 1 + episode % 2
            obs, info = env.reset(seed=args.seed + episode)
            logs, entropies, rewards = [], [], []
            while True:
                x = torch.tensor(obs["board"]).unsqueeze(0)
                mask = torch.tensor(obs["action_mask"], dtype=torch.bool)
                logits = model(x).squeeze(0).masked_fill(~mask, -1e9)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                obs, reward, terminated, truncated, info = env.step(int(action))
                logs.append(distribution.log_prob(action))
                entropies.append(distribution.entropy())
                rewards.append(reward)
                if terminated or truncated:
                    break
            returns, value = [], 0.0
            for reward in reversed(rewards):
                value = reward + .99 * value
                returns.insert(0, value)
            loss = -(torch.stack(logs) * torch.tensor(returns)).mean() - .001 * torch.stack(entropies).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            row = {"episode": episode, "reward": sum(rewards), "winner": env.state.winner,
                   "truncated": truncated, "plies": len(env.state.history), "loss": loss.item()}
            metrics.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "observation_shape": [8, 9, 9],
                "action_count": 209, "seed": args.seed}, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
