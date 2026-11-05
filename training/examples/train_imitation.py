"""Train a masked policy on observed website expert decisions, with episode-level split."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn


def network():
    return nn.Sequential(nn.Flatten(), nn.Linear(8 * 9 * 9, 256), nn.ReLU(), nn.Linear(256, 209))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/collection/transitions.jsonl")
    p.add_argument("--output", default="data/imitation.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["is_teacher"] and row["source"].startswith("barricade:")]
    if not rows:
        raise SystemExit("No observed Barricade expert decisions found")
    for row in rows:
        if not row["action_mask"][row["action_id"]]:
            raise ValueError("Dataset contains an illegal target action")
    episodes = sorted({row["episode_id"] for row in rows})
    random.Random(args.seed).shuffle(episodes)
    val_episodes = set(episodes[:max(1, len(episodes) // 5)]) if len(episodes) >= 2 else set()
    train = [row for row in rows if row["episode_id"] not in val_episodes]
    valid = [row for row in rows if row["episode_id"] in val_episodes]
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def tensors(batch):
        return (torch.tensor(np.asarray([r["observation"] for r in batch]), dtype=torch.float32),
                torch.tensor([r["action_mask"] for r in batch], dtype=torch.bool),
                torch.tensor([r["action_id"] for r in batch], dtype=torch.long))

    x, masks, y = tensors(train)
    metrics = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train))
        loss_sum = 0.0
        for indices in order.split(args.batch_size):
            logits = model(x[indices]).masked_fill(~masks[indices], -1e9)
            loss = nn.functional.cross_entropy(logits, y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        item = {"epoch": epoch + 1, "train_loss": loss_sum / len(train)}
        if valid:
            model.eval()
            vx, vm, vy = tensors(valid)
            with torch.no_grad():
                logits = model(vx).masked_fill(~vm, -1e9)
                item["validation_action_accuracy"] = (logits.argmax(1) == vy).float().mean().item()
        metrics.append(item)
        print(json.dumps(item), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "observation_shape": [8, 9, 9],
                "action_count": 209, "seed": args.seed}, output)
    output.with_suffix(".metrics.json").write_text(json.dumps({
        "teacher_samples": len(rows), "train_episodes": sorted(set(episodes) - val_episodes),
        "validation_episodes": sorted(val_episodes), "metrics": metrics,
        "note": "Action imitation accuracy is not playing strength or a recovered expert algorithm.",
    }, indent=2))


if __name__ == "__main__":
    main()
