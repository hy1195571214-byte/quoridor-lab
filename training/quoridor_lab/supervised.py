"""Monitored behavior cloning on the selected frozen corpus, with bounded-memory streaming."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .learning import network
from .rules import Action, State
from .training_monitor import TrainingJournal
from .dataset_paths import default_manifest
from .dataset_import import file_sha256
from .streaming_dataset import StreamingCorpus


def load_dataset(manifest_path, journal, sample_limit, data_mode="auto", shuffle_buffer=8192):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    corpus = manifest_path.parent / "episodes.jsonl"
    if file_sha256(corpus) != manifest["episodes_file_sha256"]:
        raise ValueError("Frozen corpus hash does not match its manifest")
    total = sum(row.get("plies", 0) for row in manifest["episodes"] if row["split"] in {"train", "validation"})
    streaming = sample_limit is None and (data_mode == "stream" or (data_mode == "auto" and total > 1_000_000))
    if streaming:
        key = hashlib.sha256(manifest_path.read_bytes() + f":absolute-v1:stream:{shuffle_buffer}".encode()).hexdigest()
        journal.update(data_mode="stream", dataset_version=manifest.get("version"),
                       dataset_manifest=str(manifest_path), prepared_samples=0)
        return StreamingCorpus(corpus, manifest, shuffle_buffer), key
    journal.update(data_mode="memory", dataset_version=manifest.get("version"), dataset_manifest=str(manifest_path))
    key = hashlib.sha256(manifest_path.read_bytes() + f":absolute-v1:{sample_limit}".encode()).hexdigest()
    cache = manifest_path.parent / f"observations-{key[:16]}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            data = {name: stored[name] for name in stored.files}
        return data, key
    expected = {row["episode_id"]: row for row in manifest["episodes"]}
    parts = {split: {name: [] for name in ["board", "mask", "action", "episode", "ply"]}
             for split in ["train", "validation"]}
    with corpus.open() as source:
        for line in source:
            wrapper = json.loads(line)
            split = wrapper["split"]
            if split not in parts:
                continue
            rows = parts[split]
            if sample_limit and len(rows["action"]) >= sample_limit:
                continue
            episode = wrapper["episode"]
            uid = episode["episode_id"]
            entry = expected[uid]
            if entry["split"] != split or entry["source"] != wrapper["dataset_source"]:
                raise ValueError("Episode metadata differs from the frozen manifest")
            if hashlib.sha256(",".join(episode["moves"]).encode()).hexdigest() != entry["sequence_sha256"]:
                raise ValueError("Episode moves differ from the frozen manifest")
            sides = episode.get("config", {}).get("remote_sides", [])
            if "master" in sides:
                raise ValueError("Master-labelled data is excluded from this training round")
            state = State()
            for ply, move in enumerate(episode["moves"]):
                observation = state.observation()
                action = Action.parse(move)
                if not observation["action_mask"][action.id]:
                    raise ValueError(f"Illegal teacher action in {uid} at {ply}")
                rows["board"].append(observation["board"])
                rows["mask"].append(observation["action_mask"].astype(np.bool_))
                rows["action"].append(action.id)
                rows["episode"].append(uid)
                rows["ply"].append(ply)
                state = state.apply(action)
                if sample_limit and len(rows["action"]) >= sample_limit:
                    break
            journal.update(prepared_samples=sum(len(part["action"]) for part in parts.values()))
    data = {}
    for split, rows in parts.items():
        if not rows["action"]:
            raise ValueError(f"No {split} samples selected")
        for name, values in rows.items():
            data[f"{split}_{name}"] = np.asarray(values)
    temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **data)
    temporary.replace(cache)
    return data, key


def sample_count(data, split):
    return data.count(split) if isinstance(data, StreamingCorpus) else len(data[f"{split}_action"])


def batches(data, split, batch_size, *, shuffle=False, seed=0, epoch=0):
    if isinstance(data, StreamingCorpus):
        yield from data.batches(split, batch_size, shuffle=shuffle, seed=seed, epoch=epoch)
        return
    count = sample_count(data, split)
    order = torch.randperm(count).numpy() if shuffle else np.arange(count)
    for start in range(0, count, batch_size):
        indices = order[start:start + batch_size]
        yield {name: data[f"{split}_{name}"][indices] for name in ("board", "mask", "action", "episode", "ply")}


def evaluate(model, data, batch_size, device):
    count = sample_count(data, "validation")
    loss_sum = correct = top3 = pawn_correct = wall_correct = pawn_count = 0
    model.eval()
    with torch.inference_mode():
        for batch in batches(data, "validation", batch_size):
            x = torch.from_numpy(batch["board"]).to(device)
            mask = torch.from_numpy(batch["mask"]).to(device)
            y = torch.from_numpy(batch["action"]).long().to(device)
            logits = model(x).masked_fill(~mask, -1e9)
            loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            matches = logits.argmax(1) == y
            correct += int(matches.sum())
            top3 += int((logits.topk(3, dim=1).indices == y[:, None]).any(1).sum())
            pawns = y < 81
            pawn_count += int(pawns.sum())
            pawn_correct += int(matches[pawns].sum())
            wall_correct += int(matches[~pawns].sum())
    return {"validation_loss": loss_sum / count,
            "validation_action_accuracy": correct / count,
            "validation_top3_accuracy": top3 / count,
            "validation_pawn_accuracy": pawn_correct / pawn_count if pawn_count else None,
            "validation_wall_accuracy": wall_correct / (count - pawn_count) if count > pawn_count else None}


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-mode", choices=["auto", "memory", "stream"], default="auto")
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", help="Readable name shown on the local training dashboard")
    parser.add_argument("--architecture", choices=["mlp", "cnn32"], default="cnn32")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warm-start", help="Load weights for a new run")
    parser.add_argument("--sample-limit", type=int, help="Smoke run only: cap samples in each development split")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads, args.shuffle_buffer) < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, threads and learning rate must be positive")
    if args.sample_limit is not None and args.sample_limit < 1:
        parser.error("Sample limit must be positive")
    if args.resume and args.warm_start:
        parser.error("Choose resume or warm-start")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    journal = TrainingJournal(args.run_dir, args.architecture, args.seed, args.epochs, args.resume)
    try:
        journal.update(device=device, learning_rate=args.learning_rate, is_smoke=bool(args.sample_limit),
                       batch_size=args.batch_size, encoding="absolute-v1", torch_version=str(torch.__version__),
                       display_name=args.name or journal.state.get("display_name") or Path(args.run_dir).name)
        data, fingerprint = load_dataset(args.manifest or default_manifest(), journal, args.sample_limit,
                                         args.data_mode, args.shuffle_buffer)
        model = network(args.architecture).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5, threshold=1e-4, threshold_mode="abs", min_lr=1e-5)
        epoch_start = global_step = bad_epochs = 0
        best_loss = early_stop_best = float("inf")
        if args.resume or args.warm_start:
            path = Path(args.run_dir) / "latest.pt" if args.resume else Path(args.warm_start)
            saved = torch.load(path, map_location=device, weights_only=True)
            if saved.get("architecture", "mlp") != args.architecture:
                raise ValueError("Checkpoint architecture does not match the selected model")
            model.load_state_dict(saved["state_dict"])
            if args.resume:
                if saved["seed"] != args.seed:
                    raise ValueError("Resume requires the original random seed")
                if saved["data_fingerprint"] != fingerprint:
                    raise ValueError("Resume requires the same frozen corpus and sample selection")
                optimizer.load_state_dict(saved["optimizer_state"])
                scheduler.load_state_dict(saved["scheduler_state"])
                epoch_start, global_step = saved["epoch"], saved["global_step"]
                best_loss, bad_epochs = saved["best_loss"], saved["bad_epochs"]
                early_stop_best = saved.get("early_stop_best", best_loss)
                torch.set_rng_state(saved["torch_random_state"].cpu())
                random.setstate(saved["python_random_state"])
                if device == "cuda" and saved.get("cuda_random_states"):
                    torch.cuda.set_rng_state_all(saved["cuda_random_states"])
                if device == "mps" and saved.get("mps_random_state") is not None:
                    torch.mps.set_rng_state(saved["mps_random_state"].cpu())
                journal.state["metrics"] = saved["metrics"]
        training_count = sample_count(data, "train")
        journal.update(status="training", train_samples=training_count,
                       validation_samples=sample_count(data, "validation"), epoch=epoch_start)
        for epoch in range(epoch_start, args.epochs):
            model.train()
            started = last_update = time.monotonic()
            loss_sum = 0.0
            total_batches = (training_count + args.batch_size - 1) // args.batch_size
            for batch_number, batch in enumerate(batches(data, "train", args.batch_size,
                    shuffle=True, seed=args.seed, epoch=epoch), 1):
                x = torch.from_numpy(batch["board"]).to(device)
                mask = torch.from_numpy(batch["mask"]).to(device)
                y = torch.from_numpy(batch["action"]).long().to(device)
                logits = model(x).masked_fill(~mask, -1e9)
                loss = torch.nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise ValueError("Training produced a non-finite loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(batch["action"])
                global_step += 1
                if batch_number == 1 or time.monotonic() - last_update >= 1 or batch_number == total_batches:
                    journal.update(epoch=epoch + 1, batch=batch_number,
                                   batches=total_batches,
                                   global_step=global_step, batch_loss=loss.item(),
                                   sample={"episode_id": str(batch["episode"][0]),
                                           "ply": int(batch["ply"][0])})
                    last_update = time.monotonic()
            metrics = {"epoch": epoch + 1, "train_loss": loss_sum / training_count,
                       "learning_rate": optimizer.param_groups[0]["lr"],
                       "seconds": time.monotonic() - started,
                       **evaluate(model, data, args.batch_size, device)}
            scheduler.step(metrics["validation_loss"])
            improved = metrics["validation_loss"] < best_loss
            best_loss = min(best_loss, metrics["validation_loss"])
            if metrics["validation_loss"] < early_stop_best - 1e-4:
                early_stop_best, bad_epochs = metrics["validation_loss"], 0
            else:
                bad_epochs += 1
            journal.metric(metrics)
            payload = {"architecture": args.architecture, "state_dict": model.state_dict(),
                       "observation_shape": [8, 9, 9], "action_count": 209,
                       "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                       "epoch": epoch + 1, "global_step": global_step, "seed": args.seed,
                       "data_fingerprint": fingerprint, "best_loss": best_loss, "bad_epochs": bad_epochs,
                       "early_stop_best": early_stop_best,
                       "torch_random_state": torch.get_rng_state(), "python_random_state": random.getstate(),
                       "cuda_random_states": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                       "mps_random_state": torch.mps.get_rng_state() if device == "mps" else None,
                       "metrics": journal.state["metrics"]}
            journal.checkpoint(payload)
            if improved:
                save_checkpoint(Path(args.run_dir) / "best.pt", payload)
            print(json.dumps(metrics), flush=True)
            if bad_epochs >= 8:
                break
        journal.update(status="completed", stopped_early=bad_epochs >= 8)
    except BaseException as error:
        journal.update(status="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        journal.close()


if __name__ == "__main__":
    main()
