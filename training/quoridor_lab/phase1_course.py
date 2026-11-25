"""Run the Phase 1 course one level at a time and retain its promotion evidence."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from .level1_arena import THRESHOLDS, passed
from .phase1_curriculum import LEVEL_GATES, LEVEL_NAMES, LEVEL_POOLS, checkpoint, evaluation_openings, frozen_arena, local_stage_paths
from .phase1_monitor import LOCK, PROGRESS, read_json, read_rows, writer_active
from .train_level1 import archive, evaluate, sha256
from .train_phase1 import ActorCritic
from .training_monitor import atomic_json


ARCHITECTURES = ("mlp", "cnn32")


def stage_folder(root, level1, level):
    return level1 if level == 1 else root.parent / f"{root.name}-level{level}"


def checkpoint_info(path, level):
    saved = checkpoint(path)
    state = saved["state"]
    return {"checkpoint": str(path.resolve()), "sha256": sha256(path), "level": level,
            "update": state["update"], "architecture": saved["config"]["architecture"],
            "device": saved["config"]["device"], "elo": state["event"]["elo"],
            "scores": state["event"]["formal_eval"],
            "game_units": state.get("game_units"), "gate_policy": state["event"].get("gate_policy")}


def qualification_series(run, qualified):
    evaluations = [row for row in read_rows(run / "metrics.jsonl") if row.get("formal_eval")][-3:]
    gate = qualified["gate_policy"]
    if ({row.get("opening_bank") for row in evaluations} != {1, 2, 3} or
        evaluations[-1]["update"] != qualified["update"] or
        evaluations[-1]["formal_eval"] != qualified["scores"]):
        raise ValueError(f"Qualification is missing one of the three independently paired evaluations for {run}")
    for streak, row in enumerate(evaluations, 1):
        scores = row["formal_eval"]
        if (row.get("gate_policy") != gate or row.get("qualifying_streak") != streak or
            not passed(scores, gate["wins"], gate["truncation_limit"],
                       truncation_limits=gate.get("truncation_limits")) or
            any(scores[name]["games"] != 200 or scores[name]["red_games"] != 100 or scores[name]["blue_games"] != 100
                for name in gate["wins"])):
            raise ValueError(f"Qualification evaluations do not satisfy the same gate for {run}")
    return [{"update": row["update"], "opening_bank": row["opening_bank"],
             "original_gate_passed": row["original_gate_passed"],
             "scores": {name: row["formal_eval"][name] for name in gate["wins"]}} for row in evaluations]


def best_graded_checkpoint(run, level, qualified):
    """Compare earlier and later evaluations under the same final gate."""
    original = LEVEL_GATES[level]
    gate = qualified["gate_policy"]

    def rank(row):
        scores = row["formal_eval"]
        return (passed(scores, original), passed(scores, gate["wins"], gate["truncation_limit"],
                                                 truncation_limits=gate.get("truncation_limits")),
                min(scores[name]["win_rate"] for name in original),
                -max(scores[name]["truncation_rate"] for name in original), row["elo"])

    records = [row for row in read_rows(run / "metrics.jsonl") if row.get("formal_eval")]
    best = max(records, key=rank)
    checkpoint = checkpoint_info(run / f"checkpoint-{best['update']:05d}.pt", level)
    if checkpoint["update"] != best["update"] or checkpoint["scores"] != best["formal_eval"]:
        raise ValueError(f"Best checkpoint does not match the evaluation record for {run}")
    return checkpoint


def promotions(folder, level):
    """Advance only when both architectures have a durable acceptance record."""
    results = {}
    for architecture in ARCHITECTURES:
        run = folder / f"{architecture}-seed0"
        if writer_active(run / LOCK):
            return {}
        summary = read_json(run / "summary.json")
        adjusted = read_json(run / "course-qualification.json")
        if summary.get("qualified") and (run / "qualified.pt").is_file():
            qualified = checkpoint_info(run / "qualified.pt", level)
            if qualified["update"] != summary["qualified_update"]:
                raise ValueError(f"Qualification artifact and summary disagree for {run}")
            adjustment = summary.get("gate_adjustment_reason")
        elif (level == 1 and adjusted.get("adjusted_passed") and (run / "course-qualified.pt").is_file()):
            qualified = checkpoint_info(run / "course-qualified.pt", level)
            if qualified["sha256"] != adjusted["sha256"]:
                raise ValueError(f"Adjusted qualification artifact does not match the verified checkpoint for {run}")
            adjustment = adjusted
        else:
            return {}
        if level > 1:
            original = {"wins": LEVEL_GATES[level], "truncation_limit": .05}
            if read_json(folder / "level.json").get("original_gates_only") and qualified["gate_policy"] != original:
                raise ValueError(f"Original-gate course requires the original qualification for {run}")
            best = best_graded_checkpoint(run, level, qualified)
        else:
            best_path = run / "best.pt"
            best = checkpoint_info(best_path, level) if best_path.is_file() else qualified
        results[architecture] = {"qualified": qualified, "best": best, "adjustment": adjustment,
                                 "run_id": f"phase1:{folder.name}:{run.name}"}
        if level > 1:
            results[architecture]["qualification_series"] = qualification_series(run, qualified)
    return results


def stage_plan(folder, level, arena_path, prior, stages, *, original_gates_only=False):
    historical = {}
    if level == 5:
        for completed in stages[:4]:
            for architecture, result in completed["promotion"].items():
                historical[f"{architecture}-level{completed['level']}"] = result["best"]
    plan = {"level": level, "arena": str(arena_path.resolve()), "arena_sha256": sha256(arena_path),
            "sources": {name: entry["qualified"] for name, entry in prior.items()}, "historical": historical,
            "opponent_weights": LEVEL_POOLS[level], "gates": LEVEL_GATES[level], "required_passes": 3}
    if original_gates_only:
        plan["original_gates_only"] = True
    path = folder / "level.json"
    if path.exists() and local_stage_paths(read_json(path), folder.parent) != plan:
        raise ValueError(f"Frozen Level {level} plan changed")
    if not path.exists():
        atomic_json(path, plan)


def launch(folder, architecture):
    command = [sys.executable, "-u", "-m", "quoridor_lab.phase1_curriculum", "--root", str(folder),
               "--architecture", architecture]
    with (folder / f"{architecture}.console.log").open("ab", buffering=0) as log:
        worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    (folder / f"{architecture}.pid").write_text(f"{worker.pid}\n")
    return worker


def verify_adjusted_level1_mlp(root, level1, pilot):
    """Re-evaluate the best eligible MLP using the explicitly approved Level 1 exception."""
    run = level1 / "mlp-seed0"
    if (run / "qualified.pt").is_file() or (run / "course-qualification.json").is_file():
        return
    records = read_rows(run / "metrics.jsonl")
    original = {"wins": THRESHOLDS, "truncation_limit": .05}
    adjusted = {"wins": {**THRESHOLDS, "random": .90}, "truncation_limit": .07,
                "truncation_opponents": ["random"]}

    def accepted(scores):
        return (passed(scores, {"bfs": THRESHOLDS["bfs"], "level1": THRESHOLDS["level1"]})
                and passed(scores, {"random": .90}, .07))

    eligible = [row for row in records if row.get("formal_eval") and accepted(row["formal_eval"])]
    plateau = [row for row in records if row.get("game_units") and row.get("formal_eval")][-25:]
    if not eligible or len(plateau) < 25 or records[-1]["game_units"]["completed"] < 50_000:
        raise ValueError("No evaluated checkpoint or enough logged training to apply the approved exception")
    reference = max(eligible, key=lambda row: (row["formal_eval"]["random"]["wins"],
        -row["formal_eval"]["random"]["truncated"], row["elo"]))
    update = reference["update"]
    source = run / f"checkpoint-{update:04d}.pt"
    saved = checkpoint(source)
    if saved["config"]["architecture"] != "mlp" or saved["state"]["update"] != update:
        raise ValueError("Level 1 verification checkpoint is inconsistent")
    torch.set_num_threads(saved["config"]["threads"])
    model = ActorCritic("mlp")
    model.load_state_dict(saved["model"])
    balance = saved["state"]["hyper"].get("action_family_balance", 0.)
    formal, _ = evaluate(model, read_json(level1 / "arena.json"), "cpu", balance)
    if not accepted(formal) or formal != reference["formal_eval"]:
        raise ValueError("The checkpoint did not reproduce its approved qualification results")
    independent, _ = evaluate(model, evaluation_openings(frozen_arena(root / "arena.json", level1, pilot), 0),
                              "cpu", balance)
    evidence = {"kind": "adjusted", "architecture": "mlp", "level": 1, "update": update,
                "checkpoint": str(source.resolve()), "sha256": sha256(source),
                "original": original, "adjusted": adjusted, "original_passed": passed(formal),
                "adjusted_passed": True, "formal_eval": formal, "independent_gate_bank_1": independent,
                "training_games_since_switch": records[-1]["game_units"]["completed"],
                "recent_evaluations": len(plateau),
                "recent_best_random_wins": max(row["formal_eval"]["random"]["wins"] for row in plateau),
                "reason": "按 500 盘续训超过 50,000 盘后仍未通过原门槛。仅 MLP 对 Random 使用用户批准的调整门槛。"}
    atomic_json(root / "level1-mlp-adjusted-verification.json", evidence)
    deadline = time.monotonic() + 30
    while writer_active(run / LOCK):
        if (run / "qualified.pt").is_file():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("No checkpoint boundary reached for the accepted Level 1 model")
        progress = read_json(run / PROGRESS)
        if progress.get("activity") == "collecting" and not progress.get("qualified"):
            pid = int((level1 / "mlp.pid").read_text())
            command = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True)
            if ("-m quoridor_lab.train_level1" not in command or f"--root {level1}" not in command
                or "--architecture mlp" not in command or "--until-first-pass-games 500" not in command):
                raise ValueError("The active MLP writer does not match the Level 1 run")
            os.kill(pid, signal.SIGTERM)
            break
        time.sleep(.1)
    with (run / LOCK).open("rb") as lock:
        deadline = time.monotonic() + 15
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("The MLP writer did not stop")
                time.sleep(.1)
        if (run / "qualified.pt").is_file():
            return
        archive(source, run / "course-qualified.pt")
        evidence.update(accepted_at=datetime.now(timezone.utc).isoformat(),
                        stopped_at_update=read_rows(run / "metrics.jsonl")[-1]["update"])
        atomic_json(run / "course-qualification.json", evidence)
    print(f"Level 1 MLP adjusted acceptance re-evaluated from update {update}", flush=True)


def supervise(root, level1, pilot, poll_seconds, arena_workers, *, original_gates_only=False):
    root.mkdir(parents=True, exist_ok=True)
    arena_path = root / "arena.json"
    frozen_arena(arena_path, level1, pilot)
    state_path = root / "course.json"
    original_gates_only = original_gates_only or read_json(state_path).get("original_gates_only", False)
    stages = [{"level": level, "name": name, "status": "pending", "runs": {
        architecture: f"phase1:{stage_folder(root, level1, level).name}:{architecture}-seed0"
        for architecture in ARCHITECTURES}}
        for level, name in enumerate(LEVEL_NAMES, 1)]
    state = {"protocol": "phase1-course-v1", "status": "training", "active_level": 1,
             "stages": stages, "games_per_unit": 500, "updated_at": ""}
    if original_gates_only:
        state["original_gates_only"] = True

    def publish():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, state)

    workers = {}
    failures = {}
    for stage in stages:
        level = stage["level"]
        folder = stage_folder(root, level1, level)
        state.update(active_level=level, status="training")
        stage["status"] = "training"
        if level > 1:
            folder.mkdir(parents=True, exist_ok=True)
            stage_plan(folder, level, arena_path, stages[level - 2]["promotion"], stages,
                       original_gates_only=original_gates_only)
        publish()
        while not (results := promotions(folder, level)):
            for architecture in ARCHITECTURES:
                run = folder / f"{architecture}-seed0"
                progress = read_json(run / PROGRESS)
                if ((run / "qualified.pt").is_file() and read_json(run / "summary.json").get("qualified")) or (run / "course-qualified.pt").is_file():
                    continue
                if writer_active(run / LOCK) or level == 1:
                    continue
                key = level, architecture
                previous = workers.get(key)
                if previous and previous.poll() is None:
                    continue
                if progress.get("status") == "stopped":
                    state["status"] = "paused"
                    publish()
                    return
                if previous and previous.returncode != 0:
                    failures[key] = failures.get(key, 0) + 1
                    if failures[key] >= 3:
                        raise RuntimeError(f"Level {level} {architecture} failed on three starts. See {folder}.")
                workers[key] = launch(folder, architecture)
            time.sleep(poll_seconds)
        stage.update(status="qualified", promotion=results)
        publish()
    state.update(status="arena", active_level=None)
    publish()
    from .phase1_final_arena import run_final_arena

    completed = run_final_arena(root, stages, arena_workers)
    state.update(status="completed", final=completed)
    publish()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--level1", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--arena-workers", type=int, default=4)
    parser.add_argument("--original-gates-only", action="store_true",
                        help="Use the original acceptance thresholds from Level 2 onward")
    parser.add_argument("--verify-adjusted-level1-mlp", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.arena_workers < 1:
        parser.error("Monitoring interval and Arena workers must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.verify_adjusted_level1_mlp:
        verify_adjusted_level1_mlp(args.root, args.level1, args.pilot)
        return
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    with (args.root / ".course.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            supervise(args.root, args.level1, args.pilot, args.poll_seconds, args.arena_workers,
                      original_gates_only=args.original_gates_only)
        except BaseException as exc:
            status_path = args.root / "course.json"
            state = read_json(status_path)
            state.update(status="paused" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc),
                         updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(status_path, state)
            raise


if __name__ == "__main__":
    main()
