"""Replay PGN from the CPP ZIP into a new, frozen, provenance-preserving corpus."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
import zipfile

from .rules import Action, State

TRAINING_ROOT = Path(__file__).resolve().parent.parent
NUMBERING = re.compile(r"\b\d+\.(?:\.\.)?")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pgn(raw):
    tokens = NUMBERING.sub(" ", raw.decode("utf-8-sig")).split()
    if tokens and tokens[-1] in RESULTS:
        tokens.pop()
    if not tokens:
        raise ValueError("Empty PGN")
    return [Action.parse(token).notation for token in tokens]


def sequence_hash(moves):
    return hashlib.sha256(",".join(moves).encode()).hexdigest()


def reflection_group(moves):
    mirrored = [
        chr(105 - (ord(move[0]) - 97)) + move[1] if len(move) == 2 else
        move[0] + chr(104 - (ord(move[1]) - 97)) + move[2]
        for move in moves
    ]
    return hashlib.sha256(min(",".join(moves), ",".join(mirrored)).encode()).hexdigest()


def new_split(group, seed):
    bucket = int(hashlib.sha256(f"cpp-split:{seed}:{group}".encode()).hexdigest(), 16) / 2**256
    return "train" if bucket < 0.8 else "validation" if bucket < 0.9 else "test"


def import_corpus(archive, base_manifest, output, seed=0):
    archive, base_manifest, output = map(Path, (archive, base_manifest, output))
    if output.exists():
        raise FileExistsError(f"Frozen dataset already exists: {output}")
    base = json.loads(base_manifest.read_text())
    base_corpus = base_manifest.parent / "episodes.jsonl"
    if file_sha256(base_corpus) != base["episodes_file_sha256"]:
        raise ValueError("Base corpus does not match its frozen manifest")
    verified_download = json.loads(archive.with_name("download-status.json").read_text())
    archive_hash = file_sha256(archive)
    if verified_download["state"] != "complete" or archive_hash != verified_download["sha256"]:
        raise ValueError("CPP archive differs from the verified download")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building")
    staging.mkdir()  # Never overwrite an incomplete or concurrently running build.
    started = time.monotonic()
    entries, exact, groups = [], {}, {}
    summary = {split: {"episodes": 0, "plies": 0, "by_source": {}, "winner_counts": {}}
               for split in ("train", "validation", "test")}
    counts = Counter()
    corpus_digest = hashlib.sha256()

    def append(target, wrapper):
        episode = wrapper["episode"]
        if "master" in episode.get("config", {}).get("remote_sides", []):
            raise ValueError("Master-labelled data must remain excluded")
        moves = episode["moves"]
        key, group = sequence_hash(moves), reflection_group(moves)
        if key in exact:
            raise ValueError("Duplicate passed the import filter")
        split, source = wrapper["split"], wrapper["dataset_source"]
        if group in groups and groups[group] != split:
            raise ValueError("A complete-game reflection group crosses partitions")
        groups[group], exact[key] = split, episode["episode_id"]
        offset = target.tell()
        encoded = (json.dumps(wrapper, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        target.write(encoded)
        corpus_digest.update(encoded)
        entries.append({"episode_id": episode["episode_id"], "source": source, "split": split,
                        "plies": len(moves), "winner": episode.get("winner"),
                        "sequence_sha256": key, "reflection_group_sha256": group,
                        "file_offset": offset})
        row = summary[split]
        row["episodes"] += 1
        row["plies"] += len(moves)
        source_row = row["by_source"].setdefault(source, {"episodes": 0, "plies": 0})
        source_row["episodes"] += 1
        source_row["plies"] += len(moves)
        winner = str(episode.get("winner"))
        row["winner_counts"][winner] = row["winner_counts"].get(winner, 0) + 1

    def progress(phase, **fields):
        value = {"phase": phase, **dict(counts), **fields,
                 "elapsed_seconds": round(time.monotonic() - started, 1)}
        (staging / "progress.json").write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps(value), flush=True)

    progress("replaying_all_games")
    with (staging / "episodes.jsonl").open("wb") as target, \
            (staging / "rejected.jsonl").open("w") as rejected, \
            (staging / "duplicates.jsonl").open("w") as duplicates:
        with base_corpus.open() as source:
            for line in source:
                wrapper = json.loads(line)
                state = State.from_moves(wrapper["episode"]["moves"])
                if state.winner != wrapper["episode"].get("winner"):
                    raise ValueError("Base episode outcome differs from rule replay")
                append(target, wrapper)
                counts["base_games_retained"] += 1
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                path = PurePosixPath(member.filename)
                if member.is_dir() or path.suffix.lower() != ".pgn" or "__MACOSX" in path.parts or path.name.startswith("._"):
                    continue
                counts["pgn_scanned"] += 1
                raw = source.read(member)
                try:
                    moves = parse_pgn(raw)
                    if len(moves) > 256:
                        raise ValueError("Game exceeds the agreed 256-ply limit")
                    key = sequence_hash(moves)
                    if key in exact:
                        duplicates.write(json.dumps({"source_file": member.filename,
                            "retained_episode_id": exact[key], "sequence_sha256": key}) + "\n")
                        counts["duplicate_records_removed"] += 1
                        continue
                    state = State.from_moves(moves)
                    if state.winner is None:
                        raise ValueError("No goal finish; result cannot be inferred from move-only PGN")
                    group = reflection_group(moves)
                    split = groups.get(group, new_split(group, seed))
                    final = state.to_dict()
                    final.pop("moves")
                    episode = {
                        "schema_version": 1, "episode_id": "quoridorcpp-pgn:" + key,
                        "source": "huggingface:KibleLab/Quoridor-Zero/quoridorCPP 20260614121829 208K.zip",
                        "source_file": member.filename, "source_file_sha256": hashlib.sha256(raw).hexdigest(),
                        "moves": moves, "ply_count": len(moves), "winner": state.winner,
                        "winner_provenance": "inferred_from_replayed_goal_position",
                        "finished_reason": "goal", "status": "completed", "final": final,
                        "legality": "full_game_replayed_with_project_rules",
                        "requested_difficulties": None, "players": None, "ratings": None,
                        "config": {"max_plies": 256, "opponent": "quoridorcpp:unspecified"},
                    }
                    append(target, {"dataset_source": "quoridorcpp-pgn", "split": split, "episode": episode})
                    counts["cpp_games_added"] += 1
                    counts["cpp_plies_added"] += len(moves)
                except ValueError as error:
                    rejected.write(json.dumps({"source_file": member.filename, "error": str(error)}) + "\n")
                    counts["rejected_records"] += 1
                if counts["pgn_scanned"] % 10000 == 0:
                    progress("replaying_all_games")
    manifest = {
        "version": "round2-cpp208k-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Combined behavior-cloning corpus; no training results are implied.",
        "parent_manifest": str(base_manifest.resolve()), "parent_manifest_sha256": file_sha256(base_manifest),
        "source_paths_relative_to": base.get("source_paths_relative_to", str(TRAINING_ROOT.resolve())),
        "source_paths_training_root": str(TRAINING_ROOT.resolve()),
        "sources": {**base.get("sources", {}), "quoridorcpp-pgn": {
            "archive": str(archive.resolve()), "sha256": archive_hash,
            "format_selected": "PGN only; TSV and history representations are not counted again",
            "player_identities_ratings_and_generation_method": "unknown_from_move_only_PGN"}},
        "sample_seed": seed,
        "split_method": "Preserve every parent split. New full-game left-right reflection groups use SHA256(cpp-split:seed:group) buckets 80/10/10; overlapping groups inherit parent split.",
        "splits": summary, "episodes_file_sha256": corpus_digest.hexdigest(),
        "integrity": {"full_rule_replay": True, "goal_finish_required": True,
                      "exact_duplicate_sequences": 0, "cross_split_reflection_groups": 0,
                      "reflection_groups": len(groups), "master_labelled_episodes": 0,
                      "cross_split_position_overlap_audited": False},
        "import_counts": dict(counts), "episodes": entries,
        "pipeline_status": "Frozen replay-checked episode corpus; tensors are generated by the training loader.",
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    progress("complete", splits=summary, episodes_sha256=manifest["episodes_file_sha256"])
    staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=TRAINING_ROOT / "data/datasets/quoridorcpp-208k/quoridorcpp-208k.zip")
    parser.add_argument("--base-manifest", type=Path, default=TRAINING_ROOT / "data/datasets/round1-2500/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import_corpus(args.archive, args.base_manifest, args.output, args.seed)


if __name__ == "__main__":
    main()
