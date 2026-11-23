"""Bounded-memory batches from frozen, whole-game JSONL records."""
import hashlib
import json

import numpy as np

from .rules import Action, State


def validate_episode(wrapper, entry):
    episode = wrapper["episode"]
    if (episode["episode_id"] != entry["episode_id"] or wrapper["split"] != entry["split"]
            or wrapper["dataset_source"] != entry["source"]):
        raise ValueError("Episode metadata differs from the frozen manifest")
    if hashlib.sha256(",".join(episode["moves"]).encode()).hexdigest() != entry["sequence_sha256"]:
        raise ValueError("Episode moves differ from the frozen manifest")
    if "master" in episode.get("config", {}).get("remote_sides", []):
        raise ValueError("Master-labelled data is excluded from training")
    return episode


class StreamingCorpus:
    def __init__(self, corpus, manifest, shuffle_buffer=8192):
        self.corpus = corpus
        self.parts = {split: [row for row in manifest["episodes"] if row["split"] == split]
                      for split in ("train", "validation")}
        self.shuffle_buffer = shuffle_buffer
        self.offsets = {}
        if any("file_offset" not in row for rows in self.parts.values() for row in rows):
            # Older corpora do not carry byte offsets. Never parse held-out moves here.
            wanted = {row["episode_id"] for rows in self.parts.values() for row in rows}
            with corpus.open("rb") as source:
                while True:
                    offset = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    wrapper = json.loads(line)
                    uid = wrapper["episode"]["episode_id"]
                    if uid in wanted:
                        self.offsets[uid] = offset
        for split in self.parts:
            if self.count(split) < 1:
                raise ValueError(f"No {split} samples selected")

    def count(self, split):
        return sum(row["plies"] for row in self.parts[split])

    @staticmethod
    def pack(rows):
        return {"board": np.stack([row[0] for row in rows]),
                "mask": np.stack([row[1] for row in rows]),
                "action": np.asarray([row[2] for row in rows]),
                "episode": np.asarray([row[3] for row in rows]),
                "ply": np.asarray([row[4] for row in rows])}

    def batches(self, split, batch_size, *, shuffle=False, seed=0, epoch=0):
        entries = self.parts[split]
        order = np.arange(len(entries))
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        if shuffle:
            rng.shuffle(order)
        capacity = max(1, (self.shuffle_buffer + batch_size - 1) // batch_size) * batch_size if shuffle else batch_size
        pending = []
        with self.corpus.open("rb") as source:
            for index in order:
                entry = entries[int(index)]
                offset = entry.get("file_offset", self.offsets.get(entry["episode_id"]))
                if offset is None:
                    raise ValueError("A frozen episode is missing from its corpus")
                source.seek(offset)
                episode = validate_episode(json.loads(source.readline()), entry)
                if len(episode["moves"]) != entry["plies"]:
                    raise ValueError("Episode length differs from the frozen manifest")
                state = State()
                for ply, move in enumerate(episode["moves"]):
                    observation = state.observation()
                    action = Action.parse(move)
                    if not observation["action_mask"][action.id]:
                        raise ValueError(f"Illegal teacher action in {episode['episode_id']} at {ply}")
                    pending.append((observation["board"], observation["action_mask"].astype(np.bool_),
                                    action.id, episode["episode_id"], ply))
                    state = state.apply(action)
                    if len(pending) == capacity:
                        if shuffle:
                            rng.shuffle(pending)
                        for start in range(0, len(pending), batch_size):
                            yield self.pack(pending[start:start + batch_size])
                        pending.clear()
        if pending:
            if shuffle:
                rng.shuffle(pending)
            for start in range(0, len(pending), batch_size):
                yield self.pack(pending[start:start + batch_size])
