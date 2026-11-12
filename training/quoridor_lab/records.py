"""Local validated record storage; SQLite transactions serialize writes."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .rules import Action, State

INITIAL = {"pawns": ["e1", "e9"], "walls": [10, 10], "turn": 1}


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("Invalid or oversized record")
    serialized = json.dumps(record, ensure_ascii=False)
    serialized.encode("utf-8")  # Reject lone surrogates before SQLite or JSON responses.
    if len(serialized) > 128000:
        raise ValueError("Invalid or oversized record")
    version, rules = record.get("schema_version"), record.get("rules_version")
    if (version is not None and (isinstance(version, bool) or version != 1)) or (rules is not None and rules != "classic-9x9-v1"):
        raise ValueError("Unsupported record or rules version")
    initial = record.get("initial")
    if initial is not None and (not isinstance(initial, dict) or initial.get("pawns") != INITIAL["pawns"] or initial.get("walls") != INITIAL["walls"] or isinstance(initial.get("turn"), bool) or initial.get("turn") != 1):
        raise ValueError("Unsupported starting position")

    def moves(value):
        if not isinstance(value, list) or len(value) > 1000:
            raise ValueError("Invalid move list")
        return [Action.parse(item).notation for item in value]

    main = moves(record.get("moves") if record.get("moves") is not None else [])
    final = State.from_moves(main)
    branches = record.get("variations") if record.get("variations") is not None else []
    if not isinstance(branches, list) or len(branches) > 100:
        raise ValueError("Invalid variations")
    normalized, total = [], len(main)
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError("Invalid variation")
        anchor = branch.get("anchor")
        if type(anchor) is not int or not -1 <= anchor < len(main):
            raise ValueError("Invalid variation anchor")
        line = moves(branch.get("moves"))
        total += len(line)
        if not line or total > 1000:
            raise ValueError("Invalid variation length")
        State.from_moves(main[:anchor + 1] + line)
        normalized.append({"anchor": anchor, "moves": line})
    result = record.get("result")
    if result is not None:
        if not isinstance(result, dict) or "winner" not in result or type(result.get("winner")) not in (int, type(None)) or result.get("winner") not in (1, 2, None):
            raise ValueError("Invalid result")
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason or len(reason) > 64:
            raise ValueError("Invalid result reason")
        if final.winner is not None and result["winner"] != final.winner:
            raise ValueError("Result contradicts final position")
        if reason == "goal" and (final.winner is None or result["winner"] != final.winner):
            raise ValueError("Goal result contradicts final position")
        result = {"winner": result["winner"], "reason": reason}
    return {"schema_version": 1, "rules_version": "classic-9x9-v1", "initial": INITIAL,
            "moves": main, "result": result, "variations": normalized}


class RecordStore:
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / "records.sqlite3"
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, record TEXT NOT NULL)")

    def list(self):
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id, updated_at, record FROM records ORDER BY updated_at DESC").fetchall()
        return [{"id": key, "updated_at": updated, "record": json.loads(record)} for key, updated, record in rows]

    def get(self, key):
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT updated_at, record FROM records WHERE id = ?", (key,)).fetchone()
        return {"id": key, "updated_at": row[0], "record": json.loads(row[1])} if row else None

    def put(self, record, key=None):
        record = validate_record(record)
        updated = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as db:
            if key is None:
                key = uuid.uuid4().hex
                db.execute("INSERT INTO records VALUES (?, ?, ?)", (key, updated, json.dumps(record, ensure_ascii=False)))
            else:
                cursor = db.execute("UPDATE records SET updated_at = ?, record = ? WHERE id = ?", (updated, json.dumps(record, ensure_ascii=False), key))
                if not cursor.rowcount:
                    return None
        return {"id": key, "updated_at": updated, "record": record}

    def delete(self, key):
        with sqlite3.connect(self.path) as db:
            return db.execute("DELETE FROM records WHERE id = ?", (key,)).rowcount > 0
