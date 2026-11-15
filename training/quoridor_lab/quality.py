"""Versioned e2 evidence; geometric warnings are not tactical verdicts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .rules import Action, State

QUALITY_VERSION = 3
# Matched requests: old route 37/40 e2, WebSocket-first 0/40 (2026-11-16).
# Exact histories only: do not generalize wall positions to arbitrary retreats.
E2_FAULT_PREFIXES = frozenset(
    ('e2', 'e8', 'e3', fourth) for fourth in ('e7', 'hc7', 've6', 'hb8')
)


def e2_quality(state: State, action: Action) -> dict:
    if action.notation != 'e2':
        return {}
    if state.error(action):
        return {'version': QUALITY_VERSION, 'category': 'illegal_e2', 'quarantine': True}
    after = state.apply(action)
    before_distance = state.distance(state.turn)
    after_distance = after.distance(state.turn)
    best = min(state.apply(Action('move', *cell)).distance(state.turn)
               for cell in state.legal_destinations())
    previous = [m for m in state.history[state.turn-1::2] if len(m) == 2]
    roundtrip = len(previous) >= 2 and previous[-2] == 'e2' and previous[-1] != 'e2'
    signature = state.turn == 1 and state.history in E2_FAULT_PREFIXES
    visits = 1
    if 'e2' in previous:
        # Both pawns, turn and all walls must match. Pawn-only backtracking after
        # a new wall is not a repeated position and must remain available.
        position = lambda s: (s.pawns, s.walls, s.remaining, s.turn)
        target = position(after)
        replay = State()
        visits += position(replay) == target
        for move in state.history:
            replay = replay.apply(move)
            visits += position(replay) == target
    cycle = visits >= 3
    flags = []
    if after_distance >= before_distance:
        flags.append('no_distance_progress')
    if after_distance > best:
        flags.append('shorter_pawn_move_available')
    if roundtrip:
        flags.append('pawn_roundtrip')
    if visits >= 2:
        flags.append('repeated_full_position')
    category = ('known_e2_fault_signature' if signature else 'repeated_position_e2' if cycle else
                'opening_e2' if not state.history else 'review_e2' if flags else 'progress_e2')
    return {'version': QUALITY_VERSION, 'category': category, 'quarantine': signature or cycle,
            'flags': flags, 'player': state.turn, 'ply': len(state.history),
            'full_position_visits': visits,
            'before_distance': before_distance, 'after_distance': after_distance,
            'best_pawn_distance': best, 'walls': len(state.walls)}


def row_e2_quality(row: dict) -> dict:
    # Recompute from original history; never trust a stale annotation to quarantine.
    if row['notation'] != 'e2':
        return {}
    state = State.from_moves(row['history'])
    action = Action.parse(row['notation'])
    if row.get('player', state.turn) != state.turn or row.get('action_id', action.id) != action.id:
        raise ValueError('e2 sample history/player/action mismatch')
    return e2_quality(state, action)


def confirmed_e2(row):
    """Require two actual same-history replies on distinct sessions, not a label."""
    metadata = row.get('metadata') or {}
    if not isinstance(metadata, dict):
        return False
    evidence = metadata.get('e2_reconsideration') or {}
    if not isinstance(evidence, dict):
        return False
    first, second = evidence.get('candidate', {}), metadata.get('request_diagnostics', {})
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    history = ','.join(row['history'])
    return (not row.get('cached', False) and evidence.get('outcome') == 'confirmed'
            and bool(metadata.get('difficulty')) and row.get('source') == 'barricade:' + metadata['difficulty']
            and bool(first.get('connection_id')) and bool(second.get('connection_id'))
            and first['connection_id'] != second['connection_id']
            and first.get('request', {}).get('difficulty') == second.get('request', {}).get('difficulty') == metadata.get('difficulty')
            and all(d.get('request', {}).get('moves') == history
                    and d.get('provider_response', {}).get('move') == 'e2'
                    and not d.get('provider_response', {}).get('error')
                    and d.get('provider_response', {}).get('ok') is not False
                    and bool(d.get('request', {}).get('correlationId'))
                    and d['request']['correlationId'] == d.get('provider_response', {}).get('correlationId')
                    and d.get('correlation_verified') is True for d in (first, second)))


def geometric_review(quality):
    return quality.get('category') == 'review_e2' and any(
        flag in quality.get('flags', []) for flag in ('no_distance_progress', 'shorter_pawn_move_available'))


def select_teacher_rows(rows, policy='verified-e2'):
    if policy not in {'verified-e2', 'quarantine-signatures', 'raw'}:
        raise ValueError('Unknown e2 training policy')
    selected = []
    counts = Counter()
    for row in rows:
        if not row['is_teacher'] or not row['source'].startswith('barricade:'):
            continue
        counts['teacher_samples_before'] += 1
        quality = row_e2_quality(row)
        if quality:
            counts[quality['category']] += 1
        if quality.get('category') == 'illegal_e2':
            raise ValueError('Dataset contains illegal e2')
        if policy != 'raw' and quality.get('quarantine'):
            counts['quarantined'] += 1
            continue
        if policy == 'verified-e2' and geometric_review(quality) and not confirmed_e2(row):
            counts['unverified_e2_excluded'] += 1
            continue
        selected.append(row)
    return selected, {'policy': policy, 'quality_version': QUALITY_VERSION,
                      **dict(counts), 'teacher_samples_after': len(selected)}


def audit_e2(source, output):
    """Read a bounded snapshot, preserving source bytes; write a separate sidecar."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise ValueError('Source must be an existing transitions JSONL file')
    output.mkdir(parents=True, exist_ok=False)
    size = source.stat().st_size
    counts, signatures = Counter(), Counter()
    digest = hashlib.sha256()
    consumed = 0
    with source.open('rb') as file, (output/'annotations.jsonl').open('x', encoding='utf-8') as annotations:
        while file.tell() < size:
            line = file.readline(size-file.tell())
            if not line.endswith(b'\n'):
                break  # Running collector may be midway through its last row.
            digest.update(line)
            consumed += len(line)
            row = json.loads(line)
            counts['rows'] += 1
            quality = row_e2_quality(row)
            if not quality:
                continue
            counts[quality['category']] += 1
            counts['e2'] += 1
            if quality.get('quarantine'):
                counts['quarantine'] += 1
                signatures[','.join(row['history'])] += 1
            teacher = bool(row.get('is_teacher')) and row.get('source', '').startswith('barricade:')
            training = ('not_teacher' if not teacher else 'exclude_signature' if quality.get('quarantine') else
                        'exclude_unverified' if geometric_review(quality) and not confirmed_e2(row) else 'include')
            counts['training_' + training] += 1
            annotations.write(json.dumps({'episode_id': row['episode_id'], 'ply': row['ply'],
                                          'history': row['history'], 'notation': row['notation'],
                                          'quality': quality, 'training_decision': training}, ensure_ascii=False)+'\n')
    report = {'version': QUALITY_VERSION, 'at': datetime.now(timezone.utc).isoformat(),
              'source': str(source), 'snapshot_bytes': size, 'audited_bytes': consumed,
              'audited_prefix_sha256': digest.hexdigest(), 'counts': dict(counts),
              'quarantined_prefixes': dict(signatures),
              'note': 'Source unchanged. Default verified-e2 training excludes known signatures, third full-position cycles, and unconfirmed geometric e2 reviews. Unconfirmed does not mean proven bad. Pure distance-progressing pawn roundtrips remain included.'}
    (output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def export_teacher(source, output, policy='verified-e2'):
    """Export original selected lines to a new dataset; never edit the source."""
    # Validate policy even when the source is empty.
    select_teacher_rows([], policy)
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise ValueError('Source must be an existing transitions JSONL file')
    output.mkdir(parents=True, exist_ok=False)
    size = source.stat().st_size
    source_hash, selected_hash = hashlib.sha256(), hashlib.sha256()
    consumed = 0
    counts = Counter()
    partial = output/'teacher-transitions.jsonl.partial'
    with source.open('rb') as original, partial.open('xb') as selected_file, \
         (output/'excluded-samples.jsonl').open('x', encoding='utf-8') as excluded:
        while original.tell() < size:
            line = original.readline(size-original.tell())
            if not line.endswith(b'\n'):
                break
            source_hash.update(line);consumed += len(line)
            row = json.loads(line)
            selected, stats = select_teacher_rows([row], policy)
            counts['source_rows'] += 1
            if selected:
                action = Action.parse(row['notation'])
                if action.id != row['action_id'] or not row['action_mask'][action.id]:
                    raise ValueError('Export encountered an inconsistent or illegal action target')
                selected_file.write(line)
                selected_hash.update(line)
                counts['selected'] += 1
            else:
                reason = ('signature_or_cycle' if stats.get('quarantined') else
                          'unverified_geometric_e2' if stats.get('unverified_e2_excluded') else 'not_teacher')
                counts[reason] += 1
                excluded.write(json.dumps({'episode_id': row['episode_id'], 'ply': row['ply'],
                                           'reason': reason}, ensure_ascii=False)+'\n')
    partial.replace(output/'teacher-transitions.jsonl')
    report = {'quality_version': QUALITY_VERSION, 'policy': policy, 'source': str(source),
              'snapshot_bytes': size, 'audited_bytes': consumed,
              'source_prefix_sha256': source_hash.hexdigest(), 'selected_sha256': selected_hash.hexdigest(),
              'counts': dict(counts), 'note': 'Selected records are byte-for-byte original lines. Exclusion is a training policy, not a claim that all excluded moves are tactically wrong.'}
    (output/'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report
