import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from quoridor_lab.collect import collect
from quoridor_lab.opponents import RemoteExpert, RequestGate
from quoridor_lab.quality import E2_FAULT_PREFIXES, audit_e2, e2_quality, select_teacher_rows, export_teacher
from quoridor_lab.rules import Action, State


def sample(history, move='e2'):
    state = State.from_moves(history)
    return {'episode_id': 'example', 'ply': len(history), 'history': list(history),
            'player': state.turn, 'notation': move, 'action_id': Action.parse(move).id,
            'is_teacher': True, 'source': 'barricade:master'}


class QualityTests(unittest.TestCase):
    def test_training_export_keeps_original_bytes_and_explains_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.jsonl'
            records=[sample([]),sample(['e2','e8','e3','e7']),sample(['e2','e8','e3','hd3'])]
            for row in records:
                row['action_mask']=State.from_moves(row['history']).observation(256)['action_mask'].tolist()
            lines=[(json.dumps(r,ensure_ascii=False)+'\r\n').encode() for r in records]
            original=b''.join(lines)+b'{"unfinished":';source.write_bytes(original)
            report=export_teacher(source,root/'export')
            result=(root/'export/teacher-transitions.jsonl').read_bytes()
            self.assertEqual(result,lines[0])
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(report['selected_sha256'],hashlib.sha256(result).hexdigest())
            self.assertEqual(report['counts'],{'source_rows':3,'selected':1,'signature_or_cycle':1,'unverified_geometric_e2':1})
            self.assertFalse((root/'export/teacher-transitions.jsonl.partial').exists())
            with self.assertRaises(FileExistsError):export_teacher(source,root/'export')

    def test_third_full_position_visit_is_a_stall_not_a_pawn_only_roundtrip(self):
        history='e2,ha1,e3,e8,e2,e7,e3,e8,e2,e7,e3,e8'.split(',')
        quality=e2_quality(State.from_moves(history),Action.parse('e2'))
        self.assertEqual(quality['full_position_visits'],3)
        self.assertEqual(quality['category'],'repeated_position_e2')
        self.assertTrue(quality['quarantine'])
        # A new wall makes a different full position, even though pawns repeat.
        changed=history[:-1]+['ha2']
        quality=e2_quality(State.from_moves(changed),Action.parse('e2'))
        self.assertEqual(quality['full_position_visits'],1)
        self.assertFalse(quality['quarantine'])
        # Two visits are reported, not prematurely quarantined.
        quality=e2_quality(State.from_moves(history[:8]),Action.parse('e2'))
        self.assertEqual(quality['full_position_visits'],2)
        self.assertFalse(quality['quarantine'])

    def test_exact_signatures_and_benign_e2_are_distinguished(self):
        for prefix in E2_FAULT_PREFIXES:
            self.assertTrue(e2_quality(State.from_moves(prefix), Action.parse('e2'))['quarantine'])
        self.assertEqual(e2_quality(State(), Action.parse('e2'))['category'], 'opening_e2')
        # A blocked advance has tactical ambiguity; no extrapolation of the four signatures.
        blocked = e2_quality(State.from_moves('e2,e8,e3,hd3'.split(',')), Action.parse('e2'))
        self.assertFalse(blocked['quarantine'])
        blue_history = 'd1,e8,d2,e7,d3,e6,d4,e5,d5,e4,d6,e3,d7'.split(',')
        blue = e2_quality(State.from_moves(blue_history), Action.parse('e2'))
        self.assertEqual(blue['category'], 'progress_e2')
        self.assertEqual(blue['player'], 2)

    def test_training_preserves_raw_rows_and_only_quarantines_exact_signatures(self):
        rows = [sample([]), sample(['e2','e8','e3','e7']), sample(['e2','e8','e3','hd3'])]
        original = json.dumps(rows)
        clean, report = select_teacher_rows(rows, 'quarantine-signatures')
        self.assertEqual(clean, [rows[0], rows[2]])
        self.assertEqual(report['quarantined'], 1)
        self.assertEqual(select_teacher_rows(rows, 'raw')[0], rows)
        self.assertEqual(select_teacher_rows(rows)[0], [rows[0]])
        self.assertEqual(json.dumps(rows), original)

    def test_read_only_audit_preserves_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'transitions.jsonl'
            raw = ''.join(json.dumps(r)+'\n' for r in [sample([]),sample(['e2','e8','e3','e7'])]).encode()
            source.write_bytes(raw+b'{"unfinished":')
            report = audit_e2(source, root/'audit')
            self.assertEqual(report['counts']['quarantine'], 1)
            self.assertEqual(report['audited_prefix_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(source.read_bytes(), raw+b'{"unfinished":')
            with self.assertRaises(FileExistsError): audit_e2(source, root/'audit')


class RecoveryTests(unittest.TestCase):
    def test_cycle_reconnect_keeps_prefix_and_excludes_loop_candidate(self):
        from quoridor_lab.opponents import RemoteError
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            state=State.from_moves('e2,ha1,e3,e8,e2,e7,e3,e8,e2,e7,e3,e8'.split(','))
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[self.client('e2',calls),self.client('e4',calls)]):
                expert=RemoteExpert(data_dir=tmp,cache=False,request_gate=RequestGate(0))
                with self.assertRaises(RemoteError) as raised:expert.choose(state)
                self.assertEqual(raised.exception.kind,'suspect_e2')
                self.assertEqual(raised.exception.diagnostics['quality']['category'],'repeated_position_e2')
                self.assertEqual(expert.choose(state).action.notation,'e4')
                self.assertEqual(calls[0]['moves'],calls[1]['moves'])
                expert.close()

    def test_circuit_stops_real_collector_before_abandonment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); calls=[]
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[self.client('e2',calls) for _ in range(6)]), \
                 patch('quoridor_lab.collect.DEFERRED_RETRY_DELAY',0):
                expert=RemoteExpert(data_dir=root/'identity',cache=False,request_gate=RequestGate(0))
                try:
                    result=collect(expert,output=root/'run',games=1,opening=['e2'],max_plies=5,
                                   remote_sides=('master','master'),request_lanes=[expert],games_per_connection=1,
                                   illegal_move_retries=2,defer_illegal_responses=True,drain_pending=True,
                                   abandon_after_retries=4,progress=lambda _:None)
                finally:expert.close()
            self.assertEqual(result['error']['kind'],'e2_service_unhealthy')
            self.assertEqual(result['abandoned_games'],0)
            self.assertEqual(len(calls),6)
            self.assertFalse((root/'run/transitions.jsonl').exists())
            [checkpoint]=list((root/'run/pending').glob('*.json'))
            self.assertEqual(json.loads(checkpoint.read_text())['moves'],['e2'])
            self.assertTrue(expert.request_gate.stopped.is_set())

    def test_healthy_other_lane_does_not_hide_a_persistently_faulty_lane(self):
        from quoridor_lab.opponents import RemoteError
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]; gate=RequestGate(0)
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[self.client('e2',calls) for _ in range(6)]):
                bad=RemoteExpert(data_dir=tmp,cache=False,request_gate=gate)
                good=RemoteExpert(data_dir=tmp,cache=False,request_gate=gate)
                good.client=self.client('e8',calls)
                for i in range(6):
                    good.choose(State.from_moves(['e2']))
                    with self.assertRaises(RemoteError) as raised:bad.choose(State.from_moves(['e2']))
                    self.assertEqual(raised.exception.kind,'e2_service_unhealthy' if i==5 else 'illegal_response')
                    self.assertEqual(raised.exception.diagnostics['e2_lane_fault_streak'],i+1)
                bad.close();good.close()

    def test_shared_circuit_pauses_repeated_illegal_e2_across_lanes(self):
        from quoridor_lab.opponents import RemoteError
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]; gate=RequestGate(0)
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[self.client('e2',calls) for _ in range(6)]):
                experts=[RemoteExpert(data_dir=tmp,cache=False,request_gate=gate) for _ in range(2)]
                for i in range(6):
                    with self.assertRaises(RemoteError) as raised: experts[i%2].choose(State.from_moves(['e2']))
                    self.assertEqual(raised.exception.kind,'e2_service_unhealthy' if i==5 else 'illegal_response')
                    self.assertEqual(raised.exception.diagnostics['e2_fault_streak'],i+1)
                for expert in experts:expert.close()
            self.assertEqual(len(calls),6)

    def test_real_success_resets_fault_streak(self):
        from quoridor_lab.opponents import RemoteError
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]; gate=RequestGate(0)
            good=self.client('e8',calls)
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[self.client('e2',calls),good]):
                expert=RemoteExpert(data_dir=tmp,cache=False,request_gate=gate)
                with self.assertRaises(RemoteError):expert.choose(State.from_moves(['e2']))
                self.assertEqual(gate.e2_fault_streak,1)
                expert.choose(State.from_moves(['e2']))
                self.assertEqual(gate.e2_fault_streak,0)
                expert.close()

    def client(self, move, calls):
        client = Mock(connected=True)
        client.transport.return_value = 'websocket'
        def reply(event, payload, **kwargs):
            calls.append(payload.copy())
            return {'ok':True,'move':move,'correlationId':payload['correlationId']}
        client.call.side_effect = reply
        return client

    def run_collect(self, root, expert, history=None):
        return collect(expert, output=root/'run', games=1, opening=history or ['e2','e8','e3','e7'],
                       max_plies=5, remote_sides=('master','master'), request_lanes=[expert],
                       games_per_connection=1,
                       illegal_move_retries=2, defer_illegal_responses=True,
                       drain_pending=True, abandon_after_retries=0, progress=lambda _: None)

    def test_suspect_reply_reconnects_same_history_and_identity_without_committing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); calls=[]
            bad, good = self.client('e2',calls), self.client('e4',calls)
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[bad,good]):
                expert=RemoteExpert(data_dir=root/'identity',cache=False,request_gate=RequestGate(0),transport='polling')
                try: result=self.run_collect(root,expert)
                finally: expert.close()
            self.assertIsNone(result['error'])
            self.assertEqual([p['moves'] for p in calls],['e2,e8,e3,e7']*2)
            self.assertEqual(calls[0]['deviceId'],calls[1]['deviceId'])
            self.assertEqual(bad.connect.call_args.kwargs['transports'],['polling'])
            self.assertEqual(good.connect.call_args.kwargs['transports'],['websocket'])
            rows=[json.loads(l) for l in (root/'run/transitions.jsonl').read_text().splitlines()]
            self.assertEqual([r['notation'] for r in rows],['e4'])
            event=json.loads((root/'run/e2-events.jsonl').read_text())
            self.assertEqual(event['status'],'quarantined')
            bad.disconnect.assert_called_once()

    def test_persistent_suspect_pauses_without_abandonment_or_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); calls=[]
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[self.client('e2',calls) for _ in range(3)]):
                expert=RemoteExpert(data_dir=root/'identity',cache=False,request_gate=RequestGate(0))
                try: result=self.run_collect(root,expert)
                finally: expert.close()
            self.assertEqual(result['error']['kind'],'suspect_e2')
            self.assertEqual(len(calls),3)
            self.assertEqual(result['abandoned_games'],0)
            self.assertFalse((root/'run/transitions.jsonl').exists())
            [checkpoint]=list((root/'run/pending').glob('*.json'))
            self.assertEqual(json.loads(checkpoint.read_text())['moves'],['e2','e8','e3','e7'])
            self.assertEqual(len((root/'run/e2-events.jsonl').read_text().splitlines()),3)
            self.assertTrue(expert.request_gate.stopped.is_set())

    def test_illegal_e2_uses_new_session_on_next_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[self.client('e2',calls),self.client('e8',calls)]):
                expert=RemoteExpert(data_dir=tmp,cache=False,request_gate=RequestGate(0))
                from quoridor_lab.opponents import RemoteError
                with self.assertRaises(RemoteError): expert.choose(State.from_moves(['e2']))
                self.assertEqual(expert.choose(State.from_moves(['e2'])).action.notation,'e8')
                expert.close()
            self.assertEqual([p['moves'] for p in calls],['e2','e2'])

    def test_poisoned_cache_bypassed_but_original_file_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); calls=[]
            expert=RemoteExpert(data_dir=root,request_gate=RequestGate(0))
            state=State.from_moves(['e2','e8','e3','e7'])
            key=hashlib.sha256(json.dumps([expert.base_url,expert.difficulty,list(state.history)]).encode()).hexdigest()
            item={'key':key,'move':'e2','response':{'move':'e2'},'observed_at':'old'}
            original=json.dumps(item)+'\n'
            expert.cache_path.write_text(original); expert.cache[key]=item
            with patch('quoridor_lab.opponents.socketio.Client',return_value=self.client('e4',calls)):
                answer=expert.choose(state); expert.close()
            self.assertFalse(answer.cached)
            self.assertEqual(answer.action.notation,'e4')
            self.assertIsNotNone(answer.metadata['cache_bypassed'])
            self.assertTrue(expert.cache_path.read_text().startswith(original))

    def test_ambiguous_retreat_requires_fresh_session_check_but_can_be_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]; client=self.client('e2',calls)
            def reply(event,payload,**kwargs):
                calls.append(payload.copy())
                return {'ok':True,'move':'e8' if payload['moves']=='e2' else 'e2',
                        'correlationId':payload['correlationId']}
            client.call.side_effect=reply
            with patch('quoridor_lab.opponents.socketio.Client',return_value=client):
                expert=RemoteExpert(data_dir=tmp,cache=False,request_gate=RequestGate(0))
                answer=expert.choose(State.from_moves(['e2','e8','e3','hd3']))
                expert.close()
            self.assertEqual(answer.action.notation,'e2')
            self.assertEqual(answer.metadata['quality']['session_check'],'passed')
            self.assertEqual([p['moves'] for p in calls],['e2,e8,e3,hd3','e2,e8,e3,hd3','e2'])
            self.assertEqual(expert.network_requests,3)
            self.assertTrue(answer.metadata['quality']['independent_confirmation'])
            row=sample(['e2','e8','e3','hd3']);row['metadata']=answer.metadata;row['source']=answer.source
            self.assertEqual(select_teacher_rows([row])[0],[row])
            row['metadata']['e2_reconsideration']['candidate']['connection_id']=answer.metadata['request_diagnostics']['connection_id']
            self.assertEqual(select_teacher_rows([row])[0],[])

    def test_default_e2_session_quarantines_ambiguous_retreat_and_honors_denial(self):
        from quoridor_lab.opponents import RemoteError
        for check_response, expected in [({'ok':True,'move':'e2'},'suspect_e2'),
                                         ({'ok':False,'error':'rate_limited'},'rate_limited')]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                calls=[]; client=self.client('e2',calls)
                def reply(event,payload,**kwargs):
                    calls.append(payload.copy())
                    return {**(check_response if payload['moves']=='e2' else {'ok':True,'move':'e2'}),
                            'correlationId':payload['correlationId']}
                client.call.side_effect=reply
                with patch('quoridor_lab.opponents.socketio.Client',return_value=client):
                    expert=RemoteExpert(data_dir=tmp,cache=False,request_gate=RequestGate(0))
                    with self.assertRaises(RemoteError) as raised:
                        expert.choose(State.from_moves(['e2','e8','e3','hd3']))
                    self.assertEqual(raised.exception.kind,expected)
                    self.assertEqual(len(calls),3)
                    self.assertEqual(raised.exception.diagnostics['verification_error']['kind'],
                                     'illegal_response' if expected=='suspect_e2' else expected)
                    expert.close()

    def test_reconsideration_uses_second_remote_move_and_records_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[self.client('e2',calls),self.client('d3',calls)]):
                expert=RemoteExpert(data_dir=root/'identity',cache=False,request_gate=RequestGate(0))
                try: result=self.run_collect(root,expert,['e2','e8','e3','hd3'])
                finally:expert.close()
            self.assertIsNone(result['error'])
            row=json.loads((root/'run/transitions.jsonl').read_text())
            self.assertEqual(row['notation'],'d3')
            self.assertEqual(calls[0]['moves'],calls[1]['moves'])
            self.assertEqual(calls[0]['deviceId'],calls[1]['deviceId'])
            event=json.loads((root/'run/e2-events.jsonl').read_text())
            self.assertEqual(event['status'],'replaced')
            self.assertEqual(event['candidate']['provider_response']['move'],'e2')
            self.assertEqual(event['accepted_move'],'d3')
            self.assertNotEqual(event['candidate']['connection_id'],event['accepted_request']['connection_id'])

    def test_confirmation_denial_stops_without_accepting_original_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[];denied=self.client('e2',calls)
            def reply(event,payload,**kwargs):
                calls.append(payload.copy())
                return {'ok':False,'error':'rate_limited','correlationId':payload['correlationId']}
            denied.call.side_effect=reply
            with patch('quoridor_lab.opponents.socketio.Client',side_effect=[self.client('e2',calls),denied]):
                expert=RemoteExpert(data_dir=root/'identity',cache=False,request_gate=RequestGate(0))
                try:result=self.run_collect(root,expert,['e2','e8','e3','hd3'])
                finally:expert.close()
            self.assertEqual(result['error']['kind'],'rate_limited')
            self.assertEqual(len(calls),2)
            self.assertFalse((root/'run/transitions.jsonl').exists())
            saved=json.loads((root/'run/last-error.json').read_text())
            self.assertEqual(saved['diagnostics']['e2_reconsideration']['candidate']['provider_response']['move'],'e2')
            [checkpoint]=list((root/'run/pending').glob('*.json'))
            self.assertEqual(json.loads(checkpoint.read_text())['moves'],['e2','e8','e3','hd3'])

    def test_legacy_ambiguous_candidate_is_requeried_on_websocket(self):
        from quoridor_lab.opponents import RemoteError
        for mode, actual in [('polling','polling'),('upgrade','websocket')]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                calls=[]; bad=self.client('e2',calls); bad.transport.return_value=actual
                good=self.client('d3',calls)
                with patch('quoridor_lab.opponents.socketio.Client',side_effect=[bad,good]):
                    expert=RemoteExpert(data_dir=tmp,cache=False,request_gate=RequestGate(0),transport=mode)
                    state=State.from_moves(['e2','e8','e3','hd3'])
                    with self.assertRaises(RemoteError) as raised: expert.choose(state)
                    self.assertEqual(raised.exception.kind,'suspect_e2')
                    self.assertEqual(expert.choose(state).action.notation,'d3')
                    self.assertEqual(good.connect.call_args.kwargs['transports'],['websocket'])
                    expert.close()


if __name__=='__main__': unittest.main()
