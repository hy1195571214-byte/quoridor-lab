"""Transport selection and redacted success/failure evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from quoridor_lab.opponents import RemoteExpert, RemoteError
from quoridor_lab.rules import State


class RemoteTransportTests(unittest.TestCase):
    def test_proxy_choice_controls_both_http_and_websocket(self):
        for enabled in [False, True]:
            with self.subTest(proxy_env=enabled), tempfile.TemporaryDirectory() as root:
                client = Mock(connected=True)
                with patch('quoridor_lab.opponents.socketio.Client', return_value=client) as factory:
                    expert = RemoteExpert(data_dir=root, proxy_env=enabled)
                    expert._connect()
                    kwargs = factory.call_args.kwargs
                    self.assertEqual(kwargs['http_session'].trust_env, enabled)
                    options = kwargs['websocket_extra_options']
                    self.assertEqual(options.get('http_no_proxy'), None if enabled else ['*'])
                    expert.close()

    def test_default_is_direct_and_reconnect_changes_only_session_identifier(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = Mock(connected=True), Mock(connected=True)
            for client in [first, second]:
                client.transport.return_value = 'websocket'
                client.call.side_effect = lambda event, payload, **kwargs: {
                    'ok': True, 'move': 'e8', 'correlationId': payload['correlationId']}
            with patch('quoridor_lab.opponents.socketio.Client', side_effect=[first, second]):
                expert = RemoteExpert(data_dir=root, cache=False, interval=.05)
                original = expert.choose(State.from_moves(['e2'])).metadata['request_diagnostics']
                self.assertEqual(first.connect.call_args.kwargs['transports'], ['websocket'])
                first.call.side_effect = TimeoutError('connection lost')
                with self.assertRaises(RemoteError):
                    expert.choose(State.from_moves(['e2']))
                recovered = expert.choose(State.from_moves(['e2'])).metadata['request_diagnostics']
                self.assertNotEqual(original['connection_id'], recovered['connection_id'])
                self.assertEqual(recovered['connection_request'], 1)
                self.assertEqual(first.connect.call_args.kwargs['auth'], second.connect.call_args.kwargs['auth'])
                self.assertEqual(original['request']['moves'], recovered['request']['moves'])
                expert.close()

    def test_requested_transport_and_single_origin(self):
        for mode, transports in [('polling', ['polling']), ('websocket', ['websocket']),
                                 ('upgrade', ['polling', 'websocket'])]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                client = Mock(connected=True)
                client.transport.return_value = 'polling' if mode == 'polling' else 'websocket'
                client.call.side_effect = lambda event, payload, **kwargs: {
                    'ok': True, 'move': 'e8', 'correlationId': payload['correlationId']}
                with patch('quoridor_lab.opponents.socketio.Client', return_value=client) as factory:
                    expert = RemoteExpert(data_dir=root, cache=False, transport=mode)
                    decision = expert.choose(State.from_moves(['e2']))
                    connection = client.connect.call_args.kwargs
                    self.assertEqual(connection['transports'], transports)
                    self.assertEqual(connection['headers'], {'Origin': 'https://barricade.gg'})
                    self.assertEqual(factory.call_args.kwargs['websocket_extra_options'],
                                     {'suppress_origin': True, 'http_no_proxy': ['*']})
                    details = decision.metadata['request_diagnostics']
                    self.assertEqual(details['request']['moves'], 'e2')
                    self.assertEqual(details['player'], 2)
                    self.assertEqual(details['ply'], 1)
                    self.assertEqual(details['transport'], client.transport())
                    self.assertTrue(details['correlation_verified'])
                    self.assertTrue(details['connection_id'])
                    self.assertEqual(details['connection_request'], 1)
                    self.assertNotIn('deviceId', details['request'])
                    self.assertNotIn(expert.device_id, repr(details))
                    previous_connection = details['connection_id']
                    client.call.side_effect = lambda event, payload, **kwargs: {
                        'ok': True, 'move': 'e2', 'correlationId': payload['correlationId']}
                    with self.assertRaises(RemoteError) as raised:
                        expert.choose(State.from_moves(['e2']))
                    self.assertEqual(raised.exception.diagnostics['connection_id'], previous_connection)
                    self.assertEqual(raised.exception.diagnostics['connection_request'], 2)
                    expert.close()

    def test_unknown_transport_fails_before_creating_identity(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                RemoteExpert(data_dir=Path(root)/'unused', transport='unknown')
            self.assertFalse((Path(root)/'unused').exists())


if __name__ == '__main__':
    unittest.main()
