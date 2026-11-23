"""A transport override persists through resume and reaches every request lane."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from quoridor_lab.dashboard import CampaignConfig, RunConfig, ResumeConfig, campaign_runs, create_dashboard
from tests.test_backlog_dashboard import OfflineExpert, saved_run


class TransportDashboardTests(unittest.TestCase):
    def test_campaign_preserves_transport_and_invalid_modes_fail(self):
        batches = list(campaign_runs(CampaignConfig(games_per_matchup=2, request_transport='websocket', proxy_env=True)))
        self.assertTrue(all(config['request_transport']=='websocket' for _,config in batches))
        self.assertTrue(all(config['proxy_env'] for _,config in batches))
        for cls in [RunConfig, ResumeConfig, CampaignConfig]:
            with self.assertRaises(ValueError):
                cls(request_transport='invalid')

    def test_resume_passes_selected_transport_to_every_lane_and_keeps_history(self):
        captured=[]
        done=threading.Event()
        def factory(**kwargs):
            captured.append(kwargs)
            return OfflineExpert(**kwargs)
        def finish(expert,**kwargs):
            done.set()
            return {'network_requests':0,'time_limit_reached':False,
                    'error':{'kind':'stopped','message':'Offline check'}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run_id,folder=saved_run(root)
            marker=folder/'pending'/'preserve.txt'
            marker.write_text('preserve existing evidence')
            app=create_dashboard(root,data_dir=root,replay_base='http://localhost',expert_factory=factory)
            with patch('quoridor_lab.dashboard.collect',side_effect=finish), TestClient(app) as api:
                response=api.post(f'/api/runs/{run_id}/resume',json={'request_transport':'websocket','proxy_env':True})
                self.assertEqual(response.status_code,200)
                self.assertTrue(done.wait(2))
                deadline=time.monotonic()+2
                while api.get(f'/api/runs/{run_id}').json()['status']=='running':
                    self.assertLess(time.monotonic(),deadline)
                    time.sleep(.01)
            self.assertEqual(len(captured),2)
            self.assertTrue(all(c['transport']=='websocket' for c in captured))
            self.assertTrue(all(c['proxy_env'] is True for c in captured))
            self.assertEqual(json.loads((folder/'run.json').read_text(encoding='utf-8'))['config']['request_transport'],'websocket')
            self.assertEqual(marker.read_text(),'preserve existing evidence')


if __name__=='__main__':
    unittest.main()
