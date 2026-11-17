"""Run with `python -m pytest -q test_smoke.py` from the project root."""
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app import app
from engine import compare_vectors
from video_pipeline import Tracker
from calibration import metrics

client=TestClient(app)

def test_http_home_and_settings():
    for endpoint in ('/','/api/status','/api/people','/api/models','/api/calibration/history','/api/gallery/embeddings/status'):
        assert client.get(endpoint).status_code==200

def test_invalid_file_refused():
    assert client.post('/api/analyze',files={'file':('bad.png',b'bad','image/png')}).status_code==400
    assert client.post('/api/search',files={'file':('bad.png',b'bad','image/png')}).status_code==422

def test_unsupported_empty_calibration_refused():
    assert client.post('/api/calibration/run',json={'model':'SFace'}).status_code==422

def test_similarity_math():
    a={'test':np.asarray([1.,0.],dtype=np.float32)}
    b={'test':np.asarray([0.,1.],dtype=np.float32)}
    result=compare_vectors(a,b)
    assert result['fused_similarity']==0
    assert result['scores']['test']['euclidean']>1

def test_tracking_preserves_identity():
    t=Tracker()
    f={'box':[10,10,40,40]}
    assert t.update([f])[0]['track_id']==1
    assert t.update([{'box':[12,10,40,40]}])[0]['track_id']==1

def test_metrics():
    m=metrics([.9,.8],[.1,.2],.5)
    assert m['tar']==1 and m['far']==0
