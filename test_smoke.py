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

def test_weighted_similarity_and_flip_setting(monkeypatch):
    import engine
    monkeypatch.setattr(engine,'settings',lambda:{'threshold':0.55,'model_weights':{'A':3,'B':1},'flip_augmentation':True})
    a={'A':np.array([1.,0.],np.float32),'B':np.array([1.,0.],np.float32)}
    b={'A':np.array([1.,0.],np.float32),'B':np.array([0.,1.],np.float32)}
    result=engine.compare_vectors(a,b)
    assert result['fused_similarity']==0.75
    assert result['scores']['A']['weight']==3

def test_settings_accept_recognition_weights():
    from registry import save_settings,settings,CONFIG
    original=CONFIG.read_bytes() if CONFIG.exists() else None
    try:
        value=save_settings({'flip_augmentation':False,'model_weights':{'SFace':2.5}})
        assert value['flip_augmentation'] is False
        assert value['model_weights']['SFace']==2.5
    finally:
        if original is None: CONFIG.unlink(missing_ok=True)
        else: CONFIG.write_bytes(original)
