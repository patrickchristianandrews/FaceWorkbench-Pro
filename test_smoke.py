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

def test_scrfd_multiscale_outputs_and_nms():
    from scrfd import _parse_outputs,_nms
    import numpy as np
    side=320
    outputs=[]
    for width in (1,4,10):
        for stride in (8,16,32):
            outputs.append(np.zeros((1,(side//stride)**2*2,width),dtype=np.float32))
    parsed=_parse_outputs(outputs,side)
    assert sorted(parsed)==[8,16,32]
    assert all(len(x)==3 for x in parsed.values())
    indices=_nms([[0,0,10,10],[1,1,11,11],[30,30,40,40]],[0.9,0.8,0.7],0.3)
    assert indices.tolist()==[0,2]

def test_empirical_far_reporting():
    from calibration import tar_at_far
    rows=tar_at_far([0.9,0.8],[0.1,0.2,0.3],targets=(0.1,0.01))
    assert len(rows)==2
    assert rows[0]['observed_far']==0
    assert rows[1]['sample_sufficient_for_target'] is False

def test_npz_import_refuses_unknown_face():
    import io,numpy as np
    import database
    mem=io.BytesIO()
    np.savez_compressed(mem,**{'99999999_SFace':np.array([1.,0.],dtype=np.float32)})
    result=database.import_vectors_npz(mem.getvalue())
    assert result['imported']==0
    assert result['errors']

def test_packaged_uvicorn_config_without_console(monkeypatch):
    """PyInstaller --windowed sets stderr/stdout to None; constructing the server must work."""
    import desktop,sys
    monkeypatch.setattr(sys,'stdout',None)
    monkeypatch.setattr(sys,'stderr',None)
    server=desktop.create_server(0)
    assert server.config.log_config is None
    assert server.config.access_log is False
