import json, os, sys
from pathlib import Path
ROOT = Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
DATA = Path(os.environ.get('DATA_DIR',ROOT/'data'))
MODELS = Path(os.environ.get('MODELS_DIR',ROOT/'models'))
DATA.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
CONFIG = ROOT/'config.json'
DEFAULT={'threshold':0.55,'detector':'yunet','detection_size':640,'cpu_threads':4,'active_packs':['SFace'],'flip_augmentation':True,'model_weights':{}}
MODEL_REGISTRY = {
'antelopev2':'glintr100.onnx','AdaFace':'adaface_ir101.onnx','AuraFace':'auraface_r100.onnx',
'buffalo_l':'w600k_r50.onnx','buffalo_s':'w600k_mbf.onnx','buffalo_sc':'buffalo_sc_rec.onnx',
'ElasticFace':'elasticface.onnx','MagFace':'magface.onnx','SFace':'face_recognition_sface_2021dec.onnx',
'YuNet':'face_detection_yunet_2023mar.onnx','raccoon_s':'raccoon_s.onnx','raccoon_l':'raccoon_l.onnx',
'cheetah_s':'cheetah_s.onnx','cheetah_l':'cheetah_l.onnx','1k3d68':'1k3d68.onnx',
'emotion':'emotion-ferplus-8.onnx','mask':'mask_detector.onnx','MiniFASNet':'minifasnet_v2.onnx'}
# Listing a model does not imply it is licensed, downloaded or operational.
PUBLIC_FETCH = {
'YuNet':'https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
'SFace':'https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'}
def settings():
    try: return {**DEFAULT,**json.loads(CONFIG.read_text())}
    except (OSError,ValueError): return DEFAULT.copy()
def save_settings(value):
    current=settings()
    if 'threshold' in value: current['threshold']=max(0.1,min(0.95,float(value['threshold'])))
    if value.get('detector') in ('haar','yunet','scrfd'): current['detector']=value['detector']
    if 'detection_size' in value and int(value['detection_size']) in (320,640,1024): current['detection_size']=int(value['detection_size'])
    if 'cpu_threads' in value: current['cpu_threads']=max(1,min(32,int(value['cpu_threads'])))
    if 'active_packs' in value: current['active_packs']=[x for x in value['active_packs'] if x in MODEL_REGISTRY]
    if 'flip_augmentation' in value: current['flip_augmentation']=bool(value['flip_augmentation'])
    if 'model_weights' in value:
        if not isinstance(value['model_weights'],dict):raise ValueError('Model weights must be an object')
        current['model_weights']={k:max(0.001,min(100.0,float(v))) for k,v in value['model_weights'].items()
                                  if k in MODEL_REGISTRY and k not in ('YuNet','emotion','mask','MiniFASNet','1k3d68')}
    CONFIG.write_text(json.dumps(current,indent=2)); return current
def model_status():
    return [{'key':key,'filename':filename,'installed':(MODELS/filename).is_file(),'fetchable':key in PUBLIC_FETCH,'active':key in settings().get('active_packs',[]),'recognition':key in ('antelopev2','AdaFace','AuraFace','buffalo_l','buffalo_s','buffalo_sc','ElasticFace','MagFace','SFace','raccoon_s','raccoon_l','cheetah_s','cheetah_l')} for key,filename in MODEL_REGISTRY.items()]
def fetch_model(key):
    if key not in PUBLIC_FETCH: raise ValueError('No verified public download URL for this model')
    import requests
    dest=MODELS/MODEL_REGISTRY[key]; tmp=dest.with_suffix('.partial')
    try:
        with requests.get(PUBLIC_FETCH[key],stream=True,timeout=40) as resp:
            resp.raise_for_status()
            with tmp.open('wb') as fp:
                for chunk in resp.iter_content(1024*256):
                    if chunk: fp.write(chunk)
        if tmp.stat().st_size < 100000: raise ValueError('Downloaded weight file is unexpectedly small')
        tmp.replace(dest)
    finally: tmp.unlink(missing_ok=True)
    return {'model':key,'size_bytes':dest.stat().st_size}
