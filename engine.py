import cv2, numpy as np
from registry import MODELS,settings,MODEL_REGISTRY
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

def decode(raw):
    arr=np.frombuffer(raw,np.uint8); im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if im is None: raise ValueError('Not a supported image')
    if im.shape[0]*im.shape[1]>36_000_000: raise ValueError('Image exceeds 36 MP processing limit')
    return im

def detect(image):
    conf=settings(); h,w=image.shape[:2]; results=[]
    if conf['detector']=='scrfd':
        from scrfd import detect_scrfd
        model_file=MODELS/MODEL_REGISTRY['SCRFD']
        if not model_file.is_file():
            raise RuntimeError('SCRFD model weights are not installed. Select YuNet or install det_10g.onnx.')
        results=detect_scrfd(image,model_file,conf['detection_size'],threads=conf['cpu_threads'])
    elif conf['detector']=='yunet' and (MODELS/MODEL_REGISTRY['YuNet']).exists():
        detector=cv2.FaceDetectorYN.create(str(MODELS/MODEL_REGISTRY['YuNet']),'',(w,h),score_threshold=0.7,nms_threshold=0.3)
        _, faces=detector.detect(image)
        for f in (faces if faces is not None else []):
            x,y,bw,bh=map(int,f[:4]); results.append({'box':[max(0,x),max(0,y),max(1,bw),max(1,bh)],'confidence':round(float(f[-1]),3),'keypoints':[[float(f[i]),float(f[i+1])] for i in (4,6,8,10,12)]})
    else:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for x,y,bw,bh in HAAR.detectMultiScale(gray,scaleFactor=1.12,minNeighbors=5,minSize=(35,35)):
            results.append({'box':[int(x),int(y),int(bw),int(bh)],'confidence':None,'keypoints':[]})
    for face in results:
        x,y,bw,bh=face['box']; crop=image[y:min(y+bh,h),x:min(x+bw,w)]
        if not crop.size: continue
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY); sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var()); bright=float(gray.mean()); minimum=min(bw,bh)
        face['quality']={'face_px':minimum,'sharpness':round(sharp,1),'brightness':round(bright,1),'flags':(["Small face"] if minimum<64 else [])+(["Blur"] if sharp<45 else [])+(["Exposure"] if bright<35 or bright>225 else []),'score':round(float(np.clip(.4*minimum/128+.35*sharp/150+.25*(1-abs(bright-128)/128),0,1)),3)}
    return results

def redact(image,boxes,mode='blur'):
    output=image.copy()
    for f in boxes:
        x,y,w,h=f['box']; roi=output[y:y+h,x:x+w]
        if roi.size:
            if mode=='mosaic':
                small=cv2.resize(roi,(max(1,w//16),max(1,h//16)))
                roi[:]=cv2.resize(small,(roi.shape[1],roi.shape[0]),interpolation=cv2.INTER_NEAREST)
            else: roi[:]=cv2.GaussianBlur(roi,(0,0),sigmaX=max(w,h)/8)
    return output

# Model inference is intentionally fail-closed: unavailable weights never generate dummy vectors.
import threading
_SESSIONS={}
_LOCK=threading.Lock()
ARCFACE_REF=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],dtype=np.float32)

def align(image,face):
    pts=face.get('keypoints') or []
    if len(pts)==5:
        # OpenCV YuNet returns right/left eye, nose, right/left mouth landmarks.
        # Infer eye ordering in image coordinates so reference points are left-to-right.
        ordered=np.asarray(pts,dtype=np.float32)
        eyes=sorted(ordered[:2].tolist(),key=lambda p:p[0]);mouth=sorted(ordered[3:].tolist(),key=lambda p:p[0])
        src=np.asarray([*eyes,ordered[2],*mouth],dtype=np.float32)
        affine,_=cv2.estimateAffinePartial2D(src,ARCFACE_REF,method=cv2.LMEDS)
        if affine is not None: return cv2.warpAffine(image,affine,(112,112),borderMode=cv2.BORDER_REFLECT_101)
    x,y,w,h=face['box'];pad=int(max(w,h)*.12);x=max(0,x-pad);y=max(0,y-pad)
    crop=image[y:min(image.shape[0],y+h+2*pad),x:min(image.shape[1],x+w+2*pad)]
    if not crop.size:raise ValueError('Empty face crop')
    return cv2.resize(crop,(112,112))

def _onnx_session(model):
    from registry import MODELS,MODEL_REGISTRY
    path=MODELS/MODEL_REGISTRY[model]
    if not path.is_file():raise RuntimeError(f'{model} weights are missing. Install/download them in System & Models.')
    try:import onnxruntime as ort
    except ImportError as e:raise RuntimeError('ONNX Runtime is not installed') from e
    with _LOCK:
        key=(str(path),int(path.stat().st_mtime),settings()['cpu_threads'])
        if key not in _SESSIONS:
            opt=ort.SessionOptions();opt.intra_op_num_threads=key[2]
            _SESSIONS[key]=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])
        return _SESSIONS[key]

def enabled_recognizers():
    from registry import MODELS,MODEL_REGISTRY
    keys=settings().get('active_packs') or ['SFace']
    return [key for key in keys if key in MODEL_REGISTRY and key in RECOGNITION_MODELS and (MODELS/MODEL_REGISTRY[key]).exists()]

RECOGNITION_MODELS={'antelopev2','AdaFace','AuraFace','buffalo_l','buffalo_s','buffalo_sc','ElasticFace','MagFace','SFace','raccoon_s','raccoon_l','cheetah_s','cheetah_l'}

def embedding(image,face,model):
    """Extract a normalized face descriptor with optional flip augmentation."""
    if model not in RECOGNITION_MODELS:raise ValueError('Not a face recognition model')
    crop=align(image,face)
    augment=bool(settings().get('flip_augmentation',True))
    crops=[crop,cv2.flip(crop,1)] if augment else [crop]
    vectors=[]
    if model=='SFace':
        from registry import MODELS,MODEL_REGISTRY
        recognizer=cv2.FaceRecognizerSF.create(str(MODELS/MODEL_REGISTRY['SFace']),'')
        for view in crops:
            vectors.append(np.asarray(recognizer.feature(view),dtype=np.float32).reshape(-1))
    else:
        session=_onnx_session(model)
        inp=session.get_inputs()[0];shape=inp.shape
        if len(shape)!=4 or shape[1] not in (3,'3',None):raise ValueError(f'Unsupported ONNX input for {model}: {shape}')
        size=112 if not isinstance(shape[2],int) else shape[2]
        for view in crops:
            rgb=cv2.cvtColor(cv2.resize(view,(size,size)),cv2.COLOR_BGR2RGB)
            tensor=((rgb.astype(np.float32)-127.5)/127.5).transpose(2,0,1)[None]
            vectors.append(np.asarray(session.run(None,{inp.name:tensor})[0],dtype=np.float32).ravel())
    normalized=[]
    for vector in vectors:
        if vector.size not in (128,256,512):raise ValueError(f'Unexpected recognition output size: {vector.size}')
        norm=np.linalg.norm(vector)
        if not np.isfinite(norm) or norm<1e-8:raise ValueError('Invalid ONNX embedding')
        normalized.append(vector/norm)
    fused=np.mean(normalized,axis=0)
    final_norm=np.linalg.norm(fused)
    if not np.isfinite(final_norm) or final_norm<1e-8:raise ValueError('Invalid augmented face embedding')
    return (fused/final_norm).astype(np.float32)

def extract_embeddings(image,face,models=None):
    vectors={};warnings={}
    for model in models if models is not None else enabled_recognizers():
        try: vectors[model]=embedding(image,face,model)
        except Exception as e:warnings[model]=str(e)
    return vectors,warnings

def compare_vectors(a,b):
    common=set(a)&set(b);scores={}
    for model in sorted(common):
        if a[model].shape==b[model].shape:
            scores[model]=float(np.clip(np.dot(a[model],b[model]),-1,1))
    if not scores:raise ValueError('No compatible recognition model embeddings')
    weights=settings().get('model_weights',{})
    fusion=float(np.average(list(scores.values()),weights=[max(0.001,float(weights.get(m,1))) for m in scores]))
    return {'fused_similarity':round(fusion,5),'threshold':settings()['threshold'],
            'above_threshold':fusion>=settings()['threshold'],
            'scores':{m:{'cosine':round(v,5),'euclidean':round(float(np.linalg.norm(a[m]-b[m])),5),'weight':max(0.001,float(weights.get(m,1))),
                          'above_threshold':v>=settings()['threshold']} for m,v in scores.items()},
            'note':'Cosine threshold is not a calibrated probability of identity'}

def search_vectors(probe,top_k=10,unfiltered=False):
    from database import embeddings
    candidates={}
    for model,vec in probe.items():
        for item in embeddings(model):
            other=item['vector']
            if other.shape!=vec.shape:continue
            score=float(np.dot(vec,other))
            entry=candidates.setdefault(item['person_id'],{'person_id':item['person_id'],'name':item['name'],'scores':{}})
            entry['scores'].setdefault(model,[]).append(score)
    results=[]
    for item in candidates.values():
        by_model={k:max(v) for k,v in item['scores'].items()}
        weights=settings().get('model_weights',{})
        s=float(np.average(list(by_model.values()),weights=[max(0.001,float(weights.get(m,1))) for m in by_model]))
        if unfiltered or s>=settings()['threshold']:
            results.append({'person_id':item['person_id'],'name':item['name'],'similarity':round(s,5),
                            'models':{k:round(v,5) for k,v in by_model.items()},'above_threshold':s>=settings()['threshold']})
    return sorted(results,key=lambda x:x['similarity'],reverse=True)[:max(1,min(100,top_k))]
