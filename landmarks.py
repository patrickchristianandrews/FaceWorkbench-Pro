"""Optional ONNX facial landmark inference for 106-point 2D and 68-point 3D models.
Only returns observations from installed compatible model weights.
"""
import cv2,numpy as np
from registry import MODELS,MODEL_REGISTRY,settings

LANDMARK_COUNTS={'2d106det':(106,2),'1k3d68':(68,3)}

def _roi(image,box,side):
    h,w=image.shape[:2]
    x,y,bw,bh=box
    cx=x+bw/2;cy=y+bh/2
    region=max(bw,bh)*1.5
    x1=cx-region/2;y1=cy-region/2
    transform=np.array([[region/side,0,x1],[0,region/side,y1]],dtype=np.float32)
    inv=cv2.invertAffineTransform(transform)
    crop=cv2.warpAffine(image,inv,(side,side),borderMode=cv2.BORDER_REPLICATE)
    return crop,transform,region

def infer_landmarks(image,face,model):
    if model not in LANDMARK_COUNTS:raise ValueError('Unsupported landmark model')
    path=MODELS/MODEL_REGISTRY[model]
    if not path.is_file():raise FileNotFoundError(f'Install {MODEL_REGISTRY[model]} to enable landmarks')
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=settings()['cpu_threads']
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    inputs=session.get_inputs()
    if len(inputs)!=1:raise ValueError('Unsupported landmark input contract')
    inp=inputs[0];shape=inp.shape
    if len(shape)!=4 or shape[1] not in (3,'3',None):raise ValueError('Landmark model must take NCHW RGB')
    side=int(shape[2]) if isinstance(shape[2],int) else 192
    if side<64 or side>512:raise ValueError('Unsupported landmark input resolution')
    crop,transform,region=_roi(image,face['box'],side)
    blob=cv2.dnn.blobFromImage(crop,scalefactor=1.0,size=(side,side),
                              mean=(0,0,0),swapRB=True)
    prediction=np.asarray(session.run(None,{inp.name:blob})[0],dtype=np.float32).ravel()
    count,dims=LANDMARK_COUNTS[model]
    if prediction.size!=count*dims:
        raise ValueError(f'Unexpected {model} landmark output: {prediction.size} values')
    coords=prediction.reshape(count,dims).copy()
    if not np.isfinite(coords).all():raise ValueError('Non-finite landmark output')
    # InsightFace 2d106det / 1k3d68 outputs are centered normalized coordinates.
    # Apply the source model's [-1,1] -> [0,side] transform.
    coords[:,:2]=(coords[:,:2]+1)*side/2
    xy=np.hstack((coords[:,:2],np.ones((count,1),dtype=np.float32)))@transform.T
    coords[:,:2]=xy
    if dims==3:coords[:,2]=coords[:,2]*region/2
    return {'model':model,'count':count,'dimensions':dims,'points':np.round(coords,3).tolist(),
            'warning':'Landmark locations are model estimates, not ground-truth measurements.'}
