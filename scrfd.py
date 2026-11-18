"""InsightFace SCRFD ONNX inference: multiscale anchor decoding and NMS.

Supports common det_10g 3-stride outputs (score, bbox, 5-point landmarks).
No synthetic detections are produced if model output contracts do not match.
"""
import cv2
import numpy as np

STRIDES=(8,16,32)

def _nms(boxes, scores, threshold=0.4):
    boxes=np.asarray(boxes,dtype=np.float32)
    scores=np.asarray(scores,dtype=np.float32)
    if len(scores)==0:return np.array([],dtype=np.int64)
    x1,y1,x2,y2=boxes.T
    area=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    order=np.argsort(scores)[::-1]
    keep=[]
    while len(order):
        i=int(order[0]);keep.append(i)
        if len(order)==1:break
        rest=order[1:]
        xx1=np.maximum(x1[i],x1[rest]);yy1=np.maximum(y1[i],y1[rest])
        xx2=np.minimum(x2[i],x2[rest]);yy2=np.minimum(y2[i],y2[rest])
        intersection=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        union=area[i]+area[rest]-intersection
        iou=intersection/np.maximum(union,1e-9)
        order=rest[iou<=threshold]
    return np.asarray(keep,dtype=np.int64)

def _parse_outputs(outputs,side):
    # SCRFD det_10g models have 9 output tensors: 3 scores, 3 boxes, 3 landmarks.
    if len(outputs)!=9:
        raise ValueError('Unsupported SCRFD output layout: expected nine score/box/landmark tensors')
    arrays=[np.asarray(out).reshape(-1,np.asarray(out).shape[-1] if np.asarray(out).ndim>1 else 1) for out in outputs]
    groups=[[],[],[]]
    for stride in STRIDES:
        expected=(side//stride)**2*2
        matching=[(i,a) for i,a in enumerate(arrays) if len(a)==expected]
        if len(matching)!=3:
            raise ValueError(f'SCRFD stride {stride} outputs do not match expected anchor count {expected}')
        for i,arr in matching:
            width=arr.shape[1]
            if width==1:groups[0].append((stride,arr))
            elif width==4:groups[1].append((stride,arr))
            elif width==10:groups[2].append((stride,arr))
            else:raise ValueError(f'Unsupported SCRFD output width {width}')
    if any(len(group)!=3 for group in groups):
        raise ValueError('Unsupported SCRFD channel arrangement')
    return {stride:tuple(next(arr for st,arr in group if st==stride) for group in groups) for stride in STRIDES}

def detect_scrfd(image, model_path, input_size=640, confidence=0.5, nms_threshold=0.4, threads=4):
    import onnxruntime as ort
    height,width=image.shape[:2]
    side=int(input_size)
    if side not in (320,640,1024):raise ValueError('SCRFD input size must be 320, 640, or 1024')
    scale=min(side/width,side/height)
    rw,rh=max(1,int(round(width*scale))),max(1,int(round(height*scale)))
    canvas=np.zeros((side,side,3),dtype=np.uint8)
    canvas[:rh,:rw]=cv2.resize(image,(rw,rh))
    # SCRFD uses BGR input normalized around 127.5.
    blob=cv2.dnn.blobFromImage(canvas,scalefactor=1/128.0,size=(side,side),
                               mean=(127.5,127.5,127.5),swapRB=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=int(threads)
    session=ort.InferenceSession(str(model_path),sess_options=opts,providers=['CPUExecutionProvider'])
    inputs=session.get_inputs()
    if len(inputs)!=1:raise ValueError('Unsupported SCRFD number of input tensors')
    outputs=session.run(None,{inputs[0].name:blob})
    branches=_parse_outputs(outputs,side)
    all_boxes=[];all_scores=[];all_kps=[]
    for stride,(score,bbox,kps) in branches.items():
        grid=side//stride
        centers=np.stack(np.meshgrid(np.arange(grid),np.arange(grid)),axis=-1).reshape(-1,2)
        centers=np.repeat(centers,2,axis=0).astype(np.float32)*stride
        conf=score.reshape(-1)
        selected=np.flatnonzero(conf>=confidence)
        for i in selected:
            center=centers[i]
            left,top,right,bottom=bbox[i]*stride
            corners=np.array([center[0]-left,center[1]-top,
                              center[0]+right,center[1]+bottom],dtype=np.float32)/scale
            points=(kps[i].reshape(5,2)*stride+center)/scale
            all_boxes.append(corners);all_scores.append(float(conf[i]));all_kps.append(points)
    results=[]
    for i in _nms(all_boxes,all_scores,nms_threshold):
        x1,y1,x2,y2=all_boxes[i]
        x1=float(np.clip(x1,0,width-1));y1=float(np.clip(y1,0,height-1))
        x2=float(np.clip(x2,0,width));y2=float(np.clip(y2,0,height))
        if x2<=x1 or y2<=y1:continue
        results.append({'box':[int(x1),int(y1),int(x2-x1),int(y2-y1)],
                        'confidence':round(all_scores[i],4),
                        'keypoints':all_kps[i].tolist()})
    return results
