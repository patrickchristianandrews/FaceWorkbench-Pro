"""Local face tracking and privacy-focused MP4 redaction.
Identity-selective blurring is deliberately limited to reference identities already
in the local gallery. Detection false negatives require manual output review.
"""
import cv2,uuid
from pathlib import Path
from registry import DATA
from engine import detect,redact,extract_embeddings,search_vectors
EXPORTS=DATA/'exports';EXPORTS.mkdir(parents=True,exist_ok=True)

def iou(a,b):
    x,y,w,h=a;u,v,p,q=b
    left,top,right,bottom=max(x,u),max(y,v),min(x+w,u+p),min(y+h,v+q)
    area=max(0,right-left)*max(0,bottom-top)
    return area/float(w*h+p*q-area+1e-6)

class Tracker:
    def __init__(self,max_age=12):self.tracks={};self.next_id=1;self.max_age=max_age
    def update(self,faces):
        used=set()
        for face in faces:
            options=sorted([(iou(face['box'],t['box']),tid) for tid,t in self.tracks.items() if tid not in used],reverse=True)
            if options and options[0][0]>.25:
                tid=options[0][1];old=self.tracks[tid]['box']
                # smooth bounding boxes for stable video blurring
                box=[round(.7*a+.3*b) for a,b in zip(face['box'],old)]
            else:tid=self.next_id;self.next_id+=1;box=face['box']
            used.add(tid);self.tracks[tid]={'box':box,'age':0,'person_id':self.tracks.get(tid,{}).get('person_id')}
            face['track_id']=tid;face['box']=box
        for tid in list(self.tracks):
            if tid not in used:self.tracks[tid]['age']+=1
            if self.tracks[tid]['age']>self.max_age:del self.tracks[tid]
        return faces

def redact_video(input_path,mode='blur',progress=None,selected_ids=None):
    cap=cv2.VideoCapture(str(input_path))
    if not cap.isOpened():raise ValueError('Video could not be opened')
    w,h=int(cap.get(3)),int(cap.get(4));fps=float(cap.get(5)) or 25.
    if w<=0 or h<=0 or w*h>8_300_000:raise ValueError('Unsupported video dimensions')
    path=EXPORTS/f'{uuid.uuid4().hex}.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise ValueError('MP4 encoder unavailable')
    tracker=Tracker();frames=0;selected_ids=set(int(x) for x in (selected_ids or []));track_hits={}
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            faces=tracker.update(detect(frame))
            if mode in ('blur-matched','keep-matched'):
                for face in faces:
                    track_id=face['track_id']
                    if frames%12==0 or track_id not in track_hits:
                        vectors,errors=extract_embeddings(frame,face)
                        matches=search_vectors(vectors,top_k=1) if vectors else []
                        if matches:track_hits[track_id]=matches[0]['person_id']
                if mode=='blur-matched':faces=[f for f in faces if track_hits.get(f['track_id']) in selected_ids]
                else:faces=[f for f in faces if track_hits.get(f['track_id']) not in selected_ids]
            writer.write(redact(frame,faces,'mosaic' if mode=='mosaic' else 'blur'));frames+=1
            if progress:progress(frames)
            if frames>18000:raise ValueError('Video exceeds 18,000-frame processing limit')
    finally:cap.release();writer.release()
    return path,frames
