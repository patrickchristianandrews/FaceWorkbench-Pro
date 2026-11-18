import io,json,tempfile,os,uuid
from pathlib import Path
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import database
from engine import decode,detect,redact,extract_embeddings,search_vectors,compare_vectors,enabled_recognizers
from registry import ROOT,DATA,MODEL_REGISTRY,settings,save_settings,model_status,fetch_model
from osint_helper import metadata,SITES
from video_pipeline import redact_video

STATIC=Path(getattr(__import__('sys'),'_MEIPASS',ROOT))/'static'
app=FastAPI(title='FaceWorkbench Pro',version='0.1.0')
app.mount('/static',StaticFiles(directory=STATIC),name='static')
@app.get('/')
def index(): return FileResponse(STATIC/'index.html')
@app.get('/manifest.json')
def manifest(): return FileResponse(STATIC/'manifest.json')
@app.get('/sw.js')
def sw(): return FileResponse(STATIC/'sw.js',media_type='text/javascript')
@app.get('/api/status')
def status(): return {'version':'0.1.0','settings':settings(),'models':model_status(),'people':len(database.people()),'capabilities':['detect','quality','gallery','verify','search','embeddings','calibration','tracking','selective-redaction','exif','backup']}
@app.post('/api/analyze')
async def analyze(file:UploadFile=File(...),dense_landmarks:bool=Form(False),landmarks_3d:bool=Form(False)):
    try:
        raw=await file.read(); image=decode(raw); faces=detect(image)
        if dense_landmarks or landmarks_3d:
            from landmarks import infer_landmarks
            for face in faces:
                errors={}
                for requested,key in ((dense_landmarks,'2d106det'),(landmarks_3d,'1k3d68')):
                    if not requested:continue
                    try:face[key]=infer_landmarks(image,face,key)
                    except Exception as error:errors[key]=str(error)
                if errors:face['landmark_warnings']=errors
        return {'faces':faces,'count':len(faces),'recognition_available':bool(enabled_recognizers()),'recognition_models':enabled_recognizers()}
    except Exception as e: raise HTTPException(400,str(e))
@app.post('/api/redact-image')
async def redact_image(file:UploadFile=File(...),style:str=Form('blur')):
    try:
        image=decode(await file.read()); faces=detect(image); output=redact(image,faces,'mosaic' if style=='mosaic' else 'blur'); ok,buf=cv2.imencode('.png',output)
        if not ok: raise ValueError('Cannot encode PNG')
        return StreamingResponse(io.BytesIO(buf.tobytes()),media_type='image/png',headers={'Content-Disposition':'attachment; filename=redacted.png'})
    except Exception as e: raise HTTPException(400,str(e))
@app.get('/api/people')
def people(): return database.people()
@app.post('/api/people')
async def enroll(name:str=Form(...),files:list[UploadFile]=File(...)):
    pid=database.add_person(name); added=0; errors=[]
    for file in files:
        try:
            im=decode(await file.read()); faces=detect(im)
            if not faces: errors.append(f'{file.filename}: no face detected'); continue
            best=max(faces,key=lambda x:x['quality']['score']); x,y,w,h=best['box']
            crop=im[y:y+h,x:x+w]; fname=database.add_face(pid,cv2.resize(crop,(112,112)),best['quality']['score']); added+=1
            fid=database.face_id_from_crop(fname)
            vectors,model_errors=extract_embeddings(im,best)
            for key,vec in vectors.items():database.save_embedding(fid,key,vec)
            if model_errors:errors.append(f'{file.filename}: recognition model errors: {model_errors}')
        except Exception as e: errors.append(f'{file.filename}: {e}')
    if not added:
        database.delete(pid); raise HTTPException(422,{'message':'No detectable faces','errors':errors})
    return {'id':pid,'added':added,'warnings':errors}
@app.get('/api/people/{pid}/samples')
def samples(pid:int): return database.samples(pid)
@app.get('/api/crops/{filename}')
def crop(filename:str):
    if Path(filename).name!=filename: raise HTTPException(400,'Invalid filename')
    file=database.CROPS/filename
    if not file.exists(): raise HTTPException(404,'Not found')
    return FileResponse(file)
@app.patch('/api/people/{pid}')
def rename(pid:int,body:dict):
    if not str(body.get('name','')).strip(): raise HTTPException(400,'Name required')
    database.rename(pid,body['name']); return {'ok':True}
@app.delete('/api/people/{pid}')
def delete(pid:int): database.delete(pid); return {'ok':True}
@app.delete('/api/samples/{sid}')
def delete_sample(sid:int): database.delete_sample(sid); return {'ok':True}
@app.post('/api/people/merge')
def merge(body:dict):
    try: database.merge(int(body['source']),int(body['target']))
    except Exception as e: raise HTTPException(400,str(e))
    return {'ok':True}
@app.get('/api/settings')
def get_settings(): return settings()
@app.put('/api/settings')
def put_settings(body:dict):
    try: return save_settings(body)
    except Exception as e: raise HTTPException(400,str(e))
@app.get('/api/models')
def models(): return model_status()
@app.post('/api/models/{key}/download')
def download_model(key:str):
    try: return fetch_model(key)
    except Exception as e: raise HTTPException(400,str(e))
@app.post('/api/metadata')
async def exif(file:UploadFile=File(...)):
    try: return {'metadata':metadata(await file.read()),'sites':SITES}
    except Exception as e: raise HTTPException(400,str(e))
@app.post('/api/video/redact')
async def video(file:UploadFile=File(...),style:str=Form('blur'),selected_ids:str=Form('[]')):
    suffix=Path(file.filename or 'input.mp4').suffix.lower()
    if suffix not in ('.mp4','.mov','.avi','.mkv'): raise HTTPException(400,'Unsupported video extension')
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as t:
        temp=Path(t.name)
        try:
            total=0
            while chunk:=await file.read(1024*1024):
                total+=len(chunk)
                if total>400*1024*1024: raise ValueError('Video exceeds 400 MB')
                t.write(chunk)
        except Exception as e:
            temp.unlink(missing_ok=True); raise HTTPException(413,str(e))
    try:
        output,frames=redact_video(temp,style if style in ('blur','mosaic','blur-matched','keep-matched') else 'blur',selected_ids=json.loads(selected_ids))
        return FileResponse(output,media_type='video/mp4',filename='redacted.mp4')
    except Exception as e: raise HTTPException(400,str(e))
    finally: temp.unlink(missing_ok=True)
@app.get('/api/backup')
def backup():
    file=DATA/'gallery-backup.zip'; database.backup(file); return FileResponse(file,filename='FaceWorkbench-gallery-backup.zip')


@app.post('/api/verify')
async def verify(a:UploadFile=File(...),b:UploadFile=File(...)):
    try:
        ia,ib=decode(await a.read()),decode(await b.read())
        fa,fb=detect(ia),detect(ib)
        if not fa or not fb:raise ValueError('Both photos must contain detectable faces')
        va,ea=extract_embeddings(ia,max(fa,key=lambda x:x['quality']['score']))
        vb,eb=extract_embeddings(ib,max(fb,key=lambda x:x['quality']['score']))
        if not va or not vb:raise ValueError(f'No functioning face recognition models: {ea} {eb}')
        return {'result':compare_vectors(va,vb),'warnings':{'a':ea,'b':eb},'models':list(va.keys())}
    except Exception as e:raise HTTPException(422,str(e))

@app.post('/api/search')
async def gallery_search(file:UploadFile=File(...),top_k:int=Form(10),lookalikes:bool=Form(False)):
    try:
        im=decode(await file.read());faces=detect(im)
        if not faces:raise ValueError('No detectable faces')
        vectors,errors=extract_embeddings(im,max(faces,key=lambda x:x['quality']['score']))
        if not vectors:raise ValueError(f'No working recognizers installed: {errors}')
        return {'results':search_vectors(vectors,top_k,lookalikes),'models':list(vectors),'warnings':errors,
                'note':'Visual similarity rankings are not verified identification or identity probabilities.'}
    except Exception as e:raise HTTPException(422,str(e))

@app.post('/api/people/{pid}/photos')
async def add_photos(pid:int,files:list[UploadFile]=File(...)):
    if not any(p['id']==pid for p in database.people()):raise HTTPException(404,'Person not found')
    added=0;warnings=[]
    for file in files:
        try:
            im=decode(await file.read());faces=detect(im)
            if not faces:raise ValueError('No detectable face')
            best=max(faces,key=lambda f:f['quality']['score']);x,y,w,h=best['box']
            crop=im[y:y+h,x:x+w];fid=database.face_id_from_crop(database.add_face(pid,cv2.resize(crop,(112,112)),best['quality']['score']))
            vectors,errors=extract_embeddings(im,best)
            for key,vec in vectors.items():database.save_embedding(fid,key,vec)
            warnings.extend(errors.values());added+=1
        except Exception as e:warnings.append(f'{file.filename}: {e}')
    return {'added':added,'warnings':warnings}

@app.post('/api/gallery/reindex')
def reindex():return database.rebuild_indexes()

@app.get('/api/gallery/vectors')
def vectors(format:str='json'):
    if format not in ('json','npz'):raise HTTPException(400,'Choose json or npz')
    path=DATA/('vectors.'+format)
    database.export_vectors(path,format)
    return FileResponse(path,filename=path.name)

@app.get('/api/calibration/history')
def calibration_history():return database.latest_calibration()

@app.post('/api/calibration/run')
def calibration_run(body:dict):
    from calibration import calibrate
    try:return calibrate(str(body['model']),float(body.get('target_far',0.01)))
    except Exception as e:raise HTTPException(422,str(e))

@app.post('/api/calibration/stacker')
def calibration_stacker(body:dict):
    from calibration import train_stacker
    try:return train_stacker(body['models'])
    except Exception as e:raise HTTPException(422,str(e))

@app.get('/api/gallery/embeddings/status')
def embeddings_status():return {'models':database.indexed_models(),'active':enabled_recognizers()}

@app.post('/api/gallery/backfill')
def backfill_embeddings():
    # Extract stored crops for newly installed models. Alignment may be less accurate than original full-resolution photos.
    from engine import embedding
    active=enabled_recognizers();total=0;errors=[]
    with database.connect() as c:faces=[dict(r) for r in c.execute('SELECT id,crop FROM faces')]
    from database import CROPS
    for row in faces:
        img=cv2.imread(str(CROPS/row['crop']))
        if img is None:continue
        face={'box':[0,0,img.shape[1],img.shape[0]],'keypoints':[]}
        for model in active:
            try:database.save_embedding(row['id'],model,embedding(img,face,model));total+=1
            except Exception as e:errors.append(f"Face {row['id']} {model}: {e}")
    database.rebuild_indexes()
    return {'embedded':total,'errors':errors[:100], 'note':'For best accuracy, re-enroll original full-resolution photos with five-point alignment.'}

@app.get('/api/clusters')
def clusters(model:str='SFace',eps:float=.33,min_samples:int=2):
    try:return {'clusters':database.unknown_clusters(model,eps,min_samples),'note':'Clusters contain existing labeled face samples; automatic unknown-face ingestion is not implemented.'}
    except Exception as e:raise HTTPException(422,str(e))

@app.post('/api/gallery/vectors/import')
async def import_vectors(file:UploadFile=File(...)):
    raw=await file.read()
    if len(raw)>30*1024*1024:raise HTTPException(413,'Maximum import size 30 MB')
    try:
        if (file.filename or '').lower().endswith('.npz'):
            output=database.import_vectors_npz(raw)
        elif (file.filename or '').lower().endswith('.json'):
            output=database.import_vectors_json(json.loads(raw))
        else:
            raise ValueError('Import requires a .json or .npz file')
        database.rebuild_indexes()
        return output
    except Exception as e:raise HTTPException(422,str(e))

@app.post('/api/people/batch-folder')
def batch_folder(body:dict):
    # Filesystem access only via user-selected local folder in native desktop window.
    root=Path(body.get('folder','')).expanduser().resolve()
    if not root.is_dir():raise HTTPException(404,'Folder not found')
    # Restricted to immediate identity-name subdirectories; no recursive/symlink traversal.
    results=[]
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.is_symlink():continue
        photos=[f for f in folder.iterdir() if f.is_file() and not f.is_symlink() and f.suffix.lower() in ('.png','.jpg','.jpeg','.webp')][:1000]
        if not photos:continue
        pid=database.add_person(folder.name);n=0;errs=[]
        for file in photos:
            try:
                im=decode(file.read_bytes());fs=detect(im)
                if not fs:raise ValueError('No face detected')
                best=max(fs,key=lambda f:f['quality']['score']);x,y,w,h=best['box']
                fname=database.add_face(pid,cv2.resize(im[y:y+h,x:x+w],(112,112)),best['quality']['score'])
                fid=database.face_id_from_crop(fname)
                vectors,warnings=extract_embeddings(im,best)
                for key,vec in vectors.items():database.save_embedding(fid,key,vec)
                errs.extend(warnings.values());n+=1
            except Exception as e:errs.append(f'{file.name}: {e}')
        if not n:database.delete(pid)
        results.append({'name':folder.name,'added':n,'warnings':errs[:10]})
    database.rebuild_indexes()
    return {'people':results}

@app.post('/api/gallery/duplicate-check')
async def duplicate_check(file:UploadFile=File(...)):
    """Offer a pre-enrollment similarity warning; never assert a confirmed identity."""
    try:
        image=decode(await file.read())
        faces=detect(image)
        if not faces:
            raise ValueError('No detectable face')
        vectors,errors=extract_embeddings(image,max(faces,key=lambda f:f['quality']['score']))
        if not vectors:
            return {'available':False,'candidates':[],'warnings':errors,
                    'note':'Duplicate checking requires installed and enabled recognition weights.'}
        candidates=search_vectors(vectors,top_k=5,unfiltered=True)
        return {'available':True,
                'candidates':[c for c in candidates if c['above_threshold']],
                'warnings':errors,'note':'Similarity warning only; confirm with the person before merging records.'}
    except Exception as e:
        raise HTTPException(422,str(e))
