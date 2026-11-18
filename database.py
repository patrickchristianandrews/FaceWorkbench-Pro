import sqlite3,datetime,zipfile
from pathlib import Path
from registry import DATA,CONFIG
DB=DATA/'gallery.db'; CROPS=DATA/'crops'; CROPS.mkdir(parents=True,exist_ok=True)
def connect():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; c.execute('PRAGMA foreign_keys=ON'); return c
def init():
    with connect() as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY,name TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS faces(id INTEGER PRIMARY KEY,person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,crop TEXT NOT NULL,quality REAL NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS model_embeddings(id INTEGER PRIMARY KEY,face_id INTEGER REFERENCES faces(id) ON DELETE CASCADE,model TEXT NOT NULL,vector BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS calibration_history(id INTEGER PRIMARY KEY,created_at TEXT NOT NULL,model TEXT,threshold REAL,metrics_json TEXT);''')
def people():
    with connect() as c: return [dict(x) for x in c.execute('''SELECT p.id,p.name,p.created_at,count(f.id) samples,round(avg(f.quality),3) quality FROM people p LEFT JOIN faces f ON p.id=f.person_id GROUP BY p.id ORDER BY p.name''')]
def add_person(name):
    if not name.strip(): raise ValueError('Name required')
    with connect() as c:
        cur=c.execute('INSERT INTO people(name,created_at) VALUES(?,?)',(name.strip(),datetime.datetime.now(datetime.timezone.utc).isoformat())); return cur.lastrowid
def add_face(person_id,crop,quality):
    import cv2,uuid
    fname=f'{uuid.uuid4().hex}.png'; cv2.imwrite(str(CROPS/fname),crop)
    with connect() as c:
        c.execute('INSERT INTO faces(person_id,crop,quality,created_at) VALUES(?,?,?,?)',(person_id,fname,quality,datetime.datetime.now(datetime.timezone.utc).isoformat()))
    return fname
def samples(person_id):
    with connect() as c: return [dict(x) for x in c.execute('SELECT id,crop,quality FROM faces WHERE person_id=?',(person_id,))]
def rename(person_id,name):
    with connect() as c: c.execute('UPDATE people SET name=? WHERE id=?',(name.strip(),person_id))
def delete(person_id):
    files=[x['crop'] for x in samples(person_id)]
    with connect() as c: c.execute('DELETE FROM people WHERE id=?',(person_id,))
    for file in files: (CROPS/file).unlink(missing_ok=True)
def delete_sample(sid):
    with connect() as c:
        item=c.execute('SELECT crop FROM faces WHERE id=?',(sid,)).fetchone()
        c.execute('DELETE FROM faces WHERE id=?',(sid,))
    if item: (CROPS/item['crop']).unlink(missing_ok=True)
def merge(source,target):
    if source==target: raise ValueError('Choose distinct identities')
    with connect() as c:
        c.execute('UPDATE faces SET person_id=? WHERE person_id=?',(target,source)); c.execute('DELETE FROM people WHERE id=?',(source,))
def backup(path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        if DB.exists(): z.write(DB,'gallery.db')
        if CONFIG.exists(): z.write(CONFIG,'config.json')
        for p in CROPS.glob('*'): z.write(p,f'crops/{p.name}')
    return path
init()

# Embeddings persist in SQLite as the source of truth; FAISS is a derived cache.
def save_embedding(face_id, model, vector):
    import numpy as np
    v=np.asarray(vector,dtype=np.float32).ravel()
    norm=float(np.linalg.norm(v))
    if not np.isfinite(v).all() or norm<1e-8: raise ValueError('Invalid embedding')
    v=v/norm
    with connect() as c:
        c.execute('DELETE FROM model_embeddings WHERE face_id=? AND model=?',(face_id,model))
        c.execute('INSERT INTO model_embeddings(face_id,model,vector) VALUES(?,?,?)',(face_id,model,v.tobytes()))

def embeddings(model=None):
    import numpy as np
    with connect() as c:
        q='''SELECT e.id,e.face_id,e.model,e.vector,f.person_id,p.name
        FROM model_embeddings e JOIN faces f ON f.id=e.face_id JOIN people p ON p.id=f.person_id'''
        records=c.execute(q+(' WHERE e.model=?' if model else ''),((model,) if model else ())).fetchall()
    return [{**dict(r),'vector':np.frombuffer(r['vector'],dtype=np.float32).copy()} for r in records]

def face_id_from_crop(crop):
    with connect() as c:
        r=c.execute('SELECT id FROM faces WHERE crop=?',(crop,)).fetchone()
    return int(r['id']) if r else None

def indexed_models():
    with connect() as c:
        return [dict(x) for x in c.execute('SELECT model,count(*) n FROM model_embeddings GROUP BY model')]

def rebuild_indexes():
    import numpy as np
    from registry import DATA
    path=DATA/'indexes';path.mkdir(exist_ok=True,parents=True)
    try: import faiss
    except ImportError: return {'faiss_available':False,'indexed':{},'message':'SQLite vectors remain usable for exact numpy search'}
    counts={}
    for old in path.glob('*.faiss'):old.unlink()
    for model in sorted({r['model'] for r in embeddings()}):
        entries=embeddings(model)
        if not entries:continue
        arr=np.stack([x['vector'] for x in entries]).astype('float32')
        index=faiss.IndexFlatIP(arr.shape[1]);index.add(arr)
        safe=''.join(c for c in model if c.isalnum() or c in '_-')
        faiss.write_index(index,str(path/f'{safe}.faiss'))
        (path/f'{safe}.ids.json').write_text(__import__('json').dumps([x['face_id'] for x in entries]))
        counts[model]=len(entries)
    return {'faiss_available':True,'indexed':counts}

def export_vectors(path,format='json'):
    import numpy as np,json
    rows=embeddings()
    if format=='npz':
        np.savez_compressed(path,**{f"{r['face_id']}_{r['model']}":r['vector'] for r in rows})
    else:
        Path(path).write_text(json.dumps([{'face_id':r['face_id'],'person_id':r['person_id'],'name':r['name'],'model':r['model'],'vector':r['vector'].tolist()} for r in rows],indent=2))
    return len(rows)

def record_calibration(model, threshold, metrics):
    import json
    with connect() as c:
        c.execute('INSERT INTO calibration_history(created_at,model,threshold,metrics_json) VALUES(?,?,?,?)',
                  (datetime.datetime.now(datetime.timezone.utc).isoformat(),model,threshold,json.dumps(metrics)))

def latest_calibration():
    import json
    with connect() as c:
        rows=c.execute('SELECT * FROM calibration_history ORDER BY id DESC LIMIT 10').fetchall()
    return [{**dict(r),'metrics':json.loads(r['metrics_json'])} for r in rows]

def unknown_clusters(model,eps=.33,min_samples=2):
    from sklearn.cluster import DBSCAN
    import numpy as np
    rows=embeddings(model)
    if len(rows)<min_samples:return []
    vectors=np.stack([r['vector'] for r in rows])
    labels=DBSCAN(eps=eps,min_samples=min_samples,metric='cosine').fit_predict(vectors)
    result=[]
    for group in sorted(set(labels)-{-1}):
        members=[{'face_id':r['face_id'],'person_id':r['person_id'],'name':r['name']} for i,r in enumerate(rows) if labels[i]==group]
        result.append({'cluster_id':int(group),'size':len(members),'members':members})
    return result

def import_vectors_json(data):
    import numpy as np
    # Only attach vectors to face IDs that already exist, ensuring person references cannot be spoofed.
    count=0;errors=[]
    if not isinstance(data,list) or len(data)>50000:raise ValueError('Expected JSON array of at most 50,000 vector records')
    with connect() as c: valid={int(r[0]) for r in c.execute('SELECT id FROM faces')}
    for i,row in enumerate(data):
        try:
            fid=int(row['face_id']);model=str(row['model']);vec=np.asarray(row['vector'],dtype=np.float32)
            if fid not in valid or not model or not 8<=vec.size<=4096:raise ValueError('Invalid face ID, model, or vector length')
            save_embedding(fid,model,vec);count+=1
        except Exception as e:errors.append(f'Row {i}: {e}')
    return {'imported':count,'errors':errors[:100]}

def import_vectors_npz(raw):
    """Import only numeric arrays keyed `<existing-face-id>_<model>`.
    NumPy pickle execution is disabled. No people or identities are created.
    """
    import io,numpy as np
    from registry import MODEL_REGISTRY
    if len(raw)>30*1024*1024:raise ValueError('Maximum NPZ size is 30 MB')
    with connect() as c: valid={int(r[0]) for r in c.execute('SELECT id FROM faces')}
    imported=0;errors=[]
    with np.load(io.BytesIO(raw),allow_pickle=False) as bundle:
        if len(bundle.files)>50000:raise ValueError('NPZ exceeds 50,000 vectors')
        for key in bundle.files:
            try:
                face_id,model=key.split('_',1)
                fid=int(face_id)
                if fid not in valid:raise ValueError('Face ID does not exist')
                if model not in MODEL_REGISTRY:raise ValueError('Unknown model')
                vec=bundle[key]
                if vec.dtype.kind not in 'fiu':raise ValueError('Embedding must be numeric')
                if vec.ndim!=1 or not 8<=vec.size<=4096:raise ValueError('Embedding length or rank invalid')
                save_embedding(fid,model,np.asarray(vec,dtype=np.float32))
                imported+=1
            except Exception as exc:errors.append(f'{key}: {exc}')
    return {'imported':imported,'errors':errors[:100]}
