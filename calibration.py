"""Empirical verification metrics. Small samples cannot establish operational FAR."""
import numpy as np
from database import embeddings,record_calibration

def metrics(genuine,impostor,threshold):
    if not genuine or not impostor:raise ValueError('Need both same-identity and distinct-identity samples')
    return {'tar':float(np.mean(np.asarray(genuine)>=threshold)),
            'far':float(np.mean(np.asarray(impostor)>=threshold)),
            'threshold':float(threshold),'genuine_count':len(genuine),'impostor_count':len(impostor)}

def pairs(model):
    rows=embeddings(model);pos=[];neg=[];hard=[]
    if len(rows)>3000:raise ValueError('Calibration limited to 3000 samples per model')
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            s=float(a['vector']@b['vector'])
            if a['person_id']==b['person_id']:pos.append(s)
            else:
                neg.append(s)
                hard.append({'left':a['name'],'right':b['name'],'similarity':round(s,5)})
    hard.sort(key=lambda x:x['similarity'],reverse=True)
    return pos,neg,hard[:30]

def calibrate(model,target_far=.01):
    if not 0<target_far<1:raise ValueError('Target FAR must be between zero and one')
    pos,neg,hard=pairs(model)
    if not pos or not neg:raise ValueError('Calibration requires at least two photos of one person and photos of a different person')
    # Conservative empirical order-statistic threshold; no interpolation to claim unobserved FAR.
    target_n=max(0,int(np.floor(len(neg)*target_far)))
    sorted_neg=sorted(neg,reverse=True)
    threshold=float(min(1.0,np.nextafter(sorted_neg[target_n] if target_n<len(neg) else -1,1)))
    scores=metrics(pos,neg,threshold)
    roc=[]
    for t in np.linspace(-1,1,101):
        values=metrics(pos,neg,float(t));roc.append({'far':values['far'],'tar':values['tar'],'threshold':float(t)})
    result={'model':model,'target_far':target_far,'threshold':threshold,**scores,'roc':roc,'hard_negatives':hard,
            'warning':'Empirical sample statistics only; this threshold is not a validated operating FAR on unseen subjects.'}
    record_calibration(model,threshold,result)
    return result

def train_stacker(model_names):
    from sklearn.linear_model import LogisticRegression
    from database import embeddings
    by_model={m:{r['face_id']:r for r in embeddings(m)} for m in model_names}
    if not by_model:raise ValueError('No models selected')
    ids=set.intersection(*(set(v.keys()) for v in by_model.values()))
    ids=sorted(ids)
    if len(ids)>1500:raise ValueError('Training limited to 1500 shared face samples')
    x=[];y=[]
    for i,aid in enumerate(ids):
        for bid in ids[i+1:]:
            x.append([float(by_model[m][aid]['vector']@by_model[m][bid]['vector']) for m in model_names]);
            y.append(int(by_model[model_names[0]][aid]['person_id']==by_model[model_names[0]][bid]['person_id']))
    if len(set(y))!=2:raise ValueError('Need both same-person and different-person pairs')
    clf=LogisticRegression(max_iter=1000,class_weight='balanced').fit(x,y)
    return {'models':model_names,'coefficients':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0]),
            'pairs':len(y),'warning':'Training-set fitted scores are not calibrated probabilities without held-out validation.'}
