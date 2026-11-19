"""CPU research experiment. Run: OPENBLAS_NUM_THREADS=1 python run.py"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,platform
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn,scipy
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
def noise(x,s,seed):
    return np.clip(x+np.random.default_rng(seed).normal(0,s,x.shape),0,1)
def blur(x):
    return gaussian_filter(x.reshape(-1,8,8),sigma=(0,.7,.7)).reshape(-1,64)
class Net:
    def __init__(self,seed,drop):
        self.rng=np.random.default_rng(seed); self.drop=drop
        self.w=[self.rng.normal(0,np.sqrt(2/64),(64,64)),np.zeros(64),self.rng.normal(0,np.sqrt(2/64),(64,10)),np.zeros(10)]
    def forward(self,x,stochastic=False):
        h=np.maximum(x@self.w[0]+self.w[1],0)
        mask=(self.rng.random(h.shape)>self.drop)/(1-self.drop) if stochastic else np.ones_like(h)
        return (h*mask)@self.w[2]+self.w[3],h,mask
    def train(self,x,y,epochs=120):
        m=[np.zeros_like(w) for w in self.w]; v=[z.copy() for z in m]; step=0
        for epoch in range(epochs):
            order=self.rng.permutation(len(x))
            for ids in np.array_split(order,int(np.ceil(len(x)/64))):
                xb=x[ids]; z,h,mask=self.forward(xb,True); dz=softmax(z,axis=1); dz[np.arange(len(ids)),y[ids]]-=1; dz/=len(ids)
                dh=(dz@self.w[2].T)*mask*(h>0)
                g=[xb.T@dh+1e-4*self.w[0],dh.sum(0),(h*mask).T@dz+1e-4*self.w[2],dz.sum(0)]
                step+=1
                for j in range(4):
                    m[j]=.9*m[j]+.1*g[j]; v[j]=.999*v[j]+.001*g[j]**2
                    self.w[j]-=.001*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
    def logits(self,x,k=1,seed=0):
        self.rng=np.random.default_rng(seed)
        return np.stack([self.forward(x,k>1)[0] for _ in range(k)])
def probs(z,t=1): return softmax(z/t,axis=-1).mean(0)
def nll(p,y): return float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())
def temperature(z,y):
    fit=minimize_scalar(lambda lt:nll(probs(z,np.exp(lt)),y),bounds=(np.log(.2),np.log(5)),method='bounded')
    return float(np.exp(fit.x))
def metrics(p,y):
    pred=p.argmax(1); correct=pred==y; conf=p.max(1); ece=0
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(conf>lo)&(conf<=hi)
        if mask.any(): ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    ent=-(p*np.log(np.clip(p,1e-12,1))).sum(1)
    order=np.argsort(-conf,kind='stable'); risk=np.cumsum(~correct[order])/np.arange(1,len(y)+1)
    return dict(accuracy=float(correct.mean()),nll=nll(p,y),brier=float(((p-np.eye(10)[y])**2).sum(1).mean()),ece=float(ece),error_auroc=float(roc_auc_score(~correct,ent)) if correct.any() and (~correct).any() else np.nan,aurc=float(risk.mean()))
def main():
    start=time.time(); d=load_digits(); x=d.data/16; y=d.target; ids=np.arange(len(y))
    train,rest=train_test_split(ids,test_size=.4,stratify=y,random_state=2026)
    tune,rest=train_test_split(rest,test_size=.75,stratify=y[rest],random_state=2027)
    cal,test=train_test_split(rest,test_size=2/3,stratify=y[rest],random_state=2028)
    np.savez_compressed(ROOT/'data.npz',x=x,y=y,train=train,tune=tune,cal=cal,test=test)
    tuning=[]; candidates={}
    for dr in [.1,.3,.5]:
        net=Net(100,dr); net.train(x[train],y[train]); score=nll(probs(net.logits(x[tune],30,101)),y[tune]); tuning.append(dict(dropout=dr,nll=score)); candidates[dr]=net
    drop=min(tuning,key=lambda a:a['nll'])['dropout']
    rows=[]; temps=[]; predictions={}; tests={'clean':x[test],'noise_0.15':noise(x[test],.15,700),'noise_0.30':noise(x[test],.30,700),'noise_0.45':noise(x[test],.45,700),'blur_0.70':blur(x[test])}
    prior=np.bincount(y[train],minlength=10)/len(train)
    lr=LogisticRegression(C=1,max_iter=2000).fit(x[train],y[train])
    for condition,xt in tests.items():
        for name,p in [('Uniform',np.full((len(test),10),.1)),('Class prior',np.tile(prior,(len(test),1))),('Logistic',lr.predict_proba(xt))]: rows.append(dict(seed=-1,condition=condition,method=name,**metrics(p,y[test])))
    for seed in range(5):
        net=Net(seed,drop); net.train(x[train],y[train]); np.savez(OUT/f'model_{seed}.npz',*net.w,dropout=drop)
        clean=net.logits(x[cal],30,900+seed)
        # Calibration domains use labels of calibration identities only.
        mixed=np.concatenate([clean,net.logits(noise(x[cal],.1,800),30,900+seed),net.logits(noise(x[cal],.2,801),30,900+seed)],axis=1)
        tc=temperature(clean,y[cal]); tm=temperature(mixed,np.tile(y[cal],3))
        zd=net.logits(x[cal]); td=temperature(zd,y[cal]); zdm=np.concatenate([zd,net.logits(noise(x[cal],.1,800)),net.logits(noise(x[cal],.2,801))],axis=1); tdm=temperature(zdm,np.tile(y[cal],3))
        temps.append(dict(seed=seed,clean=tc,mixed=tm,det_clean=td,det_mixed=tdm))
        for condition,xt in tests.items():
            zs=net.logits(xt,30,1000+seed); zd=net.logits(xt)
            methods={'MLP':probs(zd),'MLP clean TS':probs(zd,td),'MLP mixed TS':probs(zd,tdm),'MC Dropout':probs(zs),'MC clean TS':probs(zs,tc),'MC mixed TS':probs(zs,tm)}
            for name,p in methods.items():
                rows.append(dict(seed=seed,condition=condition,method=name,**metrics(p,y[test]))); predictions[f'{seed}|{condition}|{name}']=p
            for k in [5,10]: rows.append(dict(seed=seed,condition=condition,method=f'MC {k} passes',**metrics(probs(zs[:k]),y[test])))
        print('seed',seed,'complete',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False); 
    (OUT/'predictions').mkdir(exist_ok=True)
    for key,value in predictions.items(): np.save(OUT/'predictions'/f"{key.replace('|','__').replace(' ','_')}.npy",value)
    config=dict(dropout=drop,tuning=tuning,temperatures=temps,splits={n:len(a) for n,a in [('train',train),('tune',tune),('cal',cal),('test',test)]},seconds=time.time()-start,python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,epochs=120,hidden=64,learning_rate=.001,batch_size=64,l2=1e-4,mc_passes=30)
    (OUT/'config.json').write_text(json.dumps(config,indent=2)); print(json.dumps(config,indent=2))
if __name__=='__main__': main()
