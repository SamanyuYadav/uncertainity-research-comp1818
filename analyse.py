from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,OUT,metrics
f=pd.read_csv(OUT/'metrics.csv'); summary=f.groupby(['condition','method']).agg({k:['mean','std'] for k in ['accuracy','nll','brier','ece','error_auroc','aurc']});summary.to_csv(OUT/'summary.csv')
d=np.load(ROOT/'data.npz'); y=d['y'][d['test']]; rng=np.random.default_rng(42); rows=[]
for condition in f.condition.unique():
    a=np.stack([np.load(OUT/'predictions'/f'{s}__{condition}__MC_mixed_TS.npy') for s in range(5)])
    for other in ['MC clean TS','MC Dropout']:
        b=np.stack([np.load(OUT/'predictions'/f"{s}__{condition}__{other.replace(' ','_')}.npy") for s in range(5)])
        dif=-np.log(a[:,:,][...,np.arange(360),y])+np.log(b[...,np.arange(360),y])
        vals=[]
        for _ in range(2000):
            seeds=rng.integers(0,5,5); samples=rng.integers(0,360,360); vals.append(dif[seeds][:,samples].mean())
        rows.append(dict(condition=condition,comparison='mixed minus '+other,mean=dif.mean(),low=np.quantile(vals,.025),high=np.quantile(vals,.975)))
pd.DataFrame(rows).to_csv(OUT/'bootstrap.csv',index=False)
colors=['#334155','#2563eb','#dc2626','#059669']
fig,axes=plt.subplots(1,2,figsize=(9,3.1)); cond=['clean','noise_0.15','noise_0.30','noise_0.45']
for name,c in zip(['Logistic','MC Dropout','MC clean TS','MC mixed TS'],colors):
    g=f[f.method==name].groupby('condition')
    for ax,metric in zip(axes,['accuracy','nll']):
        mean=g[metric].mean().reindex(cond); sd=g[metric].std().reindex(cond).fillna(0)
        ax.plot([0,.15,.3,.45],mean,'o-',label=name,color=c); ax.fill_between([0,.15,.3,.45],mean-sd,mean+sd,color=c,alpha=.12)
for ax in axes: ax.set_xlabel('Gaussian noise standard deviation');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
axes[0].set_ylabel('Accuracy');axes[1].set_ylabel('Negative log-likelihood');axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'noise.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.1))
for ax,condition in zip(axes,['clean','noise_0.45']):
    ax.plot([0,1],[0,1],'--',color='gray')
    for name,c in zip(['MC Dropout','MC clean TS','MC mixed TS'],colors[1:]):
        ps=[np.load(OUT/'predictions'/f"{s}__{condition}__{name.replace(' ','_')}.npy") for s in range(5)]
        conf=np.concatenate([p.max(1) for p in ps]); corr=np.concatenate([p.argmax(1)==y for p in ps]); xx=[];yy=[]
        for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            m=(conf>lo)&(conf<=hi)
            if m.any():xx.append(conf[m].mean());yy.append(corr[m].mean())
        ax.plot(xx,yy,'o-',label=name,color=c)
    ax.set(title=condition,xlabel='Mean confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1)); ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'reliability.png',dpi=180)
print(pd.DataFrame(rows).round(4).to_string(index=False))
