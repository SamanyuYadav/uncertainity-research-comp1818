from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';f=pd.read_csv(O/'metrics.csv');d=np.load(R/'data.npz');y=d['y'][d['test']]
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'legend.fontsize':8,'axes.titlesize':10})
colors=['#334155','#2563eb','#dc2626','#059669'];fig,axes=plt.subplots(2,1,figsize=(4.2,3.8))
conditions=['clean','noise_0.15','noise_0.30','noise_0.45']
for name,c in zip(['Logistic','MC Dropout','MC clean TS','MC mixed TS'],colors):
 g=f[f.method==name].groupby('condition')
 for ax,metric in zip(axes,['accuracy','nll']):
  m=g[metric].mean().reindex(conditions);sd=g[metric].std().reindex(conditions).fillna(0);ax.plot([0,.15,.3,.45],m,'o-',label=name,color=c,markersize=4);ax.fill_between([0,.15,.3,.45],m-sd,m+sd,color=c,alpha=.12)
axes[0].set_ylabel('Accuracy');axes[1].set_ylabel('NLL');axes[1].set_xlabel('Gaussian noise standard deviation');axes[0].legend(ncol=2,loc='lower left');
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
fig.tight_layout(pad=.8,h_pad=.8);fig.savefig(O/'noise_large.png',dpi=250);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(4.2,3.8))
for ax,cond,title in zip(axes,['clean','noise_0.45'],['Clean images','Strong noise']):
 ax.plot([0,1],[0,1],'--',color='gray')
 for name,c in zip(['MC Dropout','MC clean TS','MC mixed TS'],colors[1:]):
  ps=[np.load(O/'predictions'/f"{s}__{cond}__{name.replace(' ','_')}.npy") for s in range(5)];conf=np.concatenate([p.max(1) for p in ps]);corr=np.concatenate([p.argmax(1)==y for p in ps]);xx=[];yy=[]
  for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
   mask=(conf>lo)&(conf<=hi)
   if mask.any():xx.append(conf[mask].mean());yy.append(corr[mask].mean())
  ax.plot(xx,yy,'o-',label=name,color=c,markersize=4)
 ax.text(.03,.95,title,transform=ax.transAxes,va='top',fontsize=9);ax.set(xlim=(0,1),ylim=(0,1),ylabel='Accuracy');ax.spines[['top','right']].set_visible(False)
axes[0].legend(loc='lower right');axes[1].set_xlabel('Mean confidence');fig.tight_layout(pad=.8,h_pad=.8);fig.savefig(O/'reliability_large.png',dpi=250)
