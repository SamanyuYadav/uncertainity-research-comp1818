"""Meaningful checks: leakage, forward gradients, proper scores, temperature argmax."""
import numpy as np
from run import Net,softmax,metrics,probs,nll,ROOT

def main():
    d=np.load(ROOT/'data.npz'); splits=[set(d[k]) for k in ['train','tune','cal','test']]
    assert len(set.union(*splits))==len(d['y'])
    assert all(not a&b for i,a in enumerate(splits) for b in splits[i+1:])
    p=np.eye(10)[d['y'][:30]]; m=metrics(p,d['y'][:30]); assert m['accuracy']==1 and m['nll']==0 and m['brier']==0
    net=Net(7,.3); x=d['x'][:8]; y=d['y'][:8]; z,h,mask=net.forward(x); dz=softmax(z,axis=1); dz[np.arange(8),y]-=1; dz/=8
    grads=[x.T@((dz@net.w[2].T)*(h>0)),((dz@net.w[2].T)*(h>0)).sum(0),h.T@dz,dz.sum(0)]
    rng=np.random.default_rng(77)
    for w,g in zip(net.w,grads):
        for _ in range(10):
            idx=tuple(rng.integers(n) for n in w.shape); old=w[idx]; eps=1e-5
            w[idx]=old+eps; a=nll(probs(net.logits(x)),y)
            w[idx]=old-eps; b=nll(probs(net.logits(x)),y); w[idx]=old
            assert abs((a-b)/(2*eps)-g[idx])<1e-5
    assert np.array_equal(probs(z[None],.2).argmax(1),probs(z[None],5).argmax(1))
    assert np.allclose(probs(net.logits(x,30,8)).sum(1),1)
    # Check saved model loading reproduces a recorded test prediction.
    c=np.load(ROOT/'results/model_0.npz'); net=Net(0,float(c['dropout'])); net.w=[c[f'arr_{i}'] for i in range(4)]
    saved=np.load(ROOT/'results/predictions/0__clean__MC_Dropout.npy')
    assert np.allclose(probs(net.logits(d['x'][d['test']],30,1000)),saved)
    print('PASS: partition isolation, metric sanity, 40 finite-difference gradients, probability normalization, saved-model replay')
if __name__=='__main__': main()
