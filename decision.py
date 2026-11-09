"""Pure CPU nested-pool rank/histogram merge; no model or GPU dependency."""
import numpy as np

def empty():
    return dict(n=0,histogram=[0]*65,top2=[],pools=[],chunks=[])

def add_counts(state,counts,start,pool_sizes):
    a=np.asarray(counts)
    assert a.ndim==1 and len(a)>0 and a.dtype.kind in 'iu'
    assert int(a.min())>=0 and int(a.max())<=64
    assert start==state['n']
    assert pool_sizes==sorted(set(pool_sizes)) and pool_sizes[0]>0
    end=start+len(a);hist=np.asarray(state['histogram'],dtype=np.int64)
    assert len(hist)==65 and hist.sum()==start
    cursor=start
    for stop in sorted(set([n for n in pool_sizes if start<n<=end]+[end])):
        values=a[cursor-start:stop-start].astype(np.int16)
        ids=np.arange(cursor,stop)
        hist+=np.bincount(values,minlength=65)
        order=np.lexsort((ids,-values))[:2]
        state['top2']=sorted(state['top2']+[[int(values[k]),int(ids[k])] for k in order],key=lambda t:(-t[0],t[1]))[:2]
        if stop in pool_sizes:
            assert len(state['top2'])==2
            assert not any(p['pool_size']==stop for p in state['pools'])
            maximum=state['top2'][0][0];ties=int(hist[maximum]);accepted=maximum>=57 and ties==1
            state['pools'].append(dict(pool_size=stop,max_bits=maximum,second_bits=state['top2'][1][0],
                top1_user_id=state['top2'][0][1],tie_count=ties,false_attribution=int(accepted),reject=int(not accepted),histogram=hist.tolist()))
        cursor=stop
    assert hist.sum()==end
    state.update(n=end,histogram=hist.tolist())
    return state
