"""Bound GPU reduction rounding without changing any training computation."""
import torch


def tensor_difference(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Checkpoint tensor shape/dtype changed')
    if torch.equal(a,b):return None
    if a.dtype!=torch.float32:raise ValueError('Non-float32 checkpoint tensor changed')
    mask=a!=b;x=a[mask];y=b[mask]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Non-finite checkpoint weights')
    lo=torch.minimum(x,y);hi=torch.maximum(x,y)
    adjacent=torch.nextafter(lo,torch.full_like(lo,float('inf')))
    if (hi>adjacent).any():raise ValueError('Checkpoint difference exceeds one float32 rounding step')
    return dict(changed_elements=int(mask.sum()),elements=a.numel(),max_absolute_difference=float((x-y).abs().max()),maximum_float32_steps=1)


def compare_safetensors(left,right):
    from safetensors import safe_open
    changed=[];total=0
    with safe_open(left,framework='pt',device='cpu') as x,safe_open(right,framework='pt',device='cpu') as y:
        if x.keys()!=y.keys():raise ValueError('Checkpoint tensor names changed')
        for key in x.keys():
            a,b=x.get_tensor(key),y.get_tensor(key);total+=a.numel()
            difference=tensor_difference(a,b)
            if difference:changed.append(dict(name=key,**difference))
    return dict(bitwise_identical=not changed,total_elements=total,changed_tensors=changed,
        max_absolute_difference=max((r['max_absolute_difference'] for r in changed),default=0.))
