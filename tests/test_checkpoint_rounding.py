import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_checkpoint_rounding import tensor_difference


def test_only_adjacent_float32_values_are_allowed():
    original=torch.tensor([1.,-.01,0.],dtype=torch.float32)
    adjacent=torch.nextafter(original,torch.full_like(original,float('inf')))
    result=tensor_difference(original,adjacent)
    assert result['changed_elements']==3 and result['maximum_float32_steps']==1
    assert tensor_difference(original,original) is None
    farther=torch.nextafter(adjacent,torch.full_like(adjacent,float('inf')))
    with pytest.raises(ValueError):tensor_difference(original,farther)


def test_corrupt_or_different_shape_and_dtype_cannot_pass():
    a=torch.tensor([1.],dtype=torch.float32)
    for bad in [torch.tensor([float('nan')]),torch.tensor([1.01]),torch.tensor([1.,2.]),a.double()]:
        with pytest.raises(ValueError):tensor_difference(a,bad)
