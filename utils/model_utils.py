from functools import lru_cache
from transformers import pipeline
import torch

import sys
sys.path.append('../')

@lru_cache(maxsize=4)
def load_pipeline(task, model_name):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return pipeline(task, model=model_name, device=device)

@lru_cache(maxsize=4)
def load_pretrained(model_class, model_name, **kwargs):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    if device != 'cpu' and (model_class, 'forward') and 'torch_dtype' not in kwargs:
        kwargs['torch_dtype'] = torch.float16
    
    obj = model_class.from_pretrained(model_name, **kwargs)

    if hasattr(obj, 'to'):
        obj = obj.to(device)
    
    return obj