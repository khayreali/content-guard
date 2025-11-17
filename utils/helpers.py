from functools import lru_cache
from transformers import pipeline, LlavaForConditionalGeneration, AutoProcessor, CLIPModel, CLIPProcessor
import torch

_models = {}

def _get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

_device = _get_device()

@lru_cache(maxsize=16)
def load_pipeline(task, model_name):
    return pipeline(task, model=model_name, device=_device)

@lru_cache(maxsize=16)
def load_pretrained(model_class, model_name, **kwargs):
    if _device != 'cpu' and hasattr(model_class, 'forward') and 'torch_dtype' not in kwargs:
        kwargs['torch_dtype'] = torch.float16
    
    obj = model_class.from_pretrained(model_name, **kwargs)
    
    if hasattr(obj, 'to'):
        obj = obj.to(_device)
    
    return obj

def get_llava_model():
    if 'llava_model' not in _models:
        _models['llava_model'] = load_pretrained(LlavaForConditionalGeneration, "llava-hf/llava-1.5-7b-hf")
        _models['llava_processor'] = load_pretrained(AutoProcessor, "llava-hf/llava-1.5-7b-hf")
    return _models['llava_model'], _models['llava_processor']

def get_text_classifier():
    if 'text_classifier' not in _models:
        _models['text_classifier'] = load_pipeline("text-classification", "patrickquick/BERTicelli")
    return _models['text_classifier']

def get_nsfw_classifier():
    if 'nsfw_classifier' not in _models:
        _models['nsfw_classifier'] = load_pipeline("image-classification", "Falconsai/nsfw_image_detection")
    return _models['nsfw_classifier']

def get_clip_model():
    if 'clip_model' not in _models:
        _models['clip_model'] = load_pretrained(CLIPModel, "openai/clip-vit-base-patch32")
        _models['clip_processor'] = load_pretrained(CLIPProcessor, "openai/clip-vit-base-patch32")
    return _models['clip_model'], _models['clip_processor']

def get_zero_shot_classifier():
    if 'zero_shot_classifier' not in _models:
        _models['zero_shot_classifier'] = load_pipeline("zero-shot-classification", "facebook/bart-large-mnli")
    return _models['zero_shot_classifier']

def get_device():
    return _device