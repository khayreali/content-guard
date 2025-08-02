from functools import lru_cache
import warnings
from PIL import Image
from transformers import logging, pipeline
import easyocr

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

@lru_cache(maxsize=4)
def load_model(model_type, model_name):
    model = pipeline(model_type, model = model_name)
    return model
        


def text_classifier(txt):
    model = load_model("text-classification", "patrickquick/BERTicelli")
    classification = model(txt)
    result = classification[0]['label']

    if result == 'NOT':
        output = "Text does not contains offensive material"
    elif result == 'OFF':
        output = "Text contains offensive material"
    
    return output

def nsfw_classifier(img):
    img = Image.open(img)
    classifier = load_model("image-classification", "Falconsai/nsfw_image_detection")
    result = classifier(img)[0]['label']

    if result == 'normal':
        output = "Image does no contain NSFW material."
    elif result == 'nsfw':
        output = 'Image contains NSFW material.'

    return output

def nsfw_score(img):
    img = Image.open(img)
    
    classifier = load_model("image-classification", "Falconsai/nsfw_image_detection")

    result = classifier(img)[0]['score']

    return result

def optical_character_recognition(img):
    img = Image.open(img)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(img)
    return results[0][1] if results else None