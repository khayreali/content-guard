import torch
import warnings
from PIL import Image
from transformers import logging, pipeline

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

# Text Model
classifier = pipeline("text-classification", model="patrickquick/BERTicelli")

def text_classifier(txt):
    classification = classifier(txt)
    
    result = classification[0]['label']

    if result == 'NOT':
        output = "Text does not contains offensive material"
    elif result == 'OFF':
        output = "Text contains offensive material"
    
    return output

def nsfw_classifier(img):
    img = Image.open(img)
    classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

    result = classifier(img)[0]['label']

    if result == 'normal':
        output = "Image does no contain NSFW material."
    elif result == 'nsfw':
        output = 'Image contains NSFW material.'

    return output

def nsfw_score(img):
    img = Image.open(img)
    classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

    result = classifier(img)[0]['score']

    return result