import warnings
from transformers import logging, pipeline

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

classifier = pipeline("text-classification", model="patrickquick/BERTicelli")

def text_classifier(txt):
    classification = classifier(txt)
    
    result = classification[0]['label']

    if result == 'NOT':
        output = "Text does not contains offensive material"
    elif result == 'OFF':
        output = "Text contains offensive material"
    
    return output