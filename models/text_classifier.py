from PIL import Image
import easyocr
from utils import load_pipeline


class TextClassifier:
    def __init__(self):
        pass        

    def text_classifier(txt):
        model = load_pipeline("text-classification", "patrickquick/BERTicelli")
        classification = model(txt)
        result = classification[0]['label']

        if result == 'NOT':
            output = "Text does not contains offensive material"
        elif result == 'OFF':
            output = "Text contains offensive material"
        
        return output