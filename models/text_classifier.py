from PIL import Image
from utils import get_text_classifier


class TextClassifier:
    def __init__(self):
        pass        

    def text_classifier(self, txt):
        model = get_text_classifier()
        classification = model(txt)
        result = classification[0]['label']

        if result == 'NOT':
            output = "Text does not contain offensive material"
        elif result == 'OFF':
            output = "Text contains offensive material"
        
        return output