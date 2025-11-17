from PIL import Image
from utils import get_nsfw_classifier

class ImageClassifier:
    def __init__(self):
        pass

    def nsfw_classifier(self, img):
        classifier = get_nsfw_classifier()
        result = classifier(img)[0]['label']

        if result == 'normal':
            output = "Image does not contain NSFW material"
        elif result == 'nsfw':
            output = 'Image contains NSFW material'

        return output