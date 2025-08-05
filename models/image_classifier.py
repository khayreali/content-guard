from PIL import Image
from utils import load_pipeline

class ImageClassifier:
    def __init__(self):
        pass

    def nsfw_classifier(img):
        img = Image.open(img)
        classifier = load_pipeline("image-classification", "Falconsai/nsfw_image_detection")
        result = classifier(img)[0]['label']

        if result == 'normal':
            output = "Image does not contain NSFW material"
        elif result == 'nsfw':
            output = 'Image contains NSFW material'

        return output