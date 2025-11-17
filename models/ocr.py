from PIL import Image
import easyocr

class OpticalCharacterRecognition:
        def __init__(self):
                pass
        
        def optical_character_recognition(self, img):
                reader = easyocr.Reader(['en'])
                results = reader.readtext(img)
                return results[0][1] if results else None