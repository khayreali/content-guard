from PIL import Image
import easyocr

def optical_character_recognition(img):
    img = Image.open(img)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(img)
    return results[0][1] if results else None