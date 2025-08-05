from PIL import Image
from utils import load_pretrained
from transformers import CLIPModel, CLIPProcessor
import torch

'''
[
        "violent content",
        "sexually explicit content", 
        "disturbing imagery",
        "graphic content"
    ]

'''

class SafetyCategorizer:
    def __init__(self, categories):
        self.nsfw_categories = categories
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
    def categorize_nsfw_content(self, img):
        model = load_pretrained(CLIPModel, "openai/clip-vit-base-patch32")
        processor = load_pretrained(CLIPProcessor, "openai/clip-vit-base-patch32")

        image = Image.open(img)

        inputs = processor(text = self.nsfw_categories, images=image, return_tensors='pt', padding=True).to(self.device)

        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        probs = probs.detach().cpu().numpy()[0]

        results = {category: float(probability) for category, probability in zip(self.nsfw_categories, probs)}
        return max(results, key=results.get)