from PIL import Image
from utils import load_pretrained, load_pipeline
from transformers import CLIPModel, CLIPProcessor
import torch

class SafetyCategorizer:
    def __init__(self, img = None, txt = None):
        self.img = img
        self.txt = txt
        self.categories = [
        "Violence",
        "Sexually Explicit", 
        "Identity Hate",
        "Drugs",
        "Weapons",
        "Self-Harm"
    ]
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
    def categorize_image_content(self):
        model = load_pretrained(CLIPModel, "openai/clip-vit-base-patch32")
        processor = load_pretrained(CLIPProcessor, "openai/clip-vit-base-patch32")
        image = Image.open(self.img)
        inputs = processor(text = self.categories, images=image, return_tensors='pt', padding=True).to(self.device)

        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        probs = probs.detach().cpu().numpy()[0]

        results = {category: float(probability) for category, probability in zip(self.categories, probs)}
        return max(results, key=results.get)

    def categorize_text_content(self):
        classifier = load_pipeline("zero-shot-classification", "facebook/bart-large-mnli")
        results = classifier(self.txt, self.categories)
        results = {category: float(scores) for category, scores in zip(results['labels'], results['scores'])}
        return max(results, key=results.get)