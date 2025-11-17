from PIL import Image
from utils import get_device, get_clip_model, get_zero_shot_classifier

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
        self.device = get_device()
        
    def categorize_image_content(self):
        model, processor = get_clip_model()
        image = self.img
        inputs = processor(text = self.categories, images=image, return_tensors='pt', padding=True).to(self.device)

        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        probs = probs.detach().cpu().numpy()[0]

        results = {category: float(probability) for category, probability in zip(self.categories, probs)}
        return max(results, key=results.get)

    def categorize_text_content(self):
        classifier = get_zero_shot_classifier()
        results = classifier(self.txt, self.categories)
        results = {category: float(scores) for category, scores in zip(results['labels'], results['scores'])}
        return max(results, key=results.get)