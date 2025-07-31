import torch
import warnings
from PIL import Image
from transformers import logging, pipeline, ViTForImageClassification, ViTFeatureExtractor


logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

classifier = pipeline("text-classification", model="patrickquick/BERTicelli")
model = ViTForImageClassification.from_pretrained('jaranohaal/vit-base-violence-detection')
feature_extractor = ViTFeatureExtractor.from_pretrained('jaranohaal/vit-base-violence-detection')

def text_classifier(txt):
    classification = classifier(txt)
    
    result = classification[0]['label']

    if result == 'NOT':
        output = "Text does not contains offensive material"
    elif result == 'OFF':
        output = "Text contains offensive material"
    
    return output

def image_classifier(img):

    # Load an image
    image = Image.open(img)

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    prediction = ("Predicted class:", model.config.id2label[predicted_class_idx])

    return prediction

# pip install accelerate

from transformers import AutoProcessor, ShieldGemma2ForImageClassification
from PIL import Image
import torch

# --- 1. Define the policy and create the prompt ---
# You must choose which policy you want to check for.
policy = "No Explicit Content" 
prompt = f"Does the image violate the following policy? {policy}"

# --- Setup remains the same ---
model_id = "google/shieldgemma-2-4b-it"
image = Image.open('flowers.jpg') # Make sure this image should violate the policy above

model = ShieldGemma2ForImageClassification.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

# --- 2. Pass both the prompt and the image to the processor ---
model_inputs = processor(
    text=[prompt], # Pass the prompt as text
    images=[image], 
    return_tensors="pt"
)

# --- Inference and output remain the same ---
with torch.inference_mode():
    scores = model(**model_inputs)

print(f"Policy: {policy}")
print(scores.probabilities)