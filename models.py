from functools import lru_cache
import warnings
from PIL import Image
from transformers import logging, pipeline, AutoProcessor, LlavaForConditionalGeneration
import easyocr
import torch

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

@lru_cache(maxsize=4)
def load_model_classification(model_type, model_name):
    model = pipeline(model_type, model = model_name)
    return model

def load_model(model_type, model_name):
    model = model_type.from_pretrained(model_name, torch_dtype=torch.float16)
    return model


@lru_cache(maxsize=1)
def load_processor(processor, model):
    processor = processor.from_pretrained(model)
    return processor

def text_classifier(txt):
    model = load_model_classification("text-classification", "patrickquick/BERTicelli")
    classification = model(txt)
    result = classification[0]['label']

    if result == 'NOT':
        output = "Text does not contains offensive material"
    elif result == 'OFF':
        output = "Text contains offensive material"
    
    return output

def nsfw_classifier(img):
    img = Image.open(img)
    classifier = load_model_classification("image-classification", "Falconsai/nsfw_image_detection")
    result = classifier(img)[0]['label']

    if result == 'normal':
        output = "Image does not contain NSFW material."
    elif result == 'nsfw':
        output = 'Image contains NSFW material.'

    return output

# def nsfw_score(img):
#     img = Image.open(img)
    
#     classifier = load_model("image-classification", "Falconsai/nsfw_image_detection")

#     result = classifier(img)[0]['score']

#     return result

def optical_character_recognition(img):
    img = Image.open(img)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(img)
    return results[0][1] if results else None

def chain_of_thought_img(img, results):
    img = Image.open(img)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = load_model(LlavaForConditionalGeneration,"llava-hf/llava-1.5-7b-hf").to(device)
    processor = load_processor(AutoProcessor, "llava-hf/llava-1.5-7b-hf")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", 
                 "text": f"You're an Trust and Safety associate at a social media company, explain why this photo aligns with your T&S software {results}"},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=200)
    return processor.batch_decode(generate_ids, skip_special_tokens=True)

def chain_of_thought_txt(txt, results):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = load_model(LlavaForConditionalGeneration,"llava-hf/llava-1.5-7b-hf").to(device)
    processor = load_processor(AutoProcessor, "llava-hf/llava-1.5-7b-hf")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": txt},
                {"type": "text", 
                 "text": f"You're an Trust and Safety associate at a social media company, explain why the results of the safety software aligns the actual text {results}"},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=200)
    return processor.batch_decode(generate_ids, skip_special_tokens=True)