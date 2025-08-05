from functools import lru_cache
import warnings
from transformers import logging, pipeline, AutoProcessor, LlavaForConditionalGeneration, ViTForImageClassification, ViTFeatureExtractor, CLIPProcessor, CLIPModel
from utils import load_pipeline, load_pretrained
import torch

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

def text_classifier(txt):
    model = load_pipeline("text-classification", "patrickquick/BERTicelli")
    classification = model(txt)
    result = classification[0]['label']

    if result == 'NOT':
        output = "Text does not contains offensive material"
    elif result == 'OFF':
        output = "Text contains offensive material"
    
    return output

def nsfw_classifier(img):
    img = Image.open(img)
    classifier = load_pipeline("image-classification", "Falconsai/nsfw_image_detection")
    result = classifier(img)[0]['label']

    if result == 'normal':
        output = "Image does not contain NSFW material"
    elif result == 'nsfw':
        output = 'Image contains NSFW material'

    return output

def chain_of_thought_img(img, results):
    img = Image.open(img)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = load_pretrained(LlavaForConditionalGeneration,"llava-hf/llava-1.5-7b-hf").to(device)
    processor = load_pretrained(AutoProcessor, "llava-hf/llava-1.5-7b-hf")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", 
                 "text": f"""
                Analyze the image content step-by-step:

                 Step 1: State what the image classifier results are... {results}.
                 Step 2: Considering adult content and violence/gore content safety policies, I need to evaluate...
                 Step 3: The key concerns I identify are...
                 Step 4: My final review on whether the results from the image classifier are correct are...

                 Provide reasoning for every step. Please be succint and professional.
                 
                 You work in the domain of Trust and Safety, so please don't be verbose. Just get straight to the point.
                 """},
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
    lst_result = processor.batch_decode(generate_ids, skip_special_tokens=True)
    response_txt = lst_result[0]
    response_txt = response_txt.split('ASSISTANT: ')
    response_txt = response_txt[1]
    parts = response_txt.split("Step ")
    response_txt = "Steps " + "\nStep ".join(parts[1:])
    return response_txt

def chain_of_thought_txt(txt, results):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = load_pretrained(LlavaForConditionalGeneration,"llava-hf/llava-1.5-7b-hf").to(device)
    processor = load_pretrained(AutoProcessor, "llava-hf/llava-1.5-7b-hf")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": txt},
                {"type": "text", 
                 "text": f"""
                 Analyze the text content step-by-step:

                 Step 1: State what the text classifier results are. {results}
                 Step 2: Considering content safety policies, I need to evaluate...
                 Step 3: The key concerns I identify are...
                 Step 4: My final review on whether the results from the text classifier are correct are...

                 Provide reasoning for every step. Please be succint and professional. 
                 You work in the domain of Trust and Safety, so please don't be verbose. Just get straight to the point.
                 """},
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

    generate_ids = model.generate(**inputs, max_new_tokens=400)
    lst_result = processor.batch_decode(generate_ids, skip_special_tokens=True)
    response_txt = lst_result[0]
    response_txt = response_txt.split('ASSISTANT: ')
    response_txt = response_txt[1]
    parts = response_txt.split("Step ")
    response_txt = "Steps " + "\nStep ".join(parts[1:])
    return response_txt
