from transformers import LlavaForConditionalGeneration, AutoProcessor
from utils import load_pretrained
from PIL import Image
import torch

class ChainOfThought:
    def __init__(self, img=None,txt=None,img_results=None, txt_results=None, category=None):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.img = img
        self.txt = txt
        self.img_results = img_results
        self.txt_results = txt_results
        self.category = category

    def chain_of_thought(self):
        model = load_pretrained(LlavaForConditionalGeneration,"llava-hf/llava-1.5-7b-hf").to(self.device)
        processor = load_pretrained(AutoProcessor, "llava-hf/llava-1.5-7b-hf")
        if self.img and self.txt:
            conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.img},
                    {"type": "text", 
                    "text": f"""
                    Analyze the text content step-by-step:

                    Step 1: State what the classifier results are for {self.img_results}. State what the text results are, {self.txt_results}, for the text {self.txt}, but don't say the text just the classifier results.
                    Step 2: State what the category results are {self.category}. If the value is none, state that there is no category.
                    Step 3: The key concerns you identify with the the image and text are, if any at all...
                    Step 4: Based on your thought process so far, are the classifications for the image and text, {self.img_results} and {self.txt_results}, correct? If so, state it's correct and why. If any or all are incorrect, state which ones are incorrect and skip to step 5 to explain.
                    Step 5: Skip this step if the classifications were found correct in step 4. Skip by stating "Classifications are correct, no further review required." If any or all classifications are incorrect from step 4, state why they are incorrect.
                    Step 6: Based on your thought process so far, is the categorization of the content, {self.category} from Step 2 correct? If so, state it's correct and why. If it's incorrect, state "Classification is incorrect" and skip to step 7 to explain.
                    Step 7: Skip this step if the category was found correct in step 6. Skip by stating "Category is correct, no further review required." If category is incorrect from step 6, state why it is incorrect.

                    Provide reasoning for every step. Please be succinct and professional.
                    """},
                    ],
            },
            ]
        elif self.img:
            conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.img},
                    {"type": "text", 
                    "text": f"""
                    Analyze the text content step-by-step:

                    Step 1: State what the classifier results are {self.img_results}.
                    Step 2: State what the category results are {self.category}. If the value is none, state that there is no category.
                    Step 3: The key concerns you identify with the the image are, if any at all...
                    Step 4: Based on your thought process so far, are the classifications for the image, {self.img_results}, correct? If so, state it's correct and why. If it's incorrect, state "Classification is incorrect" and skip to step 5 to explain.
                    Step 5: Skip this step if the classification was found correct in step 4. Skip by stating "Classification is correct, no further review required." If classification is incorrect from step 4, state why it is incorrect.
                    Step 6: Based on your thought process so far, is the categorization of the content, {self.category} from Step 2 correct? If so, state it's correct and why. If it's incorrect, state "Classification is incorrect" and skip to step 7 to explain.
                    Step 7: Skip this step if the category was found correct in step 6. Skip by stating "Category is correct, no further review required." If category is incorrect from step 6, state why it is incorrect.

                    Provide reasoning for every step. Please be succinct and professional.
                    """},
                    ],
            },
            ]
        else:
            conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.txt},
                    {"type": "text", 
                    "text": f"""
                    Analyze the text content step-by-step:

                    Step 1: State what the classifier results are {self.txt_results}.
                    Step 2: State what the category results are {self.category}. If the value is none, state that there is no category.
                    Step 3: The key concerns you identify with the the text are, if any at all...
                    Step 4: Based on your thought process so far, are the classifications for the text, {self.txt_results}, correct? If so, state it's correct and why. If it's incorrect, state "Classification is incorrect" and skip to step 5 to explain.
                    Step 5: Skip this step if the classification was found correct in step 4. Skip by stating "Classification is correct, no further review required." If classification is incorrect from step 4, state why it is incorrect.
                    Step 6: Based on your thought process so far, is the categorization of the content, {self.category} from Step 2 correct? If so, state it's correct and why. If it's incorrect, state "Classification is incorrect" and skip to step 7 to explain.
                    Step 7: Skip this step if the category was found correct in step 6. Skip by stating "Category is correct, no further review required." If category is incorrect from step 6, state why it is incorrect.

                    Provide reasoning for every step. Please be succinct and professional.
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
        ).to(self.device)

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=200)
        lst_result = processor.batch_decode(generate_ids, skip_special_tokens=True)
        response_txt = lst_result[0]
        response_txt = response_txt.split('ASSISTANT: ')
        response_txt = response_txt[1]
        parts = response_txt.split("Step ")
        response_txt = "Steps " + "\nStep ".join(parts[1:])
        return response_txt