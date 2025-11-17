import streamlit as st
from models import TextClassifier, ChainOfThought, SafetyCategorizer, ImageClassifier, OpticalCharacterRecognition
from utils import get_llava_model, get_text_classifier, get_nsfw_classifier, get_clip_model, get_zero_shot_classifier
    
def load_models():
    if 'models_loaded' not in st.session_state:
        with st.spinner('Loading AI models (this may take a few minutes on first run)...'):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('Loading text classifier...')
            get_text_classifier()
            progress_bar.progress(20)
            
            status_text.text('Loading NSFW classifier...')
            get_nsfw_classifier()
            progress_bar.progress(40)
            
            status_text.text('Loading CLIP model...')
            get_clip_model()
            progress_bar.progress(60)
            
            status_text.text('Loading zero-shot classifier...')
            get_zero_shot_classifier()
            progress_bar.progress(80)
            
            status_text.text('Loading LLaVA model (largest model)...')
            get_llava_model()
            progress_bar.progress(100)
            
            progress_bar.empty()
            status_text.empty()
            st.session_state.models_loaded = True


def text_chat(prompt):
    user = st.chat_message('user')
    user.markdown(prompt.text)
    st.session_state.messages.append({'role': 'user',
                                      'content': prompt.text})

    assistant = st.chat_message('assistant')

    text_classifier = TextClassifier()
    classifier_result = f"Result: {text_classifier.text_classifier(prompt.text)}"
    category = SafetyCategorizer(txt=prompt.text)
    category_result = f"Category: {category.categorize_text_content()}"
    chain_of_thought = ChainOfThought(txt=prompt.text, txt_results=classifier_result,category=category_result)
    
    with assistant:
        st.markdown(f'###### {classifier_result}.')
        st.markdown(f'###### {category_result}.')
        with st.spinner('Thinking...'):
            cot_response = chain_of_thought.chain_of_thought()
        with st.expander('*See reasoning*'):
            st.markdown(cot_response)
    
    st.session_state.messages.append({'role': 'assistant',
                                       'content': classifier_result})
    
def image_chat(prompt):
    user = st.chat_message('user')
    user.image(prompt['image'])
    assistant = st.chat_message('assistant')

    image_result_classifier = ImageClassifier()
    image_result = image_result_classifier.nsfw_classifier((prompt['image']))
    ocr_text_in_image = OpticalCharacterRecognition()
    text_in_image = ocr_text_in_image.optical_character_recognition(prompt['image'])
    safety_categorizer = SafetyCategorizer(img=prompt['image'])
    image_category = safety_categorizer.categorize_image_content()
    
    st.session_state.messages.append({'role': 'user', 
                                      'content': prompt['image']})
    

    if not text_in_image:
        with assistant:
            st.markdown(f'###### {image_result}.')
            with st.spinner('Thinking...'):
                cot_response = ChainOfThought(img=prompt['image'], img_results=image_result, category=image_category)
                cot_response = cot_response.chain_of_thought()
            with st.expander('*See reasoning*'):
                st.markdown(cot_response)
        st.session_state.messages.append({'role': 'assistant',
                                       'content': image_result})

    else:
        text_classifier = TextClassifier()
        text_result = f"Result: {text_classifier.text_classifier(text_in_image)}"
        with assistant:
            st.markdown(f'###### {text_result}. {image_result}')
            with st.spinner('Thinking...'):
                cot_response = ChainOfThought(img=prompt['image'], txt=text_in_image, img_results=image_result, txt_results=text_result, category=image_category)
                cot_response = cot_response.chain_of_thought()
            with st.expander('*See reasoning*'):
                st.markdown(cot_response)
        st.session_state.messages.append({'role': 'assistant',
                                       'content': f"{text_result} and {image_result}"})