import streamlit as st
from models import TextClassifier, ChainOfThought, SafetyCategorizer, ImageClassifier, OpticalCharacterRecognition
from utils import get_llava_model, get_text_classifier, get_nsfw_classifier, get_clip_model, get_zero_shot_classifier
import time
    
def load_models():
    if 'text_classifier' not in st.session_state or \
       'image_classifier' not in st.session_state or \
       'ocr_model' not in st.session_state:
        
        if 'models_loaded' not in st.session_state:
            with st.spinner('Loading models...'):
                
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
                
                status_text.text('Loading LLaVA model...')
                get_llava_model()
                progress_bar.progress(100)
                
                progress_bar.empty()
                status_text.empty()
                st.session_state.models_loaded = True
        
        st.session_state.text_classifier = TextClassifier()
        st.session_state.image_classifier = ImageClassifier()
        st.session_state.ocr_model = OpticalCharacterRecognition()

def post_process_response(response_txt):
    if not response_txt:
        return ""
    try:
        if 'ASSISTANT: ' in response_txt:
            response_txt = response_txt.split('ASSISTANT: ')[1]
        parts = response_txt.split("Step ")
        if len(parts) > 1:
            response_txt = "Steps " + "\nStep ".join(parts[1:])
        return response_txt
    except Exception:
        return response_txt

def text_chat(prompt):
    user = st.chat_message('user')
    user.markdown(prompt.text)
    st.session_state.messages.append({
        'role': 'user', 
        'content': prompt.text,
        'type': 'text'
    })

    assistant = st.chat_message('assistant')
    
    with assistant:
        classifier_result = st.session_state.text_classifier.text_classifier(prompt.text)
        category = SafetyCategorizer(txt=prompt.text)
        category_name = category.categorize_text_content()
        
        st.markdown(f'### {category_name}')
        st.markdown(f'**Text Analysis:** {classifier_result}')
        
        chain_of_thought = ChainOfThought(txt=prompt.text, 
                                         txt_results=f"Result: {classifier_result}",
                                         category=f"Category: {category_name}")

        with st.expander('Detailed Reasoning', expanded=True):
            placeholder = st.empty()
            placeholder.markdown("Generating reasoning...")
            
            full_response = ""
            for new_chunk in chain_of_thought.stream_chain_of_thought():
                for char in new_chunk:
                    full_response += char
                    placeholder.markdown(full_response)
                    time.sleep(0.005)
            
            cot_response = post_process_response(full_response)
            placeholder.markdown(cot_response)
        
        st.session_state.messages.append({
            'role': 'assistant',
            'content': {
                'text_result': classifier_result,
                'category': category_name,
                'reasoning': cot_response
            },
            'type': 'analysis'
        })
    
def image_chat(prompt):
    user = st.chat_message('user')
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        user.image(prompt['image'], use_container_width=True)
    
    st.session_state.messages.append({
        'role': 'user', 
        'content': prompt['image'],
        'type': 'image'
    })
    
    assistant = st.chat_message('assistant')
    
    with assistant:
        image_result = st.session_state.image_classifier.nsfw_classifier(prompt['image'])
        text_in_image = st.session_state.ocr_model.optical_character_recognition(prompt['image'])
        
        safety_categorizer = SafetyCategorizer(img=prompt['image'])
        image_category = safety_categorizer.categorize_image_content()
        
        if not text_in_image:
            st.markdown(f'### {image_category}')
            st.markdown(f'**Image Analysis:** {image_result}')
            
            cot = ChainOfThought(img=prompt['image'], 
                               img_results=image_result, 
                               category=image_category)
            
            with st.expander('Detailed Reasoning', expanded=True):
                placeholder = st.empty()
                placeholder.markdown("Generating reasoning...")
                
                full_response = ""
                for new_chunk in cot.stream_chain_of_thought():
                    for char in new_chunk:
                        full_response += char
                        placeholder.markdown(full_response)
                        time.sleep(0.005)
                
                cot_response = post_process_response(full_response)
                placeholder.markdown(cot_response)
            
            st.session_state.messages.append({
                'role': 'assistant',
                'content': {
                    'image_result': image_result,
                    'category': image_category,
                    'reasoning': cot_response
                },
                'type': 'image_analysis'
            })
        else:
            text_result = st.session_state.text_classifier.text_classifier(text_in_image)
            
            st.markdown(f'### {image_category}')
            st.markdown(f'**Text in image:** {text_result}')
            st.markdown(f'**Image content:** {image_result}')
            
            cot = ChainOfThought(img=prompt['image'], 
                               txt=text_in_image,
                               img_results=image_result, 
                               txt_results=f"Result: {text_result}",
                               category=image_category)
            
            with st.expander('Detailed Reasoning', expanded=True):
                placeholder = st.empty()
                placeholder.markdown("Generating reasoning...")
                
                full_response = ""
                for new_chunk in cot.stream_chain_of_thought():
                    for char in new_chunk:
                        full_response += char
                        placeholder.markdown(full_response)
                        time.sleep(0.005)
                
                cot_response = post_process_response(full_response)
                placeholder.markdown(cot_response)
            
            st.session_state.messages.append({
                'role': 'assistant',
                'content': {
                    'text_result': text_result,
                    'image_result': image_result,
                    'category': image_category,
                    'reasoning': cot_response
                },
                'type': 'combined_analysis'
            })