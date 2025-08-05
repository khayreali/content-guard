from models import ChainOfThought, ImageClassifier, OpticalCharacterRecognition, SafetyCategorizer, TextClassifier 
import streamlit as st
from transformers import logging, warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

st.title('SafeGuard: Multi-Modal Content Safety Classifier', width = 'content')

prompt = st.chat_input(accept_file=True,
                       file_type=['jpg', 'jpeg', 'png'])

# Initialize chat history

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt and prompt.text and not prompt['files']:
    user = st.chat_message('user')
    assistant = st.chat_message('assistant')
    user.markdown(prompt.text)
    st.session_state.messages.append({'role': 'user', 
                                      'content': prompt.text})
    
    with assistant:
        classifier_result = f"Result: {text_classifier(prompt.text)}"
        st.markdown(f'###### {classifier_result}.')
        with st.spinner('Thinking...'):
            response = chain_of_thought_txt(prompt.text, classifier_result)
        with st.expander('*See reasoning*'):
            st.markdown(response)
    
    st.session_state.messages.append({'role': 'assistant',
                                       'content': classifier_result})
    

if prompt and prompt['files']:
    user = st.chat_message('user')
    assistant = st.chat_message('assistant')

    
    if prompt['files'] and prompt.text:
        user.image(prompt['files'][0])
        user.markdown(prompt.text)
    else:
        user.image(prompt['files'][0])
    
    st.session_state.messages.append({'role': 'user', 
                                      'content': prompt['files'][0]})
    
    image_result = nsfw_classifier(prompt['files'][0])
    text_in_image = optical_character_recognition(prompt['files'][0])

    if not text_in_image:
        with assistant:
            st.markdown(f'###### {image_result}.')
            with st.spinner('Thinking...'):
                cot_response = chain_of_thought_img(prompt['files'][0], image_result)
            with st.expander('*See reasoning*'):
                st.markdown(cot_response)
        st.session_state.messages.append({'role': 'assistant',
                                       'content': image_result})

    else:
        text_result = text_classifier(text_in_image)
        with assistant:
            st.markdown(f'###### {text_result}.')
            st.markdown(f'###### {image_result}.')
            with st.spinner('Thinking...'):
                cot_response_txt = chain_of_thought_txt(prompt.text, text_result)
                cot_response_img = chain_of_thought_img(prompt['files'][0], image_result)
            with st.expander('*See reasoning for text*'):
                st.markdown(cot_response_txt)
            with st.expander('*See reasoning for image*'):
                st.markdown(cot_response_img)
        st.session_state.messages.append({'role': 'assistant',
                                       'content': text_result})
        st.session_state.messages.append({'role': 'assistant',
                                       'content': image_result})