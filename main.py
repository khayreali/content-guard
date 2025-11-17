from chat_logic import text_chat, image_chat, load_models
import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
from huggingface_hub import login

load_dotenv()

if os.getenv("HUGGING_FACE_TOKEN"):
    login(token=os.getenv("HUGGING_FACE_TOKEN"))

st.title('ContentGuard')

load_models()

prompt = st.chat_input(accept_file=True, file_type=['jpg', 'jpeg'])

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        try:
            st.image(message['content'])
        except:
            st.markdown(message['content'])

if prompt and prompt.text and not prompt['files']:
    text_chat(prompt)

if prompt and prompt['files']:
    modified_prompt = {'image': Image.open(prompt['files'][0]), 'text': prompt.text}
    image_chat(modified_prompt)