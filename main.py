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
        if message['role'] == 'user':
            if message.get('type') == 'image':
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(message['content'], use_container_width=True)
            else:
                st.markdown(message['content'])
        else:
            if message.get('type') == 'analysis':
                st.markdown(f"### {message['content']['category']}")
                st.markdown(f"**Text Analysis:** {message['content']['text_result']}")
                with st.expander('Detailed Reasoning'):
                    st.markdown(message['content']['reasoning'])
            elif message.get('type') == 'image_analysis':
                st.markdown(f"### {message['content']['category']}")
                st.markdown(f"**Image Analysis:** {message['content']['image_result']}")
                with st.expander('Detailed Reasoning'):
                    st.markdown(message['content']['reasoning'])
            elif message.get('type') == 'combined_analysis':
                st.markdown(f"### {message['content']['category']}")
                st.markdown(f"**Text in image:** {message['content']['text_result']}")
                st.markdown(f"**Image content:** {message['content']['image_result']}")
                with st.expander('Detailed Reasoning'):
                    st.markdown(message['content']['reasoning'])

if prompt and prompt.text and not prompt['files']:
    text_chat(prompt)

if prompt and prompt['files']:
    modified_prompt = {'image': Image.open(prompt['files'][0]), 'text': prompt.text}
    image_chat(modified_prompt)