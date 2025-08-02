from safety_models import text_classifier, nsfw_classifier
import streamlit as st

st.title('Multi-Modal Content Safety Classifier')

prompt = st.chat_input(accept_file=True,
                       file_type=['jpg', 'jpeg', 'png'])

assistant = st.chat_message('assistant')
assistant.write("Hi there! Test for offensive material please!")
user = st.chat_message('user')
if prompt and prompt.text:
    user.write(prompt.text)
    assistant.write(f"Result: {text_classifier(prompt.text)}")

if prompt and prompt['files']:
    user.image(prompt['files'][0])
    assistant.write(f"Result: {nsfw_classifier(prompt['files'][0])}")