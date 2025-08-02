from safety_models import text_classifier, nsfw_classifier
import streamlit as st

st.title('Multi-Modal Content Safety Classifier')

prompt = st.chat_input(accept_file=True,
                       file_type=['jpg', 'jpeg', 'png'])

# Initialize chat history

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Dsiplay chat messages

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt and prompt.text and not prompt['files']:
    user = st.chat_message('user')
    assistant = st.chat_message('assistant')
    user.markdown(prompt.text)
    st.session_state.messages.append({'role': 'user', 
                                      'content': prompt.text})
    response = f"Result: {text_classifier(prompt.text)}"

    with assistant:
        st.markdown(response)
    
    st.session_state.messages.append({'role': 'assistant',
                                       'content': response})

if prompt and prompt['files']:
    user = st.chat_message('user')
    assistant = st.chat_message('assistant')
    if prompt['files'] and prompt.text:
        user.image(prompt['files'][0])
        user.markdown(prompt.text)
    else:
        user.image(prompt['files'][0])
    
    st.session_state.messages.append({'role': 'user', 
                                      'content': prompt['files']})
    
    response = f"Result: {nsfw_classifier(prompt['files'][0])}"

    with assistant:
        st.markdown(response)
    
    st.session_state.messages.append({'role': 'assistant',
                                       'content': response})

