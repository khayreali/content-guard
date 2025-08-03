from models import text_classifier, nsfw_classifier, optical_character_recognition, chain_of_thought_txt, chain_of_thought_img
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
    
    classifier_result = f"Result: {text_classifier(prompt.text)}"

    response = chain_of_thought_txt(prompt.text, classifier_result)

    with assistant:
        st.markdown(f'###### {classifier_result}.')
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
                                      'content': prompt['files']})
    
    image_result = nsfw_classifier(prompt['files'][0])
    text_in_image = optical_character_recognition(prompt['files'][0])
    text_result = text_classifier(text_in_image)

    if not text_result:
        response = f"Result: {image_result}"
        with assistant:
            st.markdown(response)
        st.session_state.messages.append({'role': 'assistant',
                                       'content': response})

    else:
        img_response = f"Results: {image_result}"
        txt_response = f"Results: {text_result}"
        with assistant:
            st.markdown(img_response)
            st.markdown(txt_response)
        st.session_state.messages.append({'role': 'assistant',
                                       'content': img_response})
        st.session_state.messages.append({'role': 'assistant',
                                       'content': txt_response})