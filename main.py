from chat_logic import text_chat, image_chat
import streamlit as st

# logging.set_verbosity_error()
# warnings.filterwarnings("ignore", category=FutureWarning)

st.title('ContentGuard: Multi-Modal Content Safety Classifier', width = 'content')

prompt = st.chat_input(accept_file=True,
                    file_type=['jpg', 'jpeg'])

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
    image_chat(prompt)