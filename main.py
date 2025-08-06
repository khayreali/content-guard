from chat_logic import text_chat
import streamlit as st

# logging.set_verbosity_error()
# warnings.filterwarnings("ignore", category=FutureWarning)

st.title('SafeGuard: Multi-Modal Content Safety Classifier', width = 'content')

prompt = st.chat_input(accept_file=True,
                    file_type=['jpg', 'jpeg', 'png'])

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if prompt and prompt.text and not prompt['files']:
    text_chat(prompt)
    

# if prompt and prompt['files']:
#     image_chat(prompt)