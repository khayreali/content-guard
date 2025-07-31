from safety_models import text_classifier
import streamlit as st

prompt = st.chat_input("Test for offensive material")

if prompt:
    st.write(f"Result: {text_classifier(prompt)}")


