from safety_models import text_classifier, image_classifier
# import streamlit as st

# prompt = st.chat_input("Test for offensive material")

# if prompt:
#     st.write(f"Result: {text_classifier(prompt)}")


flower_prediction = image_classifier('flowers.jpg')
boxing_prediction = image_classifier('boxing.jpg')

print(boxing_prediction)