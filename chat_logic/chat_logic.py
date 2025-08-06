import streamlit as st
from models import TextClassifier, ChainOfThought, SafetyCategorizer, ImageClassifier, OpticalCharacterRecognition
    
def text_chat(prompt):
    user = st.chat_message('user')
    user.markdown(prompt.text)
    st.session_state.messages.append({'role': 'user', 
                                      'content': prompt.text})

    assistant = st.chat_message('assistant')

    text_classifier = TextClassifier
    classifier_result = f"Result: {text_classifier.text_classifier(prompt.text)}"
    category = SafetyCategorizer(txt=prompt.text)
    category_result = f"Category: {category.categorize_text_content()}"
    chain_of_thought = ChainOfThought(txt=prompt.text, txt_results=classifier_result,category=category_result)
    
    with assistant:
        st.markdown(f'###### {classifier_result}.')
        st.markdown(f'###### {category_result}.')
        with st.spinner('Thinking...'):
            cot_response = chain_of_thought.chain_of_thought()
        with st.expander('*See reasoning*'):
            st.markdown(cot_response)
    
    st.session_state.messages.append({'role': 'assistant',
                                       'content': classifier_result})
    
def image_chat(prompt):
    user = st.chat_message('user')
    user.image(prompt['files'][0])
    assistant = st.chat_message('assistant')

    image_result = ImageClassifier
    image_result = image_result.nsfw_classifier((prompt['files'][0]))
    text_in_image = OpticalCharacterRecognition
    text_in_image = text_in_image.optical_character_recognition(prompt['files'][0])
    image_category = SafetyCategorizer(img=prompt['files'][0])
    image_category = image_category.categorize_image_content()
    
    st.session_state.messages.append({'role': 'user', 
                                      'content': prompt['files'][0]})
    

    if not text_in_image:
        with assistant:
            st.markdown(f'###### {image_result}.')
            with st.spinner('Thinking...'):
                cot_response = ChainOfThought(img=prompt['files'][0], img_results=image_result, category=image_category)
                cot_response = cot_response.chain_of_thought()
            with st.expander('*See reasoning*'):
                st.markdown(cot_response)
        st.session_state.messages.append({'role': 'assistant',
                                       'content': image_result})

    else:
        text_result = TextClassifier
        text_result = f"Result: {text_result.text_classifier(text_in_image)}"
        with assistant:
            st.markdown(f'###### {text_result}. {image_result}')
            with st.spinner('Thinking...'):
                cot_response = ChainOfThought(img=prompt['files'][0], txt=text_in_image, img_results=image_result, txt_results=text_result, category=image_category)
                cot_response = cot_response.chain_of_thought()
            with st.expander('*See reasoning*'):
                st.markdown(cot_response)
        st.session_state.messages.append({'role': 'assistant',
                                       'content': f"{text_result} and {image_result}"})