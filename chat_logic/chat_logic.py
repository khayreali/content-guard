import streamlit as st
from models import TextClassifier, ChainOfThought, SafetyCategorizer
    
def text_chat(prompt):
    
    text_classifier = TextClassifier
    classifier_result = f"Result: {text_classifier.text_classifier(prompt.text)}"
    category = SafetyCategorizer(txt=prompt.text)
    category_result = f"Category: {category.categorize_text_content()}"
    chain_of_thought = ChainOfThought(txt=prompt.text, txt_results=classifier_result,category=category_result)
    
    user = st.chat_message('user')
    assistant = st.chat_message('assistant')
    user.markdown(prompt.text)
    st.session_state.messages.append({'role': 'user', 
                                      'content': prompt.text})
    
    with assistant:
        st.markdown(f'###### {classifier_result}.')
        st.markdown(f'###### {category_result}.')
        with st.spinner('Thinking...'):
            cot_response = chain_of_thought.chain_of_thought()
        with st.expander('*See reasoning*'):
            st.markdown(cot_response)
    
    st.session_state.messages.append({'role': 'assistant',
                                       'content': classifier_result})
    
# def image_chat(prompt):
#     user = st.chat_message('user')
#     assistant = st.chat_message('assistant')

    
#     if prompt['files'] and prompt.text:
#         user.image(prompt['files'][0])
#         user.markdown(prompt.text)
#     else:
#         user.image(prompt['files'][0])
    
#     st.session_state.messages.append({'role': 'user', 
#                                       'content': prompt['files'][0]})
    
#     image_result = nsfw_classifier(prompt['files'][0])
#     text_in_image = optical_character_recognition(prompt['files'][0])

#     if not text_in_image:
#         with assistant:
#             st.markdown(f'###### {image_result}.')
#             with st.spinner('Thinking...'):
#                 cot_response = chain_of_thought_img(prompt['files'][0], image_result)
#             with st.expander('*See reasoning*'):
#                 st.markdown(cot_response)
#         st.session_state.messages.append({'role': 'assistant',
#                                        'content': image_result})

#     else:
#         text_result = text_classifier(text_in_image)
#         with assistant:
#             st.markdown(f'###### {text_result}.')
#             st.markdown(f'###### {image_result}.')
#             with st.spinner('Thinking...'):
#                 cot_response_txt = chain_of_thought_txt(prompt.text, text_result)
#                 cot_response_img = chain_of_thought_img(prompt['files'][0], image_result)
#             with st.expander('*See reasoning for text*'):
#                 st.markdown(cot_response_txt)
#             with st.expander('*See reasoning for image*'):
#                 st.markdown(cot_response_img)
#         st.session_state.messages.append({'role': 'assistant',
#                                        'content': text_result})
#         st.session_state.messages.append({'role': 'assistant',
#                                        'content': image_result})


