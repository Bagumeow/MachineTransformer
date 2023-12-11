import nltk
nltk.download('omw-1.4',quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('wordnet2022', quiet=True)
from create_vocab import SRC,TRG 
from model_transfomer import Transformer
from config import opt
import torch
from translate import translate_sentence
import streamlit as st

model = Transformer(len(SRC.vocab)-1, len(TRG.vocab), opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout'])
model.load_state_dict(torch.load('./data/transformer_weight.pth',map_location=opt['device']))
model.eval()

def translate(sent):
    return translate_sentence(sent, model, SRC, TRG, opt['device'], opt['k'], opt['max_strlen'])

def set_background(image):
    # Custom CSS to set the background image
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url({image}) no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Machine Transfomer EntoVi",
        page_icon="🤖",
        layout="wide"
    )
    set_background('background.jpg')
    st.title("Machine Transformer EntoVi")

    # Create a two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.header("Nhập văn bản cần dịch")
        text_input = st.text_area('Text Input', value='', height=200, max_chars=1000,key='input')
        translate_button = st.button("Dịch ngay")

    with col2:
        st.header("Văn bản dịch")
        if translate_button or text_input:
            output_text = translate(text_input)
            st.text_area('Translated Text:', value=output_text, height=200)



if __name__ == '__main__':
    main()
