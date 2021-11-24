import streamlit as st
from utils.general import get_markdown


def run_cls_img():
    readme_text = st.markdown(get_markdown("empty.md"), unsafe_allow_html=True)
