import streamlit as st
from utils.general import get_markdown

from .cls_img import run_cls_img
from .cls_vid import run_cls_vid


def run_classification():
    readme_text = st.markdown(get_markdown("transition_classification.md"), unsafe_allow_html=True)
    
    st.sidebar.title("Task")    
    task_mode = st.sidebar.selectbox(
        "Choose the task", ["Introduction","📷 Image", "📽️ Video"]
    )
    if task_mode == "Introduction":
        st.sidebar.success('To continue select any task.')
    elif task_mode == "📷 Image":
        readme_text.empty()
        run_cls_img()
    elif task_mode == "📽️ Video":
        readme_text.empty()
        run_cls_vid()
