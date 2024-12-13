import streamlit as st
from streamlit_option_menu import option_menu

# 1. as sidebar menu

st.set_page_config(
    page_title="Prediction Application",
    page_icon="⭐️",
)

with st.sidebar:
    selected2 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0, orientation="horizontal")

st.write("# Welcome to the insurance review rating prediction application")