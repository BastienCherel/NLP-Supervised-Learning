import streamlit as st
from streamlit_option_menu import option_menu
import sys
# 1. as sidebar menu

st.set_page_config(
    page_title="Prediction Application",
    page_icon="⭐️",
)

with st.sidebar:
    selected = option_menu(None, ["Home","Exploration", "Models","Prediction", "Motivation", "Authors", 'Download'], 
        icons=['house', "bar-chart",'cpu',"terminal", "award","person", 'download'], 
        menu_icon="cast", default_index=0, orientation="vertical")


if selected == "Home":
    st.write("# Welcome to the insurance review rating prediction application")

if selected == "Exploration":
    import data_exploration
    data_exploration.run()

if selected == "Models":
    import models
    models.run()

if selected == "Prediction":
    import prediction
    prediction.run()


if selected == "Motivation":
    import motivation
    motivation.run()
    

if selected == "Authors":
    import authors
    authors.run()

if selected == "Download":
    import download
    download.run()