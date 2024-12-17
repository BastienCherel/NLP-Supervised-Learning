import streamlit as st

def run():
    # Display a clone guide
    st.title("Clone and Run the Project Locally")

    st.write("""
        Follow the steps below to clone this project and run it on your local machine:

        ### Steps to Clone and Run:
        
        1. **Clone the repository**: Open your terminal (or Git Bash) and run the following command:
        ```bash
        git clone https://github.com/BastienCherel/NLP-Supervised-Learning
        ```

        2. **Navigate to the project folder**:
        After cloning, change into the project directory:
        ```bash
        cd NLP-Supervised-Learning
        ```

        3. **Install dependencies**: 
        Ensure that you have the necessary dependencies installed. You can use `pip` to install the requirements:
        ```bash
        pip install -r requirements.txt
        ```

        4. **Run the Streamlit app**:
        To launch the app, run the following command in the terminal:
        ```bash
        streamlit run streamlit_app/app.py
        ```
        Make sure `app.py` is the main file of your Streamlit application (change this if your file has a different name).

        ### Additional Notes:
        - Make sure you have **Python** and **Streamlit** installed on your machine. You can install Streamlit using `pip install streamlit`.
        - If you don't have **Git** installed, you can download it from [here](https://git-scm.com/downloads).

        Enjoy experimenting with the project! If you have any issues, feel free to open an issue on GitHub or contact us.
    """)