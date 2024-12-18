import streamlit as st

def run():
    # Display the Motivation and Acknowledgments Section
    st.title("Motivation & Acknowledgments")
    st.write("""
        This project is part of our Master's program, specifically focusing on Natural Language Processing (NLP) techniques. 
        As part of our end-of-year project, we aimed to explore the potential of semantic embeddings in improving review ranking and recommendation systems. 
        Traditional methods like BM25 have long been used for information retrieval tasks, but their limitations in understanding context and user intent motivated us to investigate whether NLP-based models, specifically those leveraging semantic embeddings, could offer better, more accurate results. 
        The project serves as an opportunity to apply the knowledge gained throughout the course and pushes the boundaries of what’s possible in review-based platforms like TripAdvisor.
        
        **Acknowledgments**  
        We would like to express our sincere gratitude to our teachers, Mr. Rodriguez and Mr. [Last Name], for their continuous support, 
        been instrumental in shaping the direction of this project. We also want to thank our peers and the entire academic team for creating a 
        guidance, and invaluable feedback throughout the course and especially during the final project. Their expertise in NLP and machine learning has 
        collaborative learning environment that encouraged innovative thinking and problem-solving.
    """)