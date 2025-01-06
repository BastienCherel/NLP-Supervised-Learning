import streamlit as st
def run():
    st.title("List of models trained and comparison of their performance")

    st.write("""
        ### Roberta base go emotions
        We have used the Roberta base go emotions model to predict the emotion of the user and then we applied different models to predict the rating of the review.
        We decided to use Random Forest as the model to predict the rating of the review, it is the one who performs the best.
        https://huggingface.co/SamLowe/roberta-base-go_emotions
    """)


    image = "media/Roberta.png"
    st.image(image, use_container_width=True)

    st.write("""
        ### TF-IDF
        We have used the TF-IDF model to predict the rating of the review.
        Logistic Regression was used as the model to predict the rating of the review because this model is much perfomrant than the Random Forest model.
    """)

    image = "media/TF-IDF.png"
    st.image(image, use_container_width=True)


    st.write("""
        ### Zeroshot BART
        This model defines probabilities for each label, and the label with the highest probability is the predicted label.
        The different labels are: Very Negative, Negative, Neutral, Positive, Very Positive. Here is how the model performs.
        https://huggingface.co/facebook/bart-large-mnli
    """)

    image = "media/Zeroshot_BART.png"
    st.image(image, use_container_width=True)

    st.write("""
        ### Multilingual mDeBERTa-v3 (sentiment analysis)
        This model is similar to the Zeroshot BART model, but it is used to predict whether the review is positive, neutral or negative.
    """)

    image = "media/sentiment_analysis.png"
    st.image(image, use_container_width=True)