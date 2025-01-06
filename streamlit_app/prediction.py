from transformers import pipeline
import streamlit as st
import pickle
import numpy as np

def predict(model_name, user_input):
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    candidate_labels = ["Positive", "Neutral", "Negative"]
    result = classifier(user_input, candidate_labels, multi_label=False)
    scores = result['scores']
    labels = result['labels']
    predicted_label = labels[np.argmax(scores)]
    if model_name == "huggingface_randomforest_model":
        with open('models/huggingface_randomforest_model.pkl', 'rb') as f:
            container = pickle.load(f)

        model1 = container["classifier"]
        model2 = container["random_forest"]

        pred1 = model1.predict(user_input)

        result_dict = {}
        for item in pred1[0]:
            result_dict[item['label']] = item['score']

        values = list(result_dict.values())

        aggregated_input = [values]  
        
        rating = model2.predict(aggregated_input)
        return predicted_label,rating
    elif model_name == "tfidf_logistic_regression_model":
        container = None
        with open('models/tfidf_logistic_regression_model.pkl', 'rb') as f:
            container = pickle.load(f)
        model = container["tfidf_logistic_regression_model"]
        vectorizer = container["vectorizer"]

        user_input_vectorized = vectorizer.transform([user_input]).toarray()
        rating = model.predict(user_input_vectorized)
        return predicted_label,rating
    elif model_name == "zeroshot_bart_facebook":
        classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
        sentiment_rating_map = {"Very Negative": 1.0, "Negative": 2.0, "Neutral": 3.0, "Positive": 4.0,  "Very Positive": 5.0}
        result = classifier(user_input, candidate_labels=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"])
        return predicted_label,sentiment_rating_map[result['labels'][0]]
    else:
        print('Model not found')
        return None



def run():
    st.title("Sentiment Analysis and Rating Prediction")

    model_options = {
        "Roberta base go emotions + Random Forest": "huggingface_randomforest_model",
        "TF-IDF + Logistic Regression": "tfidf_logistic_regression_model",
        "Zeroshot BART" :"zeroshot_bart_facebook", 
    }
    selected_model_name = st.selectbox("Select a model for sentiment analysis", model_options.keys())

    # Display instructions
    st.write("""
        Please enter a text (such as a review or feedback). The model will analyze the sentiment 
        and predict a rating based on the text, ranging from 1 to 5 stars. 
        A higher rating indicates a more positive sentiment, while a lower rating indicates a more negative sentiment.
    """)

    # User input for review or feedback
    user_input = st.text_area("Enter your text here:")
    if st.button("Submit"):
        if user_input:

            # Load the selected model
            model_name = model_options[selected_model_name]
            sentiment, rating = predict(model_name, user_input)
            st.write(f"Sentiment: {sentiment}")
            st.markdown(
                """
                <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">
                """, unsafe_allow_html=True)

            # Function to display the rating with full, half, and empty stars
            def display_rating(rating):
                full_stars = int(rating)  # Full stars (integer part)
                empty_stars = 5 - full_stars  # Empty stars to fill up to 5 stars

                # Display stars using Bootstrap icons
                st.markdown(
                    f"""
                    <h1>
                        {"<i class='bi bi-star-fill'></i> " * full_stars}
                        {"<i class='bi bi-star'></i> " * empty_stars}
                    </h1>
                    """, unsafe_allow_html=True)

            # Example of displaying the rating (rating can be a float, e.g., 4.5)
            display_rating(rating)
        else:
            st.write("Please enter some text to analyze.")