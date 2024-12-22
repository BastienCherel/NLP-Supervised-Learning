import streamlit as st
def run():
    st.title("Sentiment Analysis and Rating Prediction")

    model_options = {
        "DistilBERT (default)": "distilbert-base-uncased",
        "BERT": "bert-base-uncased",
        "RoBERTa": "roberta-base",
        "GPT-2": "gpt2",  # GPT models can also be used for different purposes
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
            # Get the sentiment prediction
            # sentiment_result = sentiment_model(user_input)[0]  # Get first result
            # sentiment_label = sentiment_result['label']
            import pickle

            # Load the container and access models
            with open('models/huggingface_randomforest_model.pkl', 'rb') as f:
                container = pickle.load(f)

            model1 = container["model1"]
            model2 = container["model2"]
            # # Map the sentiment to a rating from 1 to 5 stars
            # rating = sentiment_to_rating(sentiment_label)
            import random 
            pred1 = model1.predict(user_input)

            labels_to_keep = ['disappointment', 'neutral', 'anger', 'joy', 'surprise']
            result_dict = {}
            for item in pred1[0]:
                    if item['label'] in labels_to_keep:
                        result_dict[item['label']] = item['score']

            values = list(result_dict.values())

            # If model2 expects a 2D array, you should reshape the input to match the expected shape.
            # Here, we assume that `model2` expects one sample with multiple features (the scores)
            aggregated_input = [values]  # This gives a 2D array with 1 sample and multiple features

            # Print the reshaped input to verify
            print(aggregated_input)
            # Predict with model2 (assuming it's a classifier or regressor)
            rating = model2.predict(aggregated_input)

            # rating = random.randint(1,5)
            # Display the sentiment and predicted rating
            st.write(f"Sentiment: {random.randint(0,5)}")
            st.write(f"Predicted Rating: {rating} Stars")
            # Add Bootstrap icons
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