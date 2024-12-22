import os
import pandas as pd

def import_data():
    try:
        path = os.path.join(os.getcwd() + "/data/cleaned_data.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=["date_publication", "date_exp"], infer_datetime_format=True)
        print("Data imported sucessfully.")
        return df
    except Exception as e:
        print("Cannot import data: {e}")
        return None
    
df = import_data()

import re
def clean_text(text):
    try:    
        text = re.sub(r'\n', '', text)  # Replace newlines with a space
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
        text = text.lower()  # Convert to lowercase
        return text
    except Exception as e:
        print("No translation")
        return "No translation"
df["avis_en"] = df["avis_en"].apply(clean_text)
# Removing rows without translation
df = df[df['avis_en'] != 'No translation']

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
def balance_classes():
    try:
         # Separate features and target variable
        X = train.drop("note", axis=1)  # Features
        y = train["note"]  # Target variable
        
        # Check class distribution before balancing
        print("Class distribution before balancing:", Counter(y))

        # Undersample the majority class
        undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_res, y_res = undersample.fit_resample(X, y)

        # Check class distribution after balancing
        print("Class distribution after balancing:", Counter(y_res))
        
        # Reconstruct the balanced DataFrame
        df_balanced = pd.DataFrame(X_res, columns=X.columns)
        df_balanced["note"] = y_res
        
        print("Classes balanced successfully.")
        return df_balanced
    except Exception as e:
        print("Could not balance classes")
        return None


def split_train_test():
    try:
        train = df[df["type"] == "train"].copy()
        test = df[df["type"] == "test"].copy()
        train.drop("type", axis=1, inplace=True)
        test.drop("type", axis=1, inplace=True)
        print("Data splitted sucessfully.")
        return train, test
    except Exception as e:
        print(f"Cannot split data : {e}")
        return None, None

train , test  = split_train_test()

# Assuming 'df' is your DataFrame and 'rating' is your target column
train = balance_classes()


from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorizing the cleaned text
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to top 5000 features
X = vectorizer.fit_transform(train["avis_en"]).toarray()

# Target variable (ratings)
y = train["note"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from openai import OpenAI




from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return acc, class_report, conf_matrix

# Evaluation
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

from sklearn.metrics import confusion_matrix, accuracy_score

# Evaluate Logistic Regression
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print("Accuracy for Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix for Logistic Regression:")
print(conf_matrix_lr)

# Evaluate Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Accuracy for Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix for Random Forest:")
print(conf_matrix_rf)


import matplotlib.pyplot as plt
import seaborn as sns

# Create a matplotlib figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot confusion matrix for Logistic Regression
sns.heatmap(conf_matrix_lr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Plot confusion matrix for Random Forest
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1])
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# Show the plot
plt.savefig("media/modelPerformance.png")

from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = ["I am not having a great day"]

model_outputs = classifier(sentences)
print(model_outputs)

print(model_outputs[0])