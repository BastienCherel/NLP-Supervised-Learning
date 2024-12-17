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

# client = OpenAI(
#   api_key="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"
# )



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
print("Accuracy for Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, y_pred_lr))

# Evaluate Random Forest
print("Accuracy for Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))