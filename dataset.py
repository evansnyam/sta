import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
import joblib

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# ... (rest of your code)

def preprocess_and_retrain():
    # Ask the user to upload a file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the new dataset
        df_new = pd.read_csv(uploaded_file)

        # Create a new DataFrame for preprocessed data
        df_preprocessed_new = df_new.copy()

        # Apply text preprocessing to the 'tweet_text' column
        df_preprocessed_new['cleaned_text'] = df_preprocessed_new['tweet_text'].apply(preprocess_text)

        # Encode the target variable using the saved label encoder
        df_preprocessed_new['encoded_label'] = label_encoder.transform(df_preprocessed_new['cyberbullying_type'])

        # Split the new data into training and testing sets
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            df_preprocessed_new['cleaned_text'],
            df_preprocessed_new['encoded_label'],
            test_size=0.2,
            random_state=42
        )

        # Load the pre-trained pipeline
        model_pipeline = joblib.load('/content/drive/My Drive/sgb/sgd_classifier_model.joblib')

        # Retrain the model on the new training data
        model_pipeline.fit(X_train_new, y_train_new)

        # Save the updated pipeline to the original file path
        joblib.dump(model_pipeline, '/content/drive/My Drive/sgb/sgd_classifier_model.joblib', protocol=4)

        # Optional: Print or return any relevant information
        st.success("Dataset reprocessed and model retrained successfully.")

# Example usage in Streamlit app
if st.button("Reprocess and Retrain"):
    preprocess_and_retrain()
