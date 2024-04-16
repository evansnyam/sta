from flask import Flask, request, jsonify, render_template
import re
import nltk
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import joblib
import os


nltk.download('stopwords')
nltk.download('wordnet')
app = Flask(__name__)

label_encoder = joblib.load('label_encoder.joblib')
model_pipeline = joblib.load('sgd_classifier_model.joblib')

# Load the SGD classifier, TF-IDF vectorizer, and label encoder
sgd_classifier = joblib.load('sgd_classifier_model.joblib')

# Load the logo image
logo = Image.open('logo.png')



# Function to clean and preprocess text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    
    # Lowercase the text
    text = text.lower()
    
    # Define stopwords
    stop_words = set(stopwords.words('english'))
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize the text and lemmatize each word, excluding stopwords
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    
    # Join the tokens back into a string
    return ' '.join(tokens)








def binary_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction using the loaded pipeline
        prediction = model_pipeline.predict([preprocessed_text])

        # Check for offensive words
        with open('en.txt', 'r') as f:
            offensive_words = [line.strip() for line in f]

        offending_words = [word for word in preprocessed_text.split() if word in offensive_words]

        return prediction[0], offending_words
    except Exception as e:
        return None, None















# Function for multi-class cyberbullying detection
def multi_class_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction
        decision_function_values = sgd_classifier.decision_function([preprocessed_text])[0]

        # Get the predicted class index
        predicted_class_index = np.argmax(decision_function_values)

        # Get the predicted class label using the label encoder
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

        return predicted_class_label, decision_function_values
    except Exception as e:
        return None











def format_offensive_words(offensive_words):
    return ', '.join(offensive_words)

@app.route('/')
def welcome():
    return """<div>
    <h1>Welcome to the Cyberbullying Detection API!</h1>
    <p>This API allows you to detect cyberbullying in text. Send a POST request to /detect with the 'user_input' parameter to analyze text.</p>
    <p>Check the documentation for more details on how to use the API</p>
    </div>"""

@app.route('/detect', methods=['POST'])



@app.route('/detect', methods=['POST'])
def detect():
    user_input = request.form['user_input']
    try:
        # Make binary prediction and check for offensive words
        binary_result, offensive_words = binary_cyberbullying_detection(user_input)

        # Make multi-class prediction
        multi_class_result = multi_class_cyberbullying_detection(user_input)
        predicted_class, prediction_probs = multi_class_result

        result = None

        if binary_result == 1:
            # Check if there are more than four different offensive words
            unique_offensive_words = set(offensive_words)
            if len(unique_offensive_words) > 4:
                result = {
                    "message": "This text is unsafe due to a lot of offensive words.",
                    "details": {
                        "offensive": True,
                        "offensive_reasons": [f"Detected offensive words: {format_offensive_words(offensive_words)}"],
                        "multi_class_result": f"Multi-Class Predicted Class: {predicted_class}",
                    }
                }
            else:
                result = {
                    "message": "This text is unsafe.",
                    "details": {
                        "offensive": True,
                        "offensive_reasons": [f"Detected offensive words: {format_offensive_words(offensive_words)}"],
                        "multi_class_result": f"Multi-Class Predicted Class: {predicted_class}",
                    }
                }
        else:
            # safe
            result = {
                "message": "This text is safe.",
                "details": {
                    "offensive": False,
                    "offensive_reasons": [],
                    "multi_class_result": "",
                    "context_analysis": ""  # No context analysis for safe text
                }
            }
            if len(offensive_words) > 0:
                result["details"]["offensive_reasons"] = [f"Detected offensive words: {format_offensive_words(offensive_words)}"]

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "input": user_input})









if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv("PORT", default=33507), debug=True, request_timeout=3)  # Set request timeout to 30 seconds

