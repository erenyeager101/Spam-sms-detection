import streamlit as st
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load pre-trained models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Missing `vectorizer.pkl` or `model.pkl`. Make sure they are in the same directory.")
    st.stop()

# Initialize Stemmer
port_stemmer = PorterStemmer()

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    words = [port_stemmer.stem(word) for word in words]  # Stemming
    return " ".join(words)

# Streamlit UI
st.title('📩 SMS Spam Classifier')

# User input
input_sms = st.text_area("Enter the message:", height=100)

# Prediction
if st.button('🔍 Predict'):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # 1. Preprocess the text
        transformed_text = clean_text(input_sms)

        # 2. Vectorize the text
        vector_input = tfidf.transform([transformed_text])

        # 3. Make prediction
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.error("🚨 **Spam Message!**")
        else:
            st.success("✅ **Not Spam!**")
