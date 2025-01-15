import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

def preprocess_text(text):
    """Function to clean and preprocess text."""
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# Load the vectorizer and model
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit App
st.title("Email/SMS Spam Classifier")
st.write("This app predicts whether a given message is spam or not.")

# User input
input_sms = st.text_area("Enter the message:")

if st.button("Predict"):
    if input_sms.strip():
        # Preprocess the input
        preprocessed_sms = preprocess_text(input_sms)
        # Vectorize the input
        vector_input = vectorizer.transform([preprocessed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter a message to classify.")
