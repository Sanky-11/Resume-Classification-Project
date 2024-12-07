# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:50:14 2024

@author: sanket
"""

import streamlit as st
import joblib
import pandas as pd
from PyPDF2 import PdfReader

# Load the model
lr_model= joblib.load(open("lr_model.joblib","rb"))
tfidf= joblib.load("tfidf.joblib")
# Define the classification function

def classify_resume(text):
    # Transform the text into a vector
    vectorized_text = tfidf.transform([text])  # Ensure input is a 2D array
    # Predict using the model
    return lr_model.predict(vectorized_text)[0]

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        # Try decoding with UTF-8, fallback to other encodings
        try:
            text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            text = uploaded_file.read().decode("latin1")
        return text

    elif uploaded_file.name.endswith(".pdf"):
        # Use PyPDF2 to extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()

    else:
        raise ValueError("Unsupported file format!")

# Streamlit UI
st.title("Resume Classification App")
st.subheader("Upload a Resume or Paste Text Below")

# Text Input
user_input = st.text_area("Paste Resume Text Here", height=200)

# File Upload Option
uploaded_file = st.file_uploader("Upload a Resume File (txt or PDF)", type=["txt", "pdf"])

# Process Uploaded File
if uploaded_file:
    try:
        user_input = extract_text_from_file(uploaded_file)
        st.info("File processed successfully!")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Predict Button
if st.button("Classify Resume"):
    if user_input:
        # Predict and Display
        result = classify_resume(user_input)
        st.success(f"The resume is classified as: **{result}**")
    else:
        st.warning("Please provide resume text or upload a file.")