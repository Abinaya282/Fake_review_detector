import streamlit as st
import pickle

st.title("🕵️ Fake Review Detector")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

review = st.text_area("Enter a review:")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Enter text")
    else:
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.error("🚨 Fake Review")
        else:
            st.success("✅ Genuine Review")