import streamlit as st
import pickle
from preprocess import preprocess

# Load saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("🎬 Sentiment Analysis System")
st.write("Enter a movie review and I'll predict if it's positive or negative.")

review = st.text_area("Your Review:", height=150)

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = preprocess(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0][prediction]
        if prediction == 1:
            st.success(f"POSITIVE 😊 — Confidence: {confidence:.2%}")
        else:
            st.error(f"NEGATIVE 😞 — Confidence: {confidence:.2%}")