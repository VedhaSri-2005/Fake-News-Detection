import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App UI
st.title("📰 Fake News Detector")
st.write("This app uses a Logistic Regression model to classify news as **Real** or **Fake**.")

# Text input
news_text = st.text_area("📝 Enter news content below:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        # Preprocess input
        import re, string
        def preprocess(text):
            text = text.lower()
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r"\s+", " ", text).strip()
            return text

        cleaned = preprocess(news_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0][prediction]

        if prediction == 1:
            st.success(f"✅ Real News (Confidence: {prob:.2f})")
        else:
            st.error(f"🚨 Fake News (Confidence: {prob:.2f})")



