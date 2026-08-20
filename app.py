import pickle
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

st.set_page_config(
    page_title="Spam Detection AI",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

st.title("🛡️ AI Spam Detection")
st.write("Enter an SMS or email message and let the machine-learning model classify it.")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.warning("Model files are missing. Run `python train_model.py` first.")
    st.stop()

model, vectorizer = load_artifacts()

message = st.text_area(
    "Message",
    height=160,
    placeholder="Example: Congratulations! You won a free prize..."
)

if st.button("🔍 Check Message", use_container_width=True):
    if not message.strip():
        st.error("Please enter a message.")
    else:
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100

        if prediction == "spam":
            st.error(f"🚨 SPAM DETECTED — Confidence: {confidence:.1f}%")
        else:
            st.success(f"✅ NOT SPAM — Confidence: {confidence:.1f}%")

st.divider()
st.caption("Built with Python • Scikit-learn • TF-IDF • Logistic Regression • Streamlit")
