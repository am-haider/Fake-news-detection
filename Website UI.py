import streamlit as st
import pickle
from PIL import Image

# --- Load Model and Vectorizer ---
try:
    model = pickle.load(open("fake_news_model.sav", "rb"))
    vectorizer = pickle.load(open("vectorizer.sav", "rb"))
except FileNotFoundError:
    st.error("❌ Model or vectorizer file not found. Make sure .sav files are in the correct folder.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("📰 About")
st.sidebar.info(
    """
    This Fake News Detection app predicts whether a news title is **Real** or **Fake**.
    
    - Built with Python, Streamlit, and Machine Learning
    - Author: **Muhammad Haider**
    """
)

# Optional: Add an image/logo
# image = Image.open("news_logo.png")
# st.sidebar.image(image, use_column_width=True)

# --- Main Title ---
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Enter a news title below and the model will predict if it is Real or Fake.</p>",
    unsafe_allow_html=True
)

# --- Input Area ---
news_title = st.text_area("Enter News Title", height=150)

# --- Predict Button ---
if st.button("Predict"):
    if news_title.strip() == "":
        st.warning("⚠️ Please enter a news title before predicting.")
    else:
        with st.spinner("Predicting..."):
            try:
                # Transform input and predict
                transformed_text = vectorizer.transform([news_title])
                prediction = model.predict(transformed_text)

                # Display result
                if prediction[0] == 1:
                    st.success("✅ This is Real News")
                else:
                    st.error("❌ This is Fake News")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Footer ---
st.write("---")
st.markdown("<p style='text-align: center;'>AI Project by <b>Muhammad Haider</b></p>", unsafe_allow_html=True)