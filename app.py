import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("Fake News Detector")
st.write("Enter A News Article below to check whether it is Fake or Real : ")

news_input = st.text_area("News Article : ","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        probabilities = model.predict_proba(transform_input)

        if prediction[0] == 1:
            st.success(f"The News Is Real! (Confidence: {probabilities[0][1]:.2f})")
        else:
            st.error(f"The News Is Fake! (Confidence: {probabilities[0][0]:.2f})")
    else:
        st.warning("Plz Enter Some Text To Analyze")