import streamlit as st
from transformers import pipeline

# --------------------------------
# Load model (cached)
# --------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

summarizer = load_model()

# --------------------------------
# UI
# --------------------------------
st.set_page_config(page_title="News Summarizer", layout="centered")
st.title("📰 News Article Summarization using LLM")

st.write(
    "This application summarizes news articles using a pre-trained "
    "Large Language Model (BART)."
)

article_text = st.text_area(
    "Paste the news article text below:",
    height=300
)

if st.button("Summarize"):
    if article_text.strip():
        with st.spinner("Generating summary..."):
            summary = summarizer(
                article_text,
                max_length=130,
                min_length=30,
                do_sample=False
            )
            st.subheader("Generated Summary")
            st.write(summary[0]["summary_text"])
    else:
        st.warning("Please enter some text to summarize.")
