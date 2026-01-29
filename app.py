import streamlit as st
from transformers import pipeline
from newspaper import Article
import time
import nltk

# 1. Setup NLTK (Mandatory for newspaper3k to work)
@st.cache_resource
def download_nltk():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass

download_nltk()

# 2. Load a STABLE Model (DistilBART)
# We use this because 'bart-large-cnn' is too big for Streamlit's RAM
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# 3. UI Interface
st.set_page_config(page_title="AI Article Summarizer")
st.title("📝 New AI Article Summarizer")
st.write("This version is optimized for speed and stability.")

url = st.text_input("Enter News URL:")

if st.button("Summarize"):
    if url:
        try:
            with st.spinner('Summarizing...'):
                # Scrape Article
                article = Article(url)
                article.download()
                article.parse()
                
                # AI Summary
                # We limit the text to 3000 chars to save memory
                input_text = article.text[:3000]
                summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
                
                # Display results
                st.subheader(article.title)
                st.success(summary[0]['summary_text'])
                
                # Show word counts
                st.write(f"**Original Words:** {len(article.text.split())}")
                st.write(f"**Summary Words:** {len(summary[0]['summary_text'].split())}")
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a URL.")
