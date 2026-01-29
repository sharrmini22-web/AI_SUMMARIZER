import streamlit as st
from transformers import pipeline
from newspaper import Article
import nltk
import time

# Essential Setup
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass

setup_nltk()

# Page Setup
st.set_page_config(page_title="AI Summary Tool", page_icon="📰")

# LOAD ULTRA-LIGHTWEIGHT MODEL (T5-Small)
# This model is tiny (about 240MB) compared to BART (1600MB).
@st.cache_resource
def load_ai():
    return pipeline("summarization", model="t5-small")

summarizer = load_ai()

st.title("📰 Simple News Summarizer")
st.write("A lightweight AI tool to summarize news articles quickly.")

# Input
url = st.text_input("Paste News URL here:")

if st.button("Summarize"):
    if url:
        try:
            with st.spinner('AI is processing...'):
                start_time = time.time()
                
                # Fetch Article
                article = Article(url)
                article.download()
                article.parse()
                
                # Memory Safety: Only take the first 2000 characters
                text_to_summarize = article.text[:2000]
                
                if len(text_to_summarize) < 50:
                    st.error("The article text is too short to summarize.")
                else:
                    # Summarize
                    result = summarizer(text_to_summarize, max_length=100, min_length=30, do_sample=False)
                    summary = result[0]['summary_text']
                    
                    # Output
                    st.subheader(article.title)
                    st.success(summary)
                    
                    # Metrics
                    st.info(f"Done in {round(time.time() - start_time, 2)} seconds.")
                    
        except Exception as e:
            st.error("The app ran into a small problem. Please try a different URL.")
    else:
        st.warning("Please enter a URL first.")
