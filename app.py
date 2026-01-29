import streamlit as st
from transformers import pipeline
from newspaper import Article
import nltk
import time

# Essential NLTK setup for text extraction
@st.cache_resource
def initialize_nltk():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass

initialize_nltk()

# Page Setup
st.set_page_config(page_title="AI News Brief", page_icon="📰", layout="centered")

# Model Loading - Using DistilBART for efficiency and stability
@st.cache_resource
def load_nlp_model():
    # Explicit task and model definition to prevent KeyErrors
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_nlp_model()

# Header
st.title("📰 AI News Summarizer")
st.write("Turn long news articles into quick, readable summaries using Transformer AI.")

# User Input
input_url = st.text_input("Paste the News URL here:", placeholder="https://www.bbc.com/news/...")

if st.button("Generate Summary ✨"):
    if input_url:
        try:
            with st.spinner('AI is reading the article...'):
                start_time = time.time()
                
                # Step 1: Extract Article
                article = Article(input_url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.error("Failed to extract text. Some websites block automated readers.")
                else:
                    # Step 2: Summarize (Truncate to 3000 chars to avoid memory issues)
                    summary_result = summarizer(article.text[:3000], max_length=150, min_length=40, do_sample=False)
                    summary_text = summary_result[0]['summary_text']
                    
                    end_time = time.time()
                    
                    # Step 3: Display results
                    st.divider()
                    st.subheader(f"Title: {article.title}")
                    
                    if article.top_image:
                        st.image(article.top_image, use_container_width=True)
                    
                    st.markdown(f"### AI Summary")
                    st.success(summary_text)
                    
                    # Metrics
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Words", len(article.text.split()))
                    col2.metric("Summary Words", len(summary_text.split()))
                    col3.metric("Time Taken", f"{round(end_time - start_time, 2)}s")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid URL first.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This app uses a Distilled BART model for high-speed abstractive summarization.")
