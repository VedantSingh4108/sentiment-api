import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk
import os

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="centered"
)

# Download stopwords
@st.cache_resource
def download_nltk_data():
    try:
        stopwords.words('english')
    except:
        nltk.download('stopwords')

download_nltk_data()

# Load models
@st.cache_resource
def load_models():
    model_path = os.path.join('models', 'sentiment_model.pkl')
    vectorizer_path = os.path.join('models', 'vectorizer.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_msg = str(e)

# Clean text function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Rule-based corrections
def apply_sentiment_rules(text, prediction, probability):
    text_lower = text.lower()
    
    idioms = ['hate to say', 'hate to admit', 'hate to break', 'hate to tell']
    has_idiom = any(idiom in text_lower for idiom in idioms)
    
    if has_idiom:
        return prediction, probability
    
    negative_phrases = [
        'i hate this', 'i hated this', 'i hate it', 'i hated it',
        'we hate this', 'totally hate', 'really hate this'
    ]
    
    positive_phrases = [
        'i love this', 'i loved this', 'i love it', 'i loved it',
        'we love this', 'totally love', 'really love this'
    ]
    
    negative_count = sum(1 for phrase in negative_phrases if phrase in text_lower)
    positive_count = sum(1 for phrase in positive_phrases if phrase in text_lower)
    
    if negative_count > 0:
        probability[0] = max(0.85, probability[0])
        probability[1] = 1 - probability[0]
        prediction = 0
    elif positive_count > 0:
        probability[1] = max(0.90, probability[1])
        probability[0] = 1 - probability[1]
        prediction = 1
    
    return prediction, probability

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-size: 2rem;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #f44336;
        font-size: 2rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🎭 Sentiment Analyzer")
st.write("AI-powered sentiment analysis using NLP")
st.caption("Trained on 150,000 reviews (IMDB + Twitter + Amazon) • 80% accuracy")
st.markdown("</div>", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Failed to load models: {error_msg}")
    st.stop()

# Input
text_input = st.text_area(
    "Enter your text:",
    height=150,
    placeholder="Type a movie review, tweet, product review, or any opinion...",
    key="text_input"
)

# Analyze button
if st.button("🔍 Analyze Sentiment", type="primary"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            # Clean and predict
            cleaned = clean_text(text_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0].copy()
            
            # Apply rules
            prediction, probability = apply_sentiment_rules(text_input, prediction, probability)
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = probability[prediction] * 100
            pos_score = probability[1] * 100
            neg_score = probability[0] * 100
            
            # Display results
            st.markdown("---")
            
            # Sentiment badge
            if sentiment == "Positive":
                st.markdown(f"<div class='sentiment-positive'>😊 {sentiment}</div>", 
                          unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='sentiment-negative'>😞 {sentiment}</div>", 
                          unsafe_allow_html=True)
            
            # Confidence
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{confidence:.2f}%")
            with col2:
                st.metric("Cleaned Text Length", f"{len(cleaned.split())} words")
            
            # Scores
            st.markdown("### 📊 Sentiment Breakdown")
            st.progress(pos_score / 100, text=f"😊 Positive: {pos_score:.2f}%")
            st.progress(neg_score / 100, text=f"😞 Negative: {neg_score:.2f}%")
            
            # Show cleaned text
            with st.expander("🔍 View Processed Text"):
                st.code(cleaned, language=None)
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# Examples
st.markdown("---")
st.markdown("### 💡 Try These Examples")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("😊 Positive Example"):
        st.session_state.example = "This movie was absolutely brilliant! The acting was superb."

with example_col2:
    if st.button("😞 Negative Example"):
        st.session_state.example = "Worst product I've ever bought. Complete waste of money."

with example_col3:
    if st.button("🤔 Mixed Example"):
        st.session_state.example = "The visuals were stunning but the plot was boring."

if 'example' in st.session_state:
    st.info(f"**Example loaded:** {st.session_state.example}")
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Built with:</strong> Python • Scikit-learn • NLTK • Streamlit</p>
    <p><strong>Model:</strong> Logistic Regression with TF-IDF (2000 features)</p>
    <p><strong>Dataset:</strong> 150k reviews (IMDB + Twitter + Amazon)</p>
</div>
""", unsafe_allow_html=True)