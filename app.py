import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk
import os
import numpy as np

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Download stopwords
@st.cache_resource
def download_nltk_data():
    try:
        stopwords.words('english')
    except:
        nltk.download('stopwords', quiet=True)

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
    """
    Clean text with negation handling
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # CRITICAL: Handle negations BEFORE removing characters
    # Transform "don't like" → "dont_like", "not good" → "not_good"
    negations = r'\b(not|no|never|neither|nobody|nothing|nowhere|dont|doesn\'t|didn\'t|won\'t|wouldn\'t|shouldn\'t|couldn\'t|can\'t|cannot|isn\'t|aren\'t|wasn\'t|weren\'t|doesnt|didnt|wont|wouldnt|shouldnt|couldnt|cant|isnt|arent|wasnt|werent)\s+(\w+)'
    text = re.sub(negations, r'\1_\2', text)
    
    # Now remove special characters (after negation handling)
    text = re.sub(r'[^a-zA-Z\s_]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords but keep negation-modified words
    stop_words = set(stopwords.words('english'))
    
    # Keep sentiment-bearing words
    keep_words = {
        'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 
        'none', 'nor', "don't", "doesn't", "didn't", "won't", "wouldn't", 
        "shouldn't", "couldn't", "can't", "cannot", "isn't", "aren't", "wasn't", "weren't",
        'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'shouldnt', 'couldnt', 
        'cant', 'isnt', 'arent', 'wasnt', 'werent',
        'hate', 'love', 'like', 'dislike', 'good', 'bad', 'best', 'worst',
        'great', 'terrible', 'awful', 'excellent', 'horrible', 'amazing',
        'boring', 'interesting', 'poor', 'wonderful', 'disappointing',
        'very', 'really', 'extremely', 'absolutely', 'totally', 'completely',
        'quite', 'rather', 'pretty', 'fairly', 'highly',
        'all', 'any', 'but', 'however', 'although', 'though', 'yet'
    }
    
    stop_words = stop_words - keep_words
    
    # Filter stopwords
    words = []
    for word in text.split():
        if '_' in word or word not in stop_words:
            words.append(word)
    
    text = ' '.join(words)
    
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

# ==========================================
# CUSTOM CSS - WARM BROWN/AMBER THEME
# ==========================================
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main App Background - Warm Dark Brown */
    .stApp {
        background: linear-gradient(135deg, #1a1614 0%, #2d1f1a 100%);
    }
    /* Header Styling */
    [data-testid="stHeader"] {
    background: #3B2A24 !important;
    border-bottom: 1px solid rgba(217,119,6,0.25);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1f1a 0%, #1a1614 100%);
        border-right: 1px solid rgba(217, 119, 6, 0.2);
    }
    
    [data-testid="stSidebar"] h3 {
        color: #FBBF24;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #D1D5DB;
    }
    
    /* Animated Warm Gradient Title */
    .gradient-text {
        background: linear-gradient(120deg, #F59E0B 0%, #D97706 25%, #FBBF24 50%, #F59E0B 75%, #D97706 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 5px;
        padding-bottom: 10px;
        letter-spacing: -0.03em;
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes shimmer {
        to { background-position: 200% center; }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #D1D5DB;
        font-size: 1.2rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(45, 31, 26, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(217, 119, 6, 0.15);
        padding: 35px;
        margin-top: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(217, 119, 6, 0.3);
        box-shadow: 0 12px 40px 0 rgba(217, 119, 6, 0.15);
    }

    /* Sentiment Results */
    .sentiment-positive {
        color: #10B981;
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        text-shadow: 0px 4px 20px rgba(16, 185, 129, 0.4);
        margin: 20px 0;
        letter-spacing: -0.02em;
    }
    
    .sentiment-negative {
        color: #EF4444;
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        text-shadow: 0px 4px 20px rgba(239, 68, 68, 0.4);
        margin: 20px 0;
        letter-spacing: -0.02em;
    }

    /* Primary Button (Analyze) */
    .stButton>button[kind="primary"] {
        width: 100%;
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 0.85rem 2rem;
        border-radius: 16px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(217, 119, 6, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #D97706 0%, #B45309 100%);
        box-shadow: 0 6px 30px rgba(217, 119, 6, 0.6);
        transform: translateY(-2px);
    }
    
    /* Example Buttons */
    div[data-testid="column"] .stButton>button {
        width: 100%;
        background: rgba(217, 119, 6, 0.1);
        color: #FCD34D;
        border: 1.5px solid rgba(217, 119, 6, 0.25);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }
    
    div[data-testid="column"] .stButton>button:hover {
        background: rgba(217, 119, 6, 0.2);
        border-color: #D97706;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(217, 119, 6, 0.3);
        color: #FBBF24;
    }
    
    /* Text Input Area */
    .stTextArea textarea {
        background: rgba(45, 31, 26, 0.4) !important;
        border: 1.5px solid rgba(217, 119, 6, 0.2) !important;
        border-radius: 14px !important;
        color: #F3F4F6 !important;
        font-size: 1.05rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #D97706 !important;
        box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.15) !important;
        background: rgba(45, 31, 26, 0.6) !important;
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        border-radius: 10px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #FBBF24;
    }
    
    [data-testid="stMetricLabel"] {
        color: #D1D5DB !important;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(217, 119, 6, 0.1);
        border-radius: 10px;
        color: #FCD34D !important;
        font-weight: 600;
        border: 1px solid rgba(217, 119, 6, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(217, 119, 6, 0.15);
        border-color: rgba(217, 119, 6, 0.3);
    }
    
    /* Warning/Info boxes */
    .stAlert {
        background: rgba(217, 119, 6, 0.1);
        border: 1px solid rgba(217, 119, 6, 0.3);
        border-radius: 12px;
        color: #FCD34D;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1614;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #D97706;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #F59E0B;
    }
    
    /* Creator Section */
    .creator-section {
        margin-top: 60px;
        text-align: center;
        padding: 25px;
        border-top: 1px solid rgba(217, 119, 6, 0.15);
    }
    
    .creator-text {
        color: #9CA3AF;
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: 1px;
    }
    
    .creator-name {
        background: linear-gradient(120deg, #F59E0B 0%, #FBBF24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 1.2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 5px;
    }
    
    /* Section Headers */
    h5 {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9426/9426997.png", width=90)
    st.markdown("### 🎯 About This App")
    st.write("This AI-powered tool analyzes text to determine if the underlying sentiment is positive or negative.")
    
    st.warning("⚠️ **Limitation**: This model analyzes sentiment (opinion quality), not hate speech or toxic content.")
    
    st.markdown("---")
    st.markdown("### 🧠 Under the Hood")
    st.markdown("""
    * **Model:** Linear SVC
    * **Features:** TF-IDF (3500 words, 1-2 n-grams)
    * **Accuracy:** 82.17%
    * **Training Data:** 150k reviews (IMDB, Twitter, Amazon)
    """)
    st.markdown("---")
    st.caption("Built with ❤️ using Python & Streamlit")

# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown("<h1 class='gradient-text'>Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Decode the emotion behind your text in real-time ✨</p>", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Failed to load models: {error_msg}")
    st.stop()

# Example state management
if 'text_to_analyze' not in st.session_state:
    st.session_state.text_to_analyze = ""

def set_example(text):
    st.session_state.text_to_analyze = text

# Examples Row
st.markdown("<h5>💡 Need inspiration? Try an example:</h5>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.button("😊 Positive", on_click=set_example, 
             args=("This product exceeded all my expectations. Absolutely brilliant design and superb quality!",), 
             use_container_width=True)
with col2:
    st.button("😔 Negative", on_click=set_example, 
             args=("Honestly the worst experience I've had. Complete waste of time and money. Do not recommend.",), 
             use_container_width=True)
with col3:
    st.button("😐 Mixed", on_click=set_example, 
             args=("The visuals were absolutely stunning, but the plot dragged on and became really boring halfway through.",), 
             use_container_width=True)

# Input Section
st.markdown("<br>", unsafe_allow_html=True)
text_input = st.text_area(
    label="Your Text",
    value=st.session_state.text_to_analyze,
    height=150,
    placeholder="Type a movie review, tweet, product review, or any opinion...",
    label_visibility="collapsed"
)

# Analyze Button
if st.button("🔥 Analyze Sentiment", type="primary"):
    if text_input.strip():
        with st.spinner("🔮 Analyzing your text..."):
            
            # Backend logic
            cleaned = clean_text(text_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            
            # LinearSVC decision function
            decision_score = model.decision_function(vectorized)[0]
            confidence_raw = 1 / (1 + np.exp(-abs(decision_score)))
            
            # Create probability array
            if prediction == 1:
                probability = np.array([1 - confidence_raw, confidence_raw])
            else:
                probability = np.array([confidence_raw, 1 - confidence_raw])
            
            # Apply rules
            prediction, probability = apply_sentiment_rules(text_input, prediction, probability)
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = probability[prediction] * 100
            pos_score = probability[1] * 100
            neg_score = probability[0] * 100
            
            # Celebration effect for high confidence positive
            if sentiment == "Positive" and confidence > 90:
                st.balloons()
            
            # ==========================================
            # RESULTS DASHBOARD
            # ==========================================
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            
            # Main Verdict
            if sentiment == "Positive":
                st.markdown(f"<div class='sentiment-positive'>✨ Sparkling {sentiment}! ✨</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='sentiment-negative'>🌧️ Oh no, {sentiment}...</div>", unsafe_allow_html=True)
            
            st.markdown("<hr style='opacity: 0.15; margin: 25px 0;'>", unsafe_allow_html=True)
            
            # Metrics Row
            m1, m2 = st.columns(2)
            with m1:
                st.metric("AI Confidence", f"{confidence:.1f}%")
            with m2:
                st.metric("Words Processed", f"{len(cleaned.split())}")

            # Visual Progress Bars
            st.markdown("<br><b style='color: #E2E8F0; font-size: 1.05rem;'>📊 Detailed Breakdown:</b>", unsafe_allow_html=True)
            st.progress(pos_score / 100, text=f"😊 Positive: {pos_score:.1f}%")
            st.progress(neg_score / 100, text=f"😔 Negative: {neg_score:.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)

            # Expandable technical details
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔍 View Preprocessed Text (What the AI sees)"):
                st.code(cleaned, language="text")
                st.caption("💡 Punctuation, capitalization, and common stop words are removed. Negations are preserved (e.g., 'don't like' → 'dont_like').")
                
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# ==========================================
# CREATOR FOOTER
# ==========================================
st.markdown("""
<div class="creator-section">
    <p class="creator-text"><b>Decoding Emotions By</b></p>
    <p class="creator-name">Vedant Singh</p>
</div>
""", unsafe_allow_html=True)