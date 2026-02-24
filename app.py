import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk
import os

# Page config - Set to wide for a more dashboard-like feel
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="✨",
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

# ==========================================
# CUSTOM CSS (The "Sexy" UI Magic)
# ==========================================
st.markdown("""
<style>
    /* Gradient Text for Main Title */
    .gradient-text {
        background: linear-gradient(45deg, #FF3CAC 0%, #784BA0 50%, #2B86C5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    
    /* Subtitle Styling */
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Glassmorphism Cards for Results */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    }

    /* Sentiment specific text */
    .sentiment-positive {
        color: #00C853;
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        text-shadow: 0px 4px 15px rgba(0, 200, 83, 0.3);
    }
    .sentiment-negative {
        color: #FF3D00;
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        text-shadow: 0px 4px 15px rgba(255, 61, 0, 0.3);
    }

    /* Primary Button Styling */
    .stButton>button {
        width: 100%;
        background-size: 200% auto;
        background-image: linear-gradient(to right, #667eea 0%, #764ba2 51%, #667eea 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.6rem 1rem;
        border-radius: 12px;
        border: none;
        transition: 0.5s;
        box-shadow: 0 4px 15px 0 rgba(118, 75, 162, 0.4);
    }
    .stButton>button:hover {
        background-position: right center; /* trigger gradient animation */
        color: #fff;
        text-decoration: none;
        transform: translateY(-2px);
    }
    
    /* Secondary Buttons (Examples) */
    div[data-testid="column"] .stButton>button {
        background: rgba(100, 100, 100, 0.1);
        color: inherit;
        box-shadow: none;
        border: 1px solid rgba(150, 150, 150, 0.2);
    }
    div[data-testid="column"] .stButton>button:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.1);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", width=80)
    st.markdown("### About This App")
    st.write("This AI-powered tool analyzes text to determine if the underlying sentiment is positive or negative.")
    
    st.markdown("---")
    st.markdown("### 🧠 Under the Hood")
    st.markdown("""
    * **Model:** Logistic Regression
    * **Features:** TF-IDF (2000 words)
    * **Accuracy:** 82.40%
    * **Training Data:** 150k combined reviews (IMDB, Twitter, Amazon)
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

# Example state management (Cleaner implementation)
if 'text_to_analyze' not in st.session_state:
    st.session_state.text_to_analyze = ""

def set_example(text):
    st.session_state.text_to_analyze = text

# Examples Row (Placed above the text box for better UX)
st.markdown("##### 💡 Need inspiration? Try an example:")
col1, col2, col3 = st.columns(3)
with col1:
    st.button("😊 Positive", on_click=set_example, args=("This product exceeded all my expectations. Absolutely brilliant design and superb quality!",))
with col2:
    st.button("😞 Negative", on_click=set_example, args=("Honestly the worst experience I've had. Complete waste of time and money. Do not recommend.",))
with col3:
    st.button("🤔 Mixed", on_click=set_example, args=("The visuals were absolutely stunning, but the plot dragged on and became really boring halfway through.",))

# Input Section
text_input = st.text_area(
    label="Your Text",
    value=st.session_state.text_to_analyze,
    height=150,
    placeholder="Type a movie review, tweet, product review, or any opinion...",
    label_visibility="collapsed"
)

# Analyze Button
if st.button("🚀 Analyze Sentiment", type="primary"):
    if text_input.strip():
        with st.spinner("Decoding emotions..."):
            
            # Backend logic
            cleaned = clean_text(text_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0].copy()
            prediction, probability = apply_sentiment_rules(text_input, prediction, probability)
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = probability[prediction] * 100
            pos_score = probability[1] * 100
            neg_score = probability[0] * 100
            
            # ==========================================
            # RESULTS DASHBOARD
            # ==========================================
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            
            # Main Verdict
            if sentiment == "Positive":
                st.markdown(f"<div class='sentiment-positive'>Sparkling {sentiment}! ✨</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='sentiment-negative'>Oof, {sentiment}. 🌧️</div>", unsafe_allow_html=True)
            
            st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
            
            # Metrics Row
            m1, m2 = st.columns(2)
            with m1:
                st.metric("AI Confidence Score", f"{confidence:.1f}%")
            with m2:
                st.metric("Words Processed", f"{len(cleaned.split())}")

            # Visual Progress Bars
            st.markdown("<br><b>Detailed Sentiment Breakdown:</b>", unsafe_allow_html=True)
            st.progress(pos_score / 100, text=f"🟢 Positive Energy: {pos_score:.1f}%")
            st.progress(neg_score / 100, text=f"🔴 Negative Energy: {neg_score:.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)

            # Expandable technical details
            with st.expander("🔍 View Pre-processed Text (What the AI sees)"):
                st.code(cleaned, language="text")
                st.caption("Note: Punctuation, capitalization, and common 'stop words' are removed before analysis.")
                
    else:
        st.warning("⚠️ Don't leave me hanging! Type some text to analyze.")