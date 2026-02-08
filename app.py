# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
from nltk.corpus import stopwords
import nltk
import os

app = Flask(__name__)
CORS(app)

# --- MEMORY OPTIMIZATION 1: Load NLTK data safely ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- MEMORY OPTIMIZATION 2: Initialize Stopwords Once Globally ---
# Creating this set inside the function (as you did before) creates garbage 
# memory on every request. Doing it here saves CPU and RAM.
STOP_WORDS = set(stopwords.words('english'))

# --- LOAD MODELS ---
print("Loading model...")
model_path = os.path.join('models', 'sentiment_model.pkl')
vectorizer_path = os.path.join('models', 'vectorizer.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: Model files not found. Please check 'models/' directory.")
    # Create dummy objects so app doesn't crash immediately (optional)
    model = None
    vectorizer = None

def clean_text(text):
    """Clean text using global stopwords set"""
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    # Use the global constant STOP_WORDS
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    return text

def apply_sentiment_rules(text, prediction, probability):
    # (Keep your existing logic here, it is memory safe)
    text_lower = text.lower()
    
    # Moved lists out if you want further optimization, but they are small enough to stay
    idioms = [
        'hate to say', 'hate to admit', 'hate to break', 'hate to tell',
        'love to see', 'love to know', 'love to hear'
    ]
    
    has_idiom = any(idiom in text_lower for idiom in idioms)
    if has_idiom:
        return prediction, probability
    
    negative_phrases = [
        'i hate this', 'i hated this', 'i hate it', 'i hated it',
        'we hate this', 'totally hate', 'really hate this', 'absolutely hate'
    ]
    
    positive_phrases = [
        'i love this', 'i loved this', 'i love it', 'i loved it',
        'we love this', 'totally love', 'really love this', 'absolutely love'
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

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "status": "running",
        "memory_optimization": "Active"
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not vectorizer:
             return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        user_text = data['text'].strip()
        if not user_text:
            return jsonify({"error": "Empty text"}), 400
        
        cleaned = clean_text(user_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0].copy()
        
        prediction, probability = apply_sentiment_rules(user_text, prediction, probability)
        
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(probability[prediction] * 100)
        
        return jsonify({
            "success": True,
            "sentiment": sentiment,
            "confidence": round(confidence, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # --- MEMORY OPTIMIZATION 3: Disable Debug Mode ---
    # This ensures the model is loaded only once, not twice.
    app.run(debug=False, host='0.0.0.0', port=5000)