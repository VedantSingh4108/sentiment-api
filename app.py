# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
from nltk.corpus import stopwords
import nltk
import os

# Download stopwords if not present
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Text cleaning function
def clean_text(text):
    """
    Clean text by removing special characters, HTML tags, and stopwords
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Rule-based sentiment correction
def apply_sentiment_rules(text, prediction, probability):
    """
    Rule-based sentiment correction with context awareness
    """
    text_lower = text.lower()
    
    # IGNORE these idiomatic expressions (they don't indicate sentiment)
    idioms = [
        'hate to say', 'hate to admit', 'hate to break', 'hate to tell',
        'love to see', 'love to know', 'love to hear'
    ]
    
    # Check if it's an idiom - if so, skip override
    has_idiom = any(idiom in text_lower for idiom in idioms)
    
    if has_idiom:
        # Don't override - let the ML model decide
        return prediction, probability
    
    # Strong negative phrases (only if NOT an idiom)
    negative_phrases = [
        'i hate this', 'i hated this', 'i hate it', 'i hated it',
        'we hate this', 'totally hate', 'really hate this',
        'absolutely hate'
    ]
    
    # Strong positive phrases
    positive_phrases = [
        'i love this', 'i loved this', 'i love it', 'i loved it',
        'we love this', 'totally love', 'really love this',
        'absolutely love'
    ]
    
    # Count matches
    negative_count = sum(1 for phrase in negative_phrases if phrase in text_lower)
    positive_count = sum(1 for phrase in positive_phrases if phrase in text_lower)
    
    # Apply overrides only for clear cases
    if negative_count > 0:
        probability[0] = max(0.85, probability[0])
        probability[1] = 1 - probability[0]
        prediction = 0
    elif positive_count > 0:
        probability[1] = max(0.90, probability[1])
        probability[0] = 1 - probability[1]
        prediction = 1
    
    return prediction, probability

# Load model and vectorizer
print("Loading model...")
model_path = os.path.join('models', 'sentiment_model.pkl')
vectorizer_path = os.path.join('models', 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)
print("Model loaded successfully!")

# Routes
@app.route('/', methods=['GET'])
def home():
    """Home route - API information"""
    return jsonify({
        "message": "Sentiment Analysis API",
        "status": "running",
        "model": "Logistic Regression with TF-IDF",
        "training_data": "150k reviews (IMDB + Twitter + Amazon)",
        "accuracy": "82.64%",
        "endpoints": {
            "/predict": "POST - Analyze sentiment of text",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment of provided text
    
    Request body:
    {
        "text": "Your text here"
    }
    
    Response:
    {
        "success": true,
        "sentiment": "positive" or "negative",
        "confidence": 85.5,
        "scores": {
            "positive": 85.5,
            "negative": 14.5
        }
    }
    """
    try:
        # Get text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "No text provided",
                "message": "Please send JSON with 'text' field"
            }), 400
        
        user_text = data['text'].strip()
        
        if not user_text:
            return jsonify({
                "error": "Empty text",
                "message": "Please provide non-empty text"
            }), 400
        
        # Clean and predict
        cleaned = clean_text(user_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0].copy()  # Make a copy
        
        # Apply rule-based corrections
        prediction, probability = apply_sentiment_rules(user_text, prediction, probability)
        
        # Prepare response
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(probability[prediction] * 100)
        positive_score = float(probability[1] * 100)
        negative_score = float(probability[0] * 100)
        
        return jsonify({
            "success": True,
            "original_text": user_text,
            "cleaned_text": cleaned,
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "scores": {
                "positive": round(positive_score, 2),
                "negative": round(negative_score, 2)
            }
        })
    
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
