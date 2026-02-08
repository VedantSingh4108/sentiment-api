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
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load model and vectorizer
print("Loading model...")
model_path = os.path.join('models', 'sentiment_model.pkl')
vectorizer_path = os.path.join('models', 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)
print("Model loaded successfully!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Analyze sentiment",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
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
        probability = model.predict_proba(vectorized)[0]
        
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)