# Sentiment Analysis API - Backend

Flask REST API for sentiment analysis using machine learning.

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Server runs on `http://localhost:5000`

## 📡 API Endpoints

### `GET /`
API information

### `GET /health`
Health check

### `POST /predict`
Analyze sentiment

**Request:**
```json
{
  "text": "Your text here"
}
```

**Response:**
```json
{
  "success": true,
  "sentiment": "positive",
  "confidence": 85.2,
  "scores": {
    "positive": 85.2,
    "negative": 14.8
  }
}
```

## 🛠️ Tech Stack

- Flask 3.0.0
- Scikit-learn 1.8.0
- NLTK 3.8.1
- Flask-CORS 4.0.0

## 📦 Files

- `app.py` - Main Flask application
- `models/` - Trained ML models
- `requirements.txt` - Dependencies

## 🔧 Environment Variables

None required for local development.

For production, ensure models are in `models/` directory.

## 📊 Model Info

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF (5,000 features)
- **Accuracy**: 82.64%
- **Training Data**: 150k reviews

## 🚀 Deployment

Deployed on Render with:
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app`