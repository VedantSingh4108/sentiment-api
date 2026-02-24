# Sentiment Analysis - Streamlit Application

Interactive web interface for real-time sentiment analysis using Linear SVM and TF-IDF.

## 🎯 Model Performance

- **Algorithm**: Linear SVM (Support Vector Machine)
- **Accuracy**: 82.17%
- **Precision**: 82% avg
- **Recall**: 82% avg
- **F1-Score**: 0.82
- **Training Data**: 149,472 reviews (IMDB + Twitter + Amazon)
- **Features**: 3,500 TF-IDF features with bigrams

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Opens at `http://localhost:8501`

## 📦 Tech Stack
```
streamlit==1.31.0
scikit-learn==1.4.2
nltk==3.8.1
```

## 🎨 Features

- ✨ Gradient animated UI with glassmorphism effects
- 📊 Real-time sentiment predictions with confidence scores
- 💡 One-click example loader
- 🔍 Preprocessed text viewer
- 📱 Fully responsive design
- 🎭 Custom CSS styling

## 🌐 Live Demo

**Deployed on Streamlit Cloud**: https://sentiment-api-7e96wp3akqlayqhcmtd8xv.streamlit.app

## 📊 Model Details

### Preprocessing
1. HTML tag removal
2. Lowercasing
3. Special character removal
4. Stopword filtering (NLTK)
5. TF-IDF vectorization (3,500 features, bigrams)

### Architecture
```python
LinearSVC(C=1.0, max_iter=2000, dual='auto')
```

### Rule-Based Enhancements
- Idiom detection: "I hate to say..." → neutral handling
- Strong phrases: "I love this" → confidence boost
- Negation patterns: "I hate this" → negative boost

## 📁 Files
```
backend/
├── app.py                 # Streamlit interface
├── models/
│   ├── sentiment_model.pkl    # Linear SVM (~29KB)
│   └── vectorizer.pkl         # TF-IDF (~88MB)
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🧪 Testing

Try these inputs:
- "This is absolutely brilliant! Highly recommend." → Positive
- "Terrible quality. Complete waste of money." → Negative
- "I hate to admit it but this exceeded expectations." → Positive (idiom)

## 📈 Performance

| Metric | Negative | Positive | Avg |
|--------|----------|----------|-----|
| Precision | 83% | 81% | 82% |
| Recall | 81% | 83% | 82% |
| F1-Score | 0.82 | 0.82 | 0.82 |

## 📝 Model Limitations

- **Not for hate speech detection**: This model analyzes sentiment (opinion quality), not content toxicity or harm. It will not detect or filter hate speech, slurs, or harmful content. For content moderation, use specialized toxicity detection models.
- **Binary classification**: Only positive/negative (no neutral option)
- **English only**: Not trained on multilingual data
- **Domain-specific**: Best on reviews/opinions

## 🔧 Customization

Update model info in `app.py`:
```python
st.markdown("""
* **Accuracy:** 81.90%
* **Training Data:** 150k reviews
""")
```

## 👨‍💻 Author

Built with ❤️ using Python & Streamlit

---

⭐ Star the main repository if you find this useful!