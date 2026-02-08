import pickle
import re
from nltk.corpus import stopwords
import numpy as np

# Load model
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Test sentences
test_cases = [
    "I hate this movie",
    "I love this movie",
    "This is terrible",
    "This is great",
    "worst movie ever",
    "best movie ever"
]

print("="*70)
print("TESTING MODEL PREDICTIONS")
print("="*70)

for text in test_cases:
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    sentiment = "POSITIVE ✓" if prediction == 1 else "NEGATIVE ✗"
    pos_score = probability[1] * 100
    neg_score = probability[0] * 100
    
    print(f"\nOriginal: '{text}'")
    print(f"Cleaned:  '{cleaned}'")
    print(f"Prediction: {sentiment}")
    print(f"  Positive: {pos_score:.1f}%  |  Negative: {neg_score:.1f}%")

# Check if "hate" is even in the vocabulary
print("\n" + "="*70)
print("VOCABULARY CHECK")
print("="*70)

feature_names = vectorizer.get_feature_names_out()
sentiment_words = ['hate', 'love', 'terrible', 'great', 'worst', 'best', 'bad', 'good']

for word in sentiment_words:
    if word in feature_names:
        idx = list(feature_names).index(word)
        coef = model.coef_[0][idx]
        print(f"✓ '{word}' is in vocabulary - coefficient: {coef:.3f}")
        if coef > 0:
            print(f"    → Model thinks this is POSITIVE (WRONG if it's a negative word!)")
        else:
            print(f"    → Model thinks this is NEGATIVE")
    else:
        print(f"✗ '{word}' NOT in vocabulary (model never learned it!)")