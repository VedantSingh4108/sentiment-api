import re
from nltk.corpus import stopwords

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Test
original = "I hate this movie"
cleaned = clean_text(original)

print(f"Original: '{original}'")
print(f"Cleaned: '{cleaned}'")
print()

# Check if 'hate' survived
if 'hate' in cleaned:
    print("✓ 'hate' is still there")
else:
    print("✗ 'hate' was removed!")