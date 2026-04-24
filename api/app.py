from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
# Download NLTK data
# -----------------------------
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

sia = SentimentIntensityAnalyzer()

# -----------------------------
# Aspect Keywords (EN + AR)
# -----------------------------
ASPECT_KEYWORDS = {
    'food': [
        'food','meal','dish','taste','delicious',
        'اكل','الاكل','طعام','وجبه','طعمه','طعم','لذيذ'
    ],
    'service': [
        'service','staff','waiter','slow','fast',
        'خدمه','الخدمه','خدمة','الموظف','بطيء','سريع'
    ],
    'price': [
        'price','cost','expensive','cheap',
        'سعر','السعر','غالي','رخيص'
    ],
    'ambiance': [
        'ambiance','atmosphere','music',
        'المكان','الجو','مريح','زحمه','هادي'
    ],
    'cleanliness': [
        'clean','dirty',
        'نظيف','وسخ','قذر'
    ],
    'delivery': [
        'delivery','order','late',
        'توصيل','طلب','متأخر'
    ],
    'general': [
        'good','bad','great','amazing',
        'حلو','وحش','جميل','رائع','سيء','ممتاز'
    ]
}

# -----------------------------
# Arabic Sentiment
# -----------------------------
def arabic_sentiment(text):
    positive_words = ['حلو','جميل','ممتاز','رائع','كويس','تمام']
    negative_words = ['وحش','سيء','زفت','مقرف','رديء']

    score = 0

    for w in positive_words:
        if w in text:
            score += 1

    for w in negative_words:
        if w in text:
            score -= 1

    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# -----------------------------
# Main Logic
# -----------------------------
def extract_aspect_sentiments(text):
    text_lower = text.lower()
    aspect_sentiments = {}

    for aspect, keywords in ASPECT_KEYWORDS.items():
        aspect_mentioned = any(k in text_lower for k in keywords)

        if aspect_mentioned:
            sentences = re.split(r'[.!?]+', text)
            aspect_sentences = []

            for s in sentences:
                if any(k in s.lower() for k in keywords):
                    aspect_sentences.append(s)

            combined_text = " ".join(aspect_sentences) if aspect_sentences else text

            # 🔥 Hybrid sentiment
            if re.search(r'[\u0600-\u06FF]', combined_text):
                sentiment = arabic_sentiment(combined_text)
            else:
                score = sia.polarity_scores(combined_text)['compound']

                if score >= 0.05:
                    sentiment = 'Positive'
                elif score <= -0.05:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'

            aspect_sentiments[aspect] = sentiment
        else:
            aspect_sentiments[aspect] = 'Not Mentioned'

    return aspect_sentiments

# -----------------------------
# Overall sentiment
# -----------------------------
def get_overall_sentiment(aspects):
    vals = [v for v in aspects.values() if v != 'Not Mentioned']

    if not vals:
        return 'Neutral'

    if vals.count('Positive') > vals.count('Negative'):
        return 'Positive'
    elif vals.count('Negative') > vals.count('Positive'):
        return 'Negative'
    else:
        return 'Neutral'

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty input"}), 400

    aspects = extract_aspect_sentiments(text)
    overall = get_overall_sentiment(aspects)

    mentioned = [k for k, v in aspects.items() if v != 'Not Mentioned']

    return jsonify({
        "original_text": text,
        "overall_sentiment": overall,
        "aspects": mentioned,
        "aspect_sentiments": {k: aspects[k] for k in mentioned}
    })

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("🚀 Server Running...")
    app.run(debug=True)