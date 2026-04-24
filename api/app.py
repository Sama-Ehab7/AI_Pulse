from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import json
import os
from pathlib import Path

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define aspects and their keywords (expanded)
ASPECT_KEYWORDS = {
    'food': ['food', 'meal', 'dish', 'taste', 'flavor', 'delicious', 'yummy', 'bland', 'spicy', 'sweet', 'salty', 'fresh', 'ingredient', 'portion', 'menu', 'entree', 'appetizer', 'dessert', 'breakfast', 'lunch', 'dinner', 'cuisine', 'cooked', 'quality', 'flavorful', 'tasteless', 'overcooked', 'undercooked', 'burnt', 'cold', 'warm', 'hot', 'soup', 'salad', 'main course', 'starter', 'side dish'],
    
    'service': ['service', 'waiter', 'waitress', 'staff', 'server', 'attentive', 'friendly', 'rude', 'slow', 'fast', 'helpful', 'professional', 'accommodating', 'knowledgeable', 'ignored', 'neglected', 'courteous', 'polite', 'impolite', 'efficient', 'incompetent', 'smile', 'greeted', 'seated', 'order', 'check', 'bill'],
    
    'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'worth', 'overpriced', 'reasonable', 'budget', 'pricey', 'costly', 'economical', 'inexpensive', 'deal', 'discount', 'money', 'paid', 'charge', 'fee', 'rate', 'pricing'],
    
    'ambiance': ['ambiance', 'atmosphere', 'vibe', 'decor', 'interior', 'lighting', 'music', 'noise', 'crowded', 'cozy', 'romantic', 'loud', 'quiet', 'modern', 'traditional', 'comfortable', 'setting', 'environment', 'ambience', 'decoration', 'furniture', 'seating', 'space', 'room'],
    
    'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary', 'messy', 'spotless', 'filthy', 'germs', 'bathroom', 'restroom', 'table', 'floor', 'kitchen', 'unclean', 'tidy', 'organized', 'disgusting', 'fresh', 'sticky', 'crumbs'],
    
    'delivery': ['delivery', 'takeout', 'takeaway', 'order', 'arrived', 'late', 'early', 'packaging', 'bag', 'box', 'container', 'leaked', 'missing', 'driver', 'courier', 'app', 'uber eats', 'doordash', 'delivered', 'shipping', 'dispatch', 'arrival'],
    
    'experience': ['experience', 'visit', 'time', 'enjoy', 'love', 'hate', 'recommend', 'return', 'again', 'definitely', 'absolutely', 'unforgettable', 'memorable', 'disappointing', 'amazing', 'wonderful', 'terrible', 'horrible', 'fantastic', 'awesome', 'incredible', 'perfect', 'excellent', 'great', 'good', 'bad', 'okay', 'decent', 'satisfied', 'unsatisfied']
}

def extract_aspect_sentiments(text):
    """Extract sentiments for each aspect from the text"""
    text_lower = text.lower()
    aspect_sentiments = {}
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        # Check if any keyword for this aspect appears in the text
        aspect_mentioned = any(keyword in text_lower for keyword in keywords)
        
        if aspect_mentioned:
            # Extract sentences containing aspect keywords
            sentences = re.split(r'[.!?]+', text)
            aspect_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    aspect_sentences.append(sentence)
            
            # Calculate sentiment for this aspect
            if aspect_sentences:
                combined_text = ' '.join(aspect_sentences)
                sentiment_score = sia.polarity_scores(combined_text)['compound']
            else:
                sentiment_score = sia.polarity_scores(text)['compound']
            
            # Classify sentiment
            if sentiment_score >= 0.05:
                sentiment = 'Positive'
            elif sentiment_score <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            aspect_sentiments[aspect] = sentiment
        else:
            aspect_sentiments[aspect] = 'Not Mentioned'
    
    return aspect_sentiments

def get_overall_sentiment(aspect_sentiments):
    """Calculate overall sentiment based on mentioned aspects"""
    sentiments = [s for s in aspect_sentiments.values() if s != 'Not Mentioned']
    
    if not sentiments:
        return 'Neutral'
    
    positive_count = sentiments.count('Positive')
    negative_count = sentiments.count('Negative')
    
    if positive_count > negative_count:
        return 'Positive'
    elif negative_count > positive_count:
        return 'Negative'
    else:
        return 'Neutral'

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """Analyze sentiment for a given review"""
    data = request.json
    review_text = data.get('text', '')
    
    if not review_text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    # Extract aspect sentiments
    aspect_sentiments = extract_aspect_sentiments(review_text)
    
    # Get overall sentiment
    overall = get_overall_sentiment(aspect_sentiments)
    
    # Prepare response with only aspects that were mentioned
    mentioned_aspects = [aspect for aspect, sentiment in aspect_sentiments.items() 
                        if sentiment != 'Not Mentioned']
    
    response = {
        'original_text': review_text,
        'overall_sentiment': overall,
        'aspects': mentioned_aspects,
        'aspect_sentiments': aspect_sentiments,
        'sentiment_scores': {aspect: aspect_sentiments[aspect] for aspect in mentioned_aspects}
    }
    
    return jsonify(response)

@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    print("🚀 Starting Insight ABSA Server...")
    print("📍 Server running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)