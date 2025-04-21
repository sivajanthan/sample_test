
import os
import json
import pandas as pd
import nltk
import openai
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from collections import Counter

# 📦 Setup
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
nltk.download('punkt')
nltk.download('stopwords')

# 🎭 Allowed Emotion Categories
EMOTION_CATEGORIES = ['surprised', 'happy', 'angry', 'fearful', 'disgust', 'sad', 'neutral']
SENTIMENT_CATEGORIES = ['positive', 'negative', 'neutral']

# 😃 Emoji emotion/sentiment map (fallback use)
EMOJI_SENTIMENTS = {
    '😊': ('positive', 'happy'), '😁': ('positive', 'happy'), '😍': ('positive', 'happy'),
    '👍': ('positive', 'happy'), '😢': ('negative', 'sad'), '😡': ('negative', 'angry'),
    '👎': ('negative', 'disgust'), '😭': ('negative', 'sad'), '😱': ('neutral', 'surprised'),
    '😨': ('negative', 'fearful'), '😐': ('neutral', 'neutral')
}

# 🔍 Analyze a single text
def analyze_sentiment_and_emotion(text: str) -> Dict[str, str]:
    for emoji, (sentiment, emotion) in EMOJI_SENTIMENTS.items():
        if emoji in text:
            return {"sentiment": sentiment, "emotion": emotion}
    
    prompt = f"What is the sentiment and emotion of the following review? Respond with JSON like: {{'sentiment':'positive','emotion':'happy'}}. Review: {text}"
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt}]
    )
    result = json.loads(response['choices'][0]['message']['content'])
    return result

# 📄 Analyze a CSV of reviews
def analyze_feedback_csv(file_path: str, text_column: str = "Review") -> pd.DataFrame:
    df = pd.read_csv(file_path)
    results = df[text_column].astype(str).apply(analyze_sentiment_and_emotion)
    df['Sentiment'] = results.apply(lambda x: x['sentiment'])
    df['Emotion'] = results.apply(lambda x: x['emotion'])
    return df
