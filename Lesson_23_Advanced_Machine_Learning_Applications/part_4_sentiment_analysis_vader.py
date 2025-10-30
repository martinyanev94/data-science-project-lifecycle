import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Sample text data
texts = [
    "I love this product! It works great.",
    "This is the worst service I have ever experienced.",
    "I'm not sure how I feel about this. It's okay.",
]

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

for text in texts:
    sentiment_score = analyzer.polarity_scores(text)
    print(f'Text: {text}\nSentiment Score: {sentiment_score}\n')
