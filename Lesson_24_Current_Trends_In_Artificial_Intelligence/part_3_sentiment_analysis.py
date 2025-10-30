import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Sample customer reviews
reviews = [
    "I absolutely love my new shoes! They fit perfectly and are super comfortable.",
    "The service was terrible, and I will not be returning.",
    "It was an okay experience; nothing too special.",
]

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

for review in reviews:
    sentiment_score = analyzer.polarity_scores(review)
    print(f'Review: {review}\nSentiment Score: {sentiment_score}\n')
