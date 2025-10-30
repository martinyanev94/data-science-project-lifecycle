from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset loading
reviews = pd.read_csv('reviews.csv')  # Assume this dataset contains text reviews and sentiments

# Feature and target variable definition
X_reviews = reviews['review_text']  # Features
y_reviews = reviews['sentiment']  # Target variable indicating sentiment

# Transforming text data into numerical format
vectorizer = CountVectorizer()
X_reviews_vectorized = vectorizer.fit_transform(X_reviews)

# Splitting the dataset
X_train_reviews, X_test_reviews, y_train_reviews, y_test_reviews = train_test_split(X_reviews_vectorized, y_reviews, test_size=0.2, random_state=42)

# Model training
sentiment_model = MultinomialNB()
sentiment_model.fit(X_train_reviews, y_train_reviews)

# Predictions
y_reviews_pred = sentiment_model.predict(X_test_reviews)

# Evaluation
print(confusion_matrix(y_test_reviews, y_reviews_pred))
print(classification_report(y_test_reviews, y_reviews_pred))
