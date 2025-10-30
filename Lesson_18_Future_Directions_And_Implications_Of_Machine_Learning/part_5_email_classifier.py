from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd

# Load data
data = pd.read_csv('emails.csv')

# Prepare the model
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Fit the model
model.fit(data['text'], data['label'])

# Predict on new data
new_emails = ['Congratulations! You won a lottery!', 'Your account has been updated.']
predictions = model.predict(new_emails)
print(predictions)
