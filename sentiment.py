# Import packages.
from bs4 import BeautifulSoup
import requests
import pandas as pd
import nltk
import random

# Web scraping
response = requests.get('http://www.wsj.com')
soup = BeautifulSoup(response.text, 'html.parser')
headlines = soup.select('h2') 
df = pd.DataFrame({'Headlines': [headline.text for headline in headlines]})

# NLP preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return " ".join(stemmed_tokens)

# Apply preprocessing
df['Processed_Headlines'] = df['Headlines'].apply(preprocess_text)

# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate random sentiments
sentiments = ['Positive', 'Negative', 'Neutral']
df['Sentiment'] = [random.choice(sentiments) for _ in range(len(df))]

# Print column names
print(df.columns)

# Split dataset
X = df['Processed_Headlines']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Classification
clf = MultinomialNB()
clf.fit(X_train_vect, y_train)
y_pred = clf.predict(X_test_vect)

# Evaluation
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))
