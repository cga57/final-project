from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
# Trains a new model from scratch
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from utils import tokenize_tweets
import pandas as pd

sentimental_analysis = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
     alpha=1e-3, random_state=42, max_iter=5, learning_rate="optimal", tol=None))
])

def train_model():
    data = pd.read_csv("sa_dataset.csv", header=None, names=[
                      "sentiment", "id", "date", "query", "username", "text"], encoding="ISO-8859-1")

    tweets = tokenize_tweets(data, "text", "tokenized")
    X = tweets["tokenized"]
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state=42)

    sentimental_analysis.fit(X_train, y_train)
    print("New model accuracy: ", sentimental_analysis.score(X_test, y_test))
    return sentimental_analysis