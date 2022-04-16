from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
# Trains a new model from scratch
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from utils import tokenize_tweets
import pandas as pd
import numpy as np
sentimental_analysis = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
     alpha=1e-5, random_state=42, max_iter=5, learning_rate="optimal", tol=None))
])

def train_model_pn():
    data = pd.read_csv("sa_dataset.csv", header=None, names=[
                      "sentiment", "id", "date", "query", "username", "text"], encoding="ISO-8859-1")

    tweets = tokenize_tweets(data, "text", "tokenized")
    X = tweets["tokenized"]
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state=42)

    sentimental_analysis.fit(X_train, y_train)
    print("New model(PN) accuracy: ", sentimental_analysis.score(X_test, y_test))
    return sentimental_analysis

def train_model_pnn():
    data_train = pd.read_csv("datasets/twitter_training.csv", header=None, names=["id", "entity", "sentiment", "text"])
    data_train.loc[data_train.sentiment == "Positive", 'sentiment'] = np.dtype('int32').type(4)
    data_train.loc[data_train.sentiment == "Neutral", 'sentiment'] = np.dtype('int32').type(2)
    data_train.loc[data_train.sentiment == "Irrelevant", 'sentiment'] = np.dtype('int32').type(2)
    data_train.loc[data_train.sentiment == "Negative", 'sentiment'] = np.dtype('int32').type(0)
    data_train.dropna(inplace=True)

    data_test = pd.read_csv("datasets/twitter_validation.csv", header=None, names=["id", "entity", "sentiment", "text"])
    data_test.loc[data_test.sentiment == "Positive", 'sentiment'] = np.dtype('int32').type(4)
    data_test.loc[data_test.sentiment == "Neutral", 'sentiment'] = np.dtype('int32').type(2)
    data_test.loc[data_test.sentiment == "Irrelevant", 'sentiment'] = np.dtype('int32').type(2)
    data_test.loc[data_test.sentiment == "Negative", 'sentiment'] = np.dtype('int32').type(0)
    data_test.dropna(inplace=True)


    # Work with training data
    tweets_train = tokenize_tweets(data_train, "text", "tokenized")
    X_train = tweets_train["tokenized"]
    y_train = data_train["sentiment"].astype('int32')

    # Prepare testing data
    tweets_test = tokenize_tweets(data_test, "text", "tokenized")
    X_test = tweets_test["tokenized"]
    y_test = data_test["sentiment"].astype('int32')

    sentimental_analysis.fit(X_train, y_train)
    print("New model(PNN) accuracy: ", sentimental_analysis.score(X_test, y_test))
    return sentimental_analysis