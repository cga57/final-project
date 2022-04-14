from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV

from tweet_cluster import vectorize
from utils import tokenize_tweets
import numpy as np
import pandas as pd

sentimental_analysis = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
     alpha=1e-3, random_state=42, max_iter=5, tol=None))
])

parameters = {
    'clf__loss': ('hinge', 'huber', 'perceptron'),
    'clf__alpha': (1e-1, 1e-3),
    'clf__penalty': ('l2', 'l1', 'elasticnet'),
    'clf__max_iter': (500, 2000),
    'clf__learning_rate': ('adaptive', 'optimal', 'invscaling')
}

data = pd.read_csv("sa_dataset.csv", header=None, names=[
                      "sentiment", "id", "date", "query", "username", "text"], encoding="ISO-8859-1")

tweets = tokenize_tweets(data, "text", "tokenized")
X = data["text"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state=42)

gs_clf = GridSearchCV(sentimental_analysis, parameters, cv=5, n_jobs=32)
gs_clf.fit(X_train, y_train)
print(gs_clf.score(X_test, y_test))

print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

