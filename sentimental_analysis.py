# Training Data Credits goes to http://help.sentiment140.com/
from numpy import vectorize
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

nltk.download([
    "names",
    "stopwords",
    "vader_lexicon",
])

# https://www.nltk.org/api/nltk.tokenize.casual.html?highlight=tweet#nltk.tokenize.casual.TweetTokenizer

def tokenizer(tweet):
    tweet_tokenizer = TweetTokenizer()
    return tweet_tokenizer.tokenize(tweet)
 
 
 

data = pd.read_csv("sa_dataset.csv", header=None, names=["sentiment", "id", "date", "query", "username", "text"], encoding="ISO-8859-1")
data = data.iloc[1500000: , :]
X = data["text"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state=42)



vectorizer = TfidfVectorizer(min_df=20, max_df=0.95, ngram_range=(1,1), stop_words='english', tokenizer=tokenizer)


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(vectorizer.fit_transform(X_train), y_train)

print(knn.score(vectorizer.transform(X_test), y_test))
print(knn.predict(vectorizer.transform(["h"])))
# print(X_train, y_train)

# sa = data.drop(columns=["id", "username", "date", "query"])

# import spacy

# nlp = spacy.load("en_core_web_lg")  # make sure to use larger package!
# doc = nlp("Lebron James is a basketball player")

# filtered_tokens = [token for token in doc if not token.is_stop]

