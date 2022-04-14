import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
stop_words = set(nltk.corpus.stopwords.words('english'))

# Takes a tweet and convert it into tokens
# @username type of tokens will be removed
# stopwords will be removed
def tweet_tokenizer(tweet, remove_stopwords=True):
    tweet_tokenizer = TweetTokenizer()
    with_stopwords = tweet_tokenizer.tokenize(tweet.lower())

    without_stopwords = []
    for word in with_stopwords:
        if ((word not in stop_words) and (word[0] != "@")):
            without_stopwords.append(word)            
    
    text = " ".join(without_stopwords)
    text = text.lower().strip()

    return text


def sentence_tokenizer(sentence):
    return sentence

def tokenizer(tweet):
    tweet_tokenizer = TweetTokenizer()
    with_stopwords = tweet_tokenizer.tokenize(tweet.lower())

    without_stopwords = []
    for word in with_stopwords:
        if word not in stop_words:
            without_stopwords.append(word)

    return without_stopwords