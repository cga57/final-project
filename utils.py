from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
stop_words = set(nltk.corpus.stopwords.words('english'))

# Takes a tweet and convert it into tokens
# @username type of tokens will be removed
# stopwords will be removed
# TODO: REMOVE PUNCTUATION
def tweet_tokenizer(tweet, remove_stopwords=True):
    tweet = re.sub(r"http\S+", "", tweet.lower())
    
    tweet_tokenizer = TweetTokenizer()
    with_stopwords = tweet_tokenizer.tokenize(tweet)

    without_stopwords = []
    for word in with_stopwords:
        if word[0] == "@":
            continue
        if word == "rt":
            continue
        if word in stop_words and remove_stopwords:
            continue
        
        word = re.sub("[^a-z]+", "", word)
        if word == "":
            continue
        
        without_stopwords.append(word)

    text = " ".join(without_stopwords)
    text = text.strip()

    return text

def tokenize_tweets(input_df, input_col, output_col, remove_stopwords=True):
    input_df[output_col] = input_df[input_col].apply(
        lambda tweet: tweet_tokenizer(tweet, remove_stopwords))

    return input_df