import nltk
from nltk.tokenize import TweetTokenizer
import re

stop_words = set(nltk.corpus.stopwords.words('english'))

def tokenize_tweets(input_df, input_col, output_col, remove_stopwords=True):
    input_df[output_col] = input_df[input_col].apply(
        lambda tweet: tweet_tokenizer(tweet, remove_stopwords))

    return input_df


# Takes a tweet and convert it into tokens
# @username type of tokens will be removed
# stopwords will be removed
def tweet_tokenizer(tweet, remove_stopwords=True):
    # Remove links right away or it messes with later steps
    tweet = re.sub(r"http\S+", "", tweet.lower())
    
    tweet_tokenizer = TweetTokenizer()
    with_stopwords = tweet_tokenizer.tokenize(tweet)

    without_stopwords = []
    for word in with_stopwords:
        # Remove usernames
        if word[0] == "@":
            continue
        # Remove retweet heading
        if word == "rt":
            continue
        if len(word) < 4:
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