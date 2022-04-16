import nltk
from nltk.tokenize import TweetTokenizer
import re

stop_words = set(nltk.corpus.stopwords.words('english'))

def tokenize_tweets(input_df, input_col, output_col, remove_stopwords=True, remove_word=""):
    input_df[output_col] = input_df[input_col].apply(
        lambda tweet: tweet_tokenizer(tweet, remove_stopwords, remove_word))

    return input_df


# Takes a tweet and convert it into tokens
# @username type of tokens will be removed
# stopwords will be removed
def tweet_tokenizer(tweet, remove_stopwords=True, remove_word=""):
    # Remove links right away or it messes with later steps
    if isinstance(tweet, str):
        tweet = tweet.lower()
    else:
        print(tweet)
    tweet = re.sub(r"http\S+", "", tweet)
    
    tweet_tokenizer = TweetTokenizer()
    tweet_wordlist_raw = tweet_tokenizer.tokenize(tweet)

    tweet_wordlist = []
    for word in tweet_wordlist_raw:
        # Remove usernames
        if word[0] == "@":
            continue
        # Remove retweet heading
        if word == "rt":
            continue
        # Remove small words
        if len(word) <= 1:
            continue

        # Remove the word if its a stop and remove_stopwords is enabled
        if word in stop_words and remove_stopwords:
            continue

        # Used to remove a particular word if needed
        if remove_word != "":
            if word == remove_word:
                continue
        
        word = re.sub("[^a-z]+", "", word)
        if word == "":
            continue
        
        # Append the word's lemma
        tweet_wordlist.append(nltk.stem.WordNetLemmatizer().lemmatize(word))

    # Join list and remove unecessary whitespaces
    text = " ".join(tweet_wordlist)
    text = text.strip()

    return text