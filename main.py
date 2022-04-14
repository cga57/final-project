import sys
import pandas as pd
import json

from getTweets import getTweets
from tweet_cluster import TweetCluster

from sentimental_analysis import get_samodel
from utils import tokenize_tweets

def main(topic):
    # Get tweets for the topic. This will be a list of tweets
    print("Retrieving Tweets")
    tweets_list = getTweets(topic)

    print("Loading tweets into dataframe")
    tweets_df = pd.DataFrame(tweets_list, columns = ["text"])

    print("Tokenizing and cleaning tweets")
    tweets_df = tokenize_tweets(tweets_df, "text", "tokenized", remove_stopwords=False)

    # Use the sklearn model to add the sentiment scores for each tweet
    sa_model = get_samodel()

    print("Predicting tweet sentiments")
    tweets_df["sentiment"] = sa_model.predict(tweets_df["tokenized"])

    print("Checkout rand_samples.json for some predictions")
    save_rand_samples(tweets_df)

    print("Trying to cluster tweets into groups to find main topics")
    tweets_cluster = TweetCluster()
    tweets_df_clusters = tweets_cluster.fit(tweets_df, "text")
    print(tweets_df_clusters)
    tweets_cluster.get_clusters(keywords=4)


def save_rand_samples(input_df, count=5):
    sample = input_df.sample(n=count)
    tweets = []
    for index, row in sample.iterrows():
        tweet = {
            'tweet': row["text"],
            'tokenized': row["tokenized"],
            'sentiment': row["sentiment"]
        } 
        tweets.append(tweet)

    with open("rand_samples.json","w", encoding='utf-8') as jsonfile:
        json.dump(tweets, jsonfile, ensure_ascii=False, indent=3)
    # json_file = open('rand_samples.json', 'wb')
    # json_string = json.dumps(tweets, indent=3)

    # json_file.write()
    # print(json_string)



if __name__ == '__main__':
    import sys

    topic = sys.argv[1]
    main(topic)