import sys
import pandas as pd
import json
import yaml

from getTweets import getTweets
from tweet_cluster import TweetCluster

from sentimental_analysis import get_samodel_pn
from sentimental_analysis import get_samodel_pnn
from utils import tokenize_tweets

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def main(options):
    tweets_df = get_and_clean_tweets(options["search_term"], options["include_search_term"])

    tweets_df = get_sentiment_prediction(tweets_df)

    save_rand_samples(tweets_df, options["random_tweets_count"])

    sentiments, keywords, tweets = calculate_topics(tweets_df, num_clusters=options["topics_count"], num_keywords=options["keywords_count"], alg="kmeans")
    print_topic_stats(sentiments, keywords, tweets)

    sentiments, keywords, tweets = calculate_topics(tweets_df, num_clusters=options["topics_count"], num_keywords=options["keywords_count"], alg="nmf")
    print_topic_stats(sentiments, keywords, tweets)


def get_and_clean_tweets(search_term, include_search_term):
    print(f"Your chosen topic: {search_term}")
    print("Retrieving Tweets")
    tweets_list = getTweets(search_term)
    print("Valid tweets found:", len(tweets_list))

    print("Loading tweets into dataframe")
    tweets_df = pd.DataFrame(tweets_list, columns = ["text"])

    print("Tokenizing and cleaning tweets")
    if include_search_term:
        tweets_df = tokenize_tweets(tweets_df, "text", "tokenized", remove_word=search_term)
    else:
        tweets_df = tokenize_tweets(tweets_df, "text", "tokenized")

    return tweets_df



def get_sentiment_prediction(tweets_df):
    # We will train/import two sentimental analysis models
    # sa_model_pn => pn stands for Positive and Negative
    # sa_model_pnn => pnn stands for Positive, Negative and Neutral
    sa_model_pn = get_samodel_pn()
    sa_model_pnn = get_samodel_pnn()

    print("Predicting tweet sentiments")
    tweets_df["sentiment_pn"] = sa_model_pn.predict(tweets_df["tokenized"])
    tweets_df["sentiment_pnn"] = sa_model_pnn.predict(tweets_df["tokenized"])

    return tweets_df



def calculate_topics(tweets_df, num_clusters, num_keywords, alg=""):
    if alg == "nmf":
        print(f"Topics found by Non-Negative Matrix Factorization")
        print("-----------------------------------------------\n")
        tweets_cluster = TweetCluster(num_clusters)
        tweets_df_clusters = tweets_cluster.fit_nmf(tweets_df, "text")

        topic_sentiments = None
        topic_top_keywords = tweets_cluster.top_keywords_per_topic_nmf(count=num_keywords)
        topic_top_tweets = None

        # tweets_cluster.visualize_nmf()

    elif alg == "kmeans":
        print(f"\nTrying to cluster tweets with {alg} algorithm")
        print("-----------------------------------------------\n")
        tweets_cluster = TweetCluster(num_clusters)
        tweets_df_clusters = tweets_cluster.fit_kmeans(tweets_df, "text")

        topic_sentiments = tweets_df_clusters.groupby(tweets_df_clusters["cluster"]).mean()
        topic_top_keywords = tweets_cluster.top_keywords_per_topic(count=num_keywords)
        topic_top_tweets = tweets_cluster.top_tweets_per_topic()

        tweets_cluster.visualize_kmeans()
    

    return topic_sentiments, topic_top_keywords, topic_top_tweets



def save_rand_samples(tweets_df, count):
    print("Checkout output/rand_samples.json for some predictions")
    sample_count = min(count, len(tweets_df["text"]))
    sample = tweets_df.sample(n=sample_count)
    tweets = []
    for idx, row in sample.iterrows():
        tweet = {
            'tweet': row["text"],
            'tokenized': row["tokenized"],
            'sentiment_v1(Positive and Negative)': row["sentiment_pn"],
            'sentiment_v2(Positive, Neutral and Negative)': row["sentiment_pnn"],
        } 
        tweets.append(tweet)
    
    import pathlib
    pathlib.Path('output').mkdir(exist_ok=True) 
    
    with open("output/rand_samples.json","w", encoding='utf-8') as jsonfile:
        json.dump(tweets, jsonfile, ensure_ascii=False, indent=3)



def print_topic_stats(sentiments, top_keywords, top_tweets):
    # Print NMF Stats
    if sentiments is None or top_tweets is None:
        for i in top_keywords:
            print(f"Topic: {i}", "No sentiment score available with NMF")
            print("Top keywords:", ", ".join(top_keywords[i]))
            print("\n")
        return
    # Print kmeans stats
    for i, sentiment_row in sentiments.iterrows():
        print(f"Topic: {i}", f"Sentiment(PN): {sentiment_row.sentiment_pn}", f"Sentiment(PNN): {sentiment_row.sentiment_pnn}")
        print("Top keywords:", ", ".join(top_keywords[i]))
        for j, tweet_row in top_tweets.iterrows():
            if tweet_row.cluster == i:
                print("\nTop tweet:", tweet_row.text)
                print("Top tweet score:", tweet_row.tweet_score)
                print("Top tweet sentiment(PN):", tweet_row.sentiment_pn)
                print("Top tweet sentiment(PNN):", tweet_row.sentiment_pnn)
                break
        print("\n\n")
        






if __name__ == '__main__':
    try:
        with open("options.yaml", 'r') as f:
            options = yaml.safe_load(f)
    except:
        print("Parsing 'options.yaml' failed. Loading default options")
        options = {
            "search_term": "Microsoft",
            "topics_count": 3,
            "keywords_count": 4,
            "include_search_term": True,
            "random_tweets_count": 40,
        }
    if len(sys.argv) > 1:
        options["search_term"] = sys.argv[1]
    main(options)