from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

def vectorize(input_df, vectorizer, input_col):
    X = vectorizer.fit_transform(input_df[input_col])
    return X

class TweetCluster:
    def __init__(self, num_clusters=3) -> None:
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True, min_df=5, max_df=0.95)

        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.pca = PCA(n_components=2, random_state=42)
        self.nmf = NMF(n_components=num_clusters, random_state=42, init=None)
        self.X = None
        self.data = None

    def fit_kmeans(self, input_df, input_col):
        self.X = vectorize(input_df, vectorizer=self.vectorizer,
                           input_col="tokenized")

        self.kmeans.fit(self.X)

        pca_vecs = self.pca.fit_transform(self.X.toarray())

        # Source: https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7
        x0 = pca_vecs[:, 0]
        x1 = pca_vecs[:, 1]

        input_df['cluster'] = self.kmeans.labels_
        input_df['x0'] = x0
        input_df['x1'] = x1

        self.data = input_df

        return input_df


    def fit_nmf(self, input_df, input_col):
        self.X = vectorize(input_df, vectorizer=self.vectorizer,
                           input_col="tokenized")

        self.nmf.fit(self.X)

        # Source: https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7
        pca_vecs = self.pca.fit_transform(self.X.toarray())

        x0 = pca_vecs[:, 0]
        x1 = pca_vecs[:, 1]

        input_df['x0'] = x0
        input_df['x1'] = x1

        self.data = input_df

        return input_df

    # Source: https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7
    def visualize_kmeans(self):
        pathlib.Path('output').mkdir(exist_ok=True)
        plt.figure(figsize=(12, 7))
        plt.title("Kmeans cluster distribution", fontdict={"fontsize": 18})
        sns.scatterplot(data=self.data, x='x0', y='x1', hue='cluster')
        plt.savefig('output/kmeans_cluster.png')
        
        plt.figure(figsize=(12, 7))
        plt.title("Bar Chart for number of tweets", fontdict={"fontsize": 18})
        plt.xlabel("Cluster Group Number", fontdict={"fontsize": 16})
        plt.ylabel("Number of Tweets", fontdict={"fontsize": 16})

        myFrame = self.data.groupby("cluster").count()
        barChartXAxis = myFrame.index.values.tolist()
        plt.bar(barChartXAxis,myFrame["text"])
        plt.savefig('output/kmeans_cluster_size.png')



        plt.figure(figsize=(12, 7))
        plt.title("Topic and Avg Tweet sentiment(PNN)", fontdict={"fontsize": 18})
        plt.xlabel("Topic Number", fontdict={"fontsize": 16})
        plt.ylabel("Avg Tweet sentiment(0-Negative, 2-Neutral, 4-Postive)", fontdict={"fontsize": 16})
        
        df = self.data.groupby("cluster").mean()
        plt.bar(barChartXAxis, df["sentiment_pnn"])
        plt.savefig('output/kmeans_cluster_sentiment_pnn.png')



        plt.figure(figsize=(12, 7))
        plt.title("Topic and Avg Tweet sentiment(PN)", fontdict={"fontsize": 18})
        plt.xlabel("Topic Number", fontdict={"fontsize": 16})
        plt.ylabel("Avg Tweet sentiment(0-Negative, 4-Postive)", fontdict={"fontsize": 16})
        
        df = self.data.groupby("cluster").mean()
        plt.bar(barChartXAxis, df["sentiment_pn"])
        plt.savefig('output/kmeans_cluster_sentiment_pn.png')

    # # Source: https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7
    # def visualize_nmf(self):
    #     plt.figure(figsize=(12, 7))
    #     plt.title("Non-Negative Matrix Factorization cluster distribution", fontdict={"fontsize": 18})
    #     pathlib.Path('output').mkdir(exist_ok=True)
    #     sns.scatterplot(data=self.data, x='x0', y='x1', hue='cluster')
    #     plt.savefig('output/nmf_cluster.png')

    # Source: https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7
    # This code is adapted from a function in this article
    def top_keywords_per_topic(self, count=4):
        df = pd.DataFrame(self.X.todense()).groupby(self.kmeans.labels_).mean()
        keywords = self.vectorizer.get_feature_names_out()
        topic_keywords = {}
        for i, row in df.iterrows():
            sorted_indices = np.flip(np.argsort(row))
            top_keywords = []
            [top_keywords.append(keywords[idx])
             for idx in sorted_indices[:count]]
            topic_keywords[i] = top_keywords

        return topic_keywords

    # Source: https://blog.mlreview.com/topic-modeling-with-scikit-learn-e80d33668730
    # This code is heavily based on display_topic() function in this article
    def top_keywords_per_topic_nmf(self, count=4):
        topic_keywords = {}
        keywords = self.vectorizer.get_feature_names_out()
        for i, topic in enumerate(self.nmf.components_):
            top_keywords = [keywords[i] for i in topic.argsort()[:-count - 1:-1]]
            topic_keywords[i] = top_keywords

        return topic_keywords
    
    def top_tweets_per_topic(self):
        max_keywords = 6
        topic_keywords = self.top_keywords_per_topic(max_keywords)
        self.data["tweet_score"] = self.data.apply(
            lambda row: get_tweet_score(row, topic_keywords), axis=1)

        return self.data.sort_values(by='tweet_score', ascending=False)

    def sentiment_per_topic(self):
        topic_sentiments = self.data.groupby("cluster").mean()


# A tweet score refers to the amount of top keywords of a topic
# present in that tweet. If tweetX belongs topicK and tweetX contains the top 6 key words
# in topicK tweetX will have a score of 6
def get_tweet_score(row, keywords):
    score = 0
    token_array = row.tokenized.split(' ')
    weight = 6
    for word in keywords[row.cluster]:
        if word in token_array:
            score += weight
        weight -= 1
    return score


