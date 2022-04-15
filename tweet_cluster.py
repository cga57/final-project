from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')


def text_to_tokens(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^A-Za-z]+", " ", text)

    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]

    text = " ".join(tokens)
    text = text.lower().strip()

    return text


# Given a corpus output a dataframe
def get_data(input):
    df = pd.read_csv(input,
                     header=None,
                     names=["sentiment", "id", "date",
                            "query", "username", "text"],
                     encoding="ISO-8859-1")

    return df





def vectorize(input_df, vectorizer, input_col):
    X = vectorizer.fit_transform(input_df[input_col])
    return X


def kmeans(X):
    kmeans.fit(X)


def get_top_keywords(n_terms, X, clusters, vectorizer):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(
        clusters).mean()
    print(df)  # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out()  # access tf-idf terms
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        # for each row of the dataframe, find the n terms that have the highest tf idf score
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]]))

    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    print(df['tokenized'])

    print(X.toarray())

    # initialize KMeans with 3 clusters

    kmeans.fit(X)

    # store cluster labels in a variable
    clusters = kmeans.labels_

    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    df['cluster'] = clusters
    df['x0'] = x0
    df['x1'] = x1

    get_top_keywords(10, X, clusters, vectorizer)


class TweetCluster:
    def __init__(self, num_clusters=3) -> None:
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True, min_df=5, max_df=0.95)

        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.pca = PCA(n_components=2, random_state=42)
        self.lda = LatentDirichletAllocation(n_components=num_clusters, random_state=42)
        self.X = None
        self.data = None

    def fit(self, input_df, input_col):
        from utils import tokenize_tweets

        tokenize_tweets(input_df, input_col, output_col="tokenized", remove_stopwords=True)

        self.X = vectorize(input_df, vectorizer=self.vectorizer,
                  input_col="tokenized")

        self.kmeans.fit(self.X)

        pca_vecs = self.pca.fit_transform(self.X.toarray())

        # save our two dimensions into x0 and x1
        x0 = pca_vecs[:, 0]
        x1 = pca_vecs[:, 1]

        input_df['cluster'] = self.kmeans.labels_
        input_df['x0'] = x0
        input_df['x1'] = x1

        self.data = input_df

        return input_df

        
    def get_clusters(self, keywords=4):
        df = pd.DataFrame(self.X.todense()).groupby(self.kmeans.labels_).mean()
        terms = self.vectorizer.get_feature_names_out()  # access tf-idf terms
        for i, r in df.iterrows():
            print('\nTopic {}'.format(i + 1))
            # for each row of the dataframe, find the n terms that have the highest tf idf score
            print(','.join([terms[t] for t in np.argsort(r)[-keywords:]]))
    
    def visualize(self):
        # set image size
        plt.figure(figsize=(12, 7))
        # set a title
        plt.title("TF-IDF + KMeans 20newsgroup clustering", fontdict={"fontsize": 18})
        # set axes names
        plt.xlabel("X Axis", fontdict={"fontsize": 16})
        plt.ylabel("Y Axis", fontdict={"fontsize": 16})
        # create scatter plot with seaborn, where hue is the class used to group the data
        sns.scatterplot(data=self.data, x='x0', y='x1', hue='cluster', palette="viridis")
        plt.show()
        
        plt.figure(figsize=(12, 7))
        plt.title("Bar Chart for number of tweets", fontdict={"fontsize": 18})
        plt.xlabel("Cluster Group Number", fontdict={"fontsize": 16})
        plt.ylabel("Number of Tweets", fontdict={"fontsize": 16})

        myFrame = self.data.groupby("cluster").count()
        barChartXAxis = myFrame.index.values.tolist()
        plt.bar(barChartXAxis,myFrame["text"])
        plt.show()





