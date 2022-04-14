from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
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


categories = [
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'alt.atheism',
    'soc.religion.christian',
]

# Given a corpus output a dataframe


def get_data(input):
    df = pd.read_csv(input,
                     header=None,
                     names=["sentiment", "id", "date",
                            "query", "username", "text"],
                     encoding="ISO-8859-1")

    return df


def tokenize(input_df):
    from utils import tweet_tokenizer
    input_df["tokenized"] = input_df["text"].apply(
        lambda tweet: tweet_tokenizer(tweet))

    return input_df


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


def main(input):
    # dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, remove=(
    #     'headers', 'footers', 'quotes'))

    # df = pd.DataFrame(dataset.data, columns=["corpus"])
    # df["cleaned"] = df["corpus"].apply(lambda text: text_to_tokens(text))

    # print(df)
     
    df = get_data(input)
    df = tokenize(df)

    # initialize vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    print(df['tokenized'])
    X = vectorizer.fit_transform(df['tokenized'])
    print(X.toarray())

    # initialize KMeans with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # store cluster labels in a variable
    clusters = kmeans.labels_
    # print(clusters)

    # # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ms.fit(X.toarray())
    # labels = ms.labels_
    # print(labels)

    # cluster_centers = ms.cluster_centers_

    # labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)

    # print("number of estimated clusters : %d" % n_clusters_)
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


if __name__ == '__main__':
    import sys

    in_directory = sys.argv[1]
    main(in_directory)
    # main()
