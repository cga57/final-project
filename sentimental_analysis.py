# Training Data Credits goes to http://help.sentiment140.com/
from operator import mod
from numpy import indices
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

import nltk
from nltk.tokenize import TweetTokenizer

from pyspark import SparkConf
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.feature import HashingTF, IDF, MinMaxScaler
from pyspark.sql.functions import *
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.linalg import DenseVector

spark = SparkSession.builder.appName('Twitter Sentimental Analysis').getOrCreate()
spark.sparkContext.setLogLevel('WARN')


assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.3'  # make sure we have Spark 2.3+


nltk.download([
    "names",
    "stopwords",
])

tweet_data_schema = types.StructType([
    types.StructField("sentiment", types.IntegerType(), True),
    types.StructField("id", types.IntegerType(), True),
    types.StructField("date", types.DateType(), True),
    types.StructField("query", types.StringType(), True),
    types.StructField("username", types.StringType(), True),
    types.StructField("text", types.StringType(), True)
])

stop_words = set(nltk.corpus.stopwords.words('english'))

# https://www.nltk.org/api/nltk.tokenize.casual.html?highlight=tweet#nltk.tokenize.casual.TweetTokenizer
# https://docs.anaconda.com/anaconda-scale/howto/spark-nltk/
# For example, "I love breakfast!" => ["I", "love", "breakfast", "!"]
# The tokenizer also deals with stuff like "@someuser"
# also removes stopwords
def tokenizer(tweet):
    tweet_tokenizer = TweetTokenizer()
    with_stopwords = tweet_tokenizer.tokenize(tweet.lower())

    without_stopwords = []
    for word in with_stopwords:
        if (word[0] != "@"):
            without_stopwords.append(word)

    return without_stopwords

# # https://stackoverflow.com/a/65624870
def expand_features(feature):
    v = DenseVector(feature)

    # expanded = list([float(x) for x in v])

    return v 

def main(in_dir):
    # Lets register the tokenizer function as udf
    tokenizer_udf = functions.udf(
        tokenizer, types.ArrayType(types.StringType()))


    # Lets read in the entire data set
    tweets = spark.read.csv(in_dir, header=None,
                            schema=tweet_data_schema, encoding="ISO-8859-1")

    tweets = tweets.withColumn(
        "tokenized", tokenizer_udf(tweets["text"]))

    # https://spark.apache.org/docs/latest/ml-features.html#tf-idf
    hash_transformer = HashingTF(inputCol="tokenized", outputCol="raw_vector", numFeatures=1500)
    tweets = hash_transformer.transform(tweets)

    idf = IDF(inputCol="raw_vector", outputCol="features")
    idf_model = idf.fit(tweets)
    tweets = idf_model.transform(tweets)

    
    indexer = StringIndexer(inputCol = 'sentiment', outputCol = 'label')
    tweets = indexer.fit(tweets).transform(tweets)
    # tweets.show(5)
    # tweets.select('sentiment','label').groupBy('sentiment','label').count().show()
    # assembler = VectorAssembler(inputCols="sentiment", outputCol="sentiment_scaled")
    # tweets = assembler.transform(tweets)
    # tweets = tweets.withColumn("features", expand_features_udf(tweets["features_sparse"]))

    # Scale Data
    # scaler_tweets = MinMaxScaler(min=0.0, max=1.0, inputCol='features_sparse', outputCol='features_scaled')
    # tweets = scaler_tweets.fit(tweets).transform(tweets)

    # scaler_sent = MinMaxScaler(min=0.0, max=1.0, inputCol='sentiment', outputCol='sentiment_scaled')
    # tweets = scaler_sent.fit(tweets).transform(tweets)

    # tweets.write.json("test.json", mode="overwrite")
    # tweets.show()

    # We split the datasets to test and train
    (tweets_train,tweets_test) = tweets.randomSplit([0.7, 0.3])

    # # We train the training data
    dt = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=[1500, 700, 2], maxIter=100,blockSize=128,seed=42, stepSize=0.025)
    model = dt.fit(tweets_train)
    model.save('model')
    predictions = model.transform(tweets_test)
    predictions.select("label", "prediction", "text").show(5)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions) 
    print("Test Error = %g " % (1.0 - accuracy))
    print("Accuracy = %g " % accuracy)

# Now, we create a vectorizer/word embeddings that convert our tweets into vectors to be used in when training the model.
# vectorizer = TfidfVectorizer(min_df=20, max_df=0.95, ngram_range=(1,1), stop_words='english', tokenizer=tokenizer)

# text = ["The quick brown fox jumped over the lazy dog.",
# 	"The dog.",
# 	"The fox"]


# data = data.iloc[1500000: , :]
# X = data["text"]
# y = data["sentiment"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state=42)


# knn = KNeighborsClassifier(n_neighbors=5)

# knn.fit(vectorizer.fit_transform(X_train), y_train)

# print(knn.score(vectorizer.transform(X_test), y_test))
# print(knn.predict(vectorizer.transform(["h"])))
# # print(X_train, y_train)

# # sa = data.drop(columns=["id", "username", "date", "query"])

# # import spacy

# # nlp = spacy.load("en_core_web_lg")  # make sure to use larger package!
# # doc = nlp("Lebron James is a basketball player")

# # filtered_tokens = [token for token in doc if not token.is_stop]


if __name__ == '__main__':
    in_directory = sys.argv[1]
    main(in_directory)
