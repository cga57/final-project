# Twitter Sentiment Analysis and Clustering Tweets

## Project Installation and Requirements
Please run the following command:
```
pip install -r requirements.txt
```
Our program was tested on `Python 3.8.10`

## Running the Project
In our root folder, we have a file called `options.yaml`. The purpose of this file is to load some configurable options when running the program (instead of CLI which can be cumbersome)

The file documents each option and why it is there.

Running the project is as simple as the following
```
python3 main.py
```

You can provide the search term either in `option.yaml`, or as CLI argument:
```
python3 main.py "covid 19"
```
When a CLI argument present, it will override the search term set in `options.yaml`

Running the program writes a lot of information to terminal which is where the bulk of the interesting information is. It also produces a folder called `output` which contains some relevant information from running the program


## Credits/Sources
### Dataset
We have two sentimental analysis models. The differentiating factor between them is the dataset they were trained on:
    
- Positive and Negative Sentiment Dataset(1.6 million tweets) - http://help.sentiment140.com/

- Positive, Neutral and Negative Sentiment Dataset(76 thousand tweets) - https://www.kaggle.com/datasets/jp797498e/

### Sentimental Analysis
- Sklearn Docs, Spark docs

### Other sources(graphing, kmeans, nmf etc.)
- https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7
 - https://blog.mlreview.com/topic-modeling-with-scikit-learn-e80d33668730
- https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
