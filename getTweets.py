from searchtweets import ResultStream, load_credentials, collect_results,gen_request_parameters
import sys
# import pandas as pd
# import numpy as np
import os, json

v2_search_args = load_credentials(".twitter_keys.yaml", yaml_key="search_tweets_v2", env_overwrite=False)

# takes in 10000 tweets from
stringList=[];
size = 1;
def getTweets(topic):
  for i in range(size):
    query = gen_request_parameters(topic, tweet_fields = "created_at,geo,lang",results_per_call=100,granularity=None)
    tweets = collect_results(query, max_tweets=100,
                          result_stream_args=v2_search_args)
    for i in range(100):
      if(tweets[0]["data"][i]["lang"] != "en"):
        continue
      stringList.append(tweets[0]["data"][i]["text"])
  return stringList



