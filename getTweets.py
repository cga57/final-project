from searchtweets import ResultStream, load_credentials, collect_results,gen_request_parameters
import sys
# import pandas as pd
# import numpy as np
import os, json

v2_search_args = load_credentials(".twitter_keys.yaml", yaml_key="search_tweets_v2", env_overwrite=False)
topic = sys.argv[1]
# takes in 10000 tweets from
jsonList = [];
combinedJsonList = [];
size = 1;
for i in range(size):
  query = gen_request_parameters(topic, tweet_fields = "created_at,geo,lang",results_per_call=100,granularity=None)
  tweets = collect_results(query, max_tweets=100,
                         result_stream_args=v2_search_args)
  # if(tweets.lang != "en"):
  #   continue;
  
  # print(tweets[0])
  #
  # [print(tweet, end='\n\n') for tweet in tweets[0]]
  for i in range(100):
    if(tweets[0]["data"][i]["lang"] != "en"):
      continue
    with open('hi.txt', 'a', encoding="utf-8") as f:
      f.write(tweets[0]["data"][i]["text"])
      f.write(",")
    # print(tweets[0]["data"][i]["text"])
    

  # print(tweets.data)
  # jsonList.append(tweets);



# join them back together
# for i in range(size):
#   combinedJsonList += jsonList[i];

# print (combinedJsonList)

# with open('data.json', 'w', encoding='utf-8') as f:
#     json.dump(jsonList, f, ensure_ascii=False, indent=4)



