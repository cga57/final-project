from searchtweets import ResultStream, load_credentials, collect_results,gen_request_parameters
# import pandas as pd
# import numpy as np

import os, json

v2_search_args = load_credentials(
    ".twitter_keys.yaml", yaml_key="search_tweets_v2", env_overwrite=False)

# testing with a sandbox account
query = gen_request_parameters("covid",tweet_fields = "created_at,geo,lang",
 results_per_call=100,granularity=None)

tweets = collect_results(query, max_tweets=1,
                         result_stream_args=v2_search_args)

print (tweets)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(tweets, f, ensure_ascii=False, indent=4)



