# final-project

parameters for twieet field in call:
[attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,non_public_metrics,organic_metrics,possibly_sensitive,promoted_metrics,public_metrics,referenced_tweets,reply_settings,source,text,withheld]

## Project Plan

1. We get twitter tweets into JSON format
2. 


- Input file: 7 days of twitter tweets (.json)
- TA runs the main.py
- It reads this .json file into pandas dataframe
- We use our Sentimental Analysis model(Random Forest) and give each of your tweets a score of positivity
- We use our ( Top story classification model) which takes tweets as input it produces the top 3 stories and their score.


- Visualization(of TOP 3)




## Our Python Project
- get_tweets.py
- sentimental_analysis.py (Output model)
- top_story_model.py (Output model)
- main.py (inputs - 7_day_tweets.json, sa_model, top_story_model)

# Our Project Tasks
## Getting Twitter Tweets
## Train Sentimental Analysis Model
## Train Top Story Model
## Data Analysis of Recent Twitter Tweets (main.py)

## Credits/Sources
### Dataset
We have two sentimental analysis models. The differentiating factor between them is the dataset they were trained on:
    - Positive and Negative Sentiment Dataset(1.6 million tweets) - http://help.sentiment140.com/
    - Positive, Neutral and Negative Sentiment Dataset(76 thousand tweets) - https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
- https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7
    - 