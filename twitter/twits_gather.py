import tweepy
import pandas as pd
import os

from dotenv import load_dotenv

load_dotenv()

#  keys
CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("ACCESS_SECRET")

# Create an instance of the OAuthHandler class
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
# OAuth authentication
api = tweepy.API(auth)

# The Twitter user
userId = os.getenv("ANNA_JORDANOUS")
# Number of tweets to pull
tweetCount = 1000

# Calling the user_timeline function with parameters
results = api.user_timeline(user_id=userId, count=tweetCount, tweet_mode="extended")
results = list(results)

# foreach through all tweets pulled
df = pd.DataFrame(columns=['content'])
for tweet in results:
    if 'RT' not in tweet.full_text:
        df = df.append({'content': tweet.full_text}, ignore_index=True)

#  save tweets to file
df.to_csv('twitter/twitter_corpuses/annajordanous.csv', index=False)

