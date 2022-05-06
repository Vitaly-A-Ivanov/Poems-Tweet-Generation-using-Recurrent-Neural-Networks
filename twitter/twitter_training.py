import pickle
import numpy
import pandas as pd
import re

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

from keras import callbacks

#  callback function to stop overfitting the model
earlystopping = callbacks.EarlyStopping(monitor="loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)

#  read csv file with available tweets dataset
tweets = pd.read_csv('twitter_corpuses/annajordanous.csv')
pd.options.display.max_colwidth = 1000


#  function to clean tweets from links and mentions
def cleanTweets(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"www.\S+", "", tweet)
    tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)
    tweet = re.sub("#[A-Za-z0-9_]+", "", tweet)
    tweet = " ".join(tweet.split())
    return tweet


# clean tweets from links and mentions
tweets.content = tweets['content'].apply(cleanTweets)

# get rid rows  with empty tweet after the cleaning
for i, text in enumerate(tweets['content']):
    if not (len(text.strip())):
        tweets.drop([i], inplace=True)

# reset dataframe indexing
tweets = tweets.reset_index()

# delete newly created index column
tweets.drop('index', axis=1, inplace=True)

# save clean tweets to text file
tweets.to_csv('twitter_corpuses/annajordanous.txt', sep='\n', index=False, header=False)

# read clean tweets from text file
tweetsTxt = open('twitter_corpuses/annajordanous.txt', 'rb').read().decode(encoding='utf-8').lower()

# create all unique characters that exists in a text in a sorted way
chars = sorted(set(tweetsTxt))

# save all unique characters that exists in a text to the file for later usage
pickle.dump(chars, open('twitter_models/annajordanous/chars.pkl', 'wb'))

# create two dictionaries and map each character to its index and other way round to represent them as a numerical
charsToIndex = dict((c, i) for i, c in enumerate(chars))
indexToChars = dict((i, c) for i, c in enumerate(chars))

# length of the sentence
sequenceLength = 40
#  number of characters as step to start the next sentence
stepLength = 3

#  training data for the sentences
sentences = []
#  training data for the characters
nextChar = []

# training data creation.
# Iteration over the whole text taking whole sentence with given sequence length and saving the next character
for i in range(0, len(tweetsTxt) - sequenceLength, stepLength):
    sentences.append(tweetsTxt[i: i + sequenceLength])
    nextChar.append(tweetsTxt[i + sequenceLength])

#  two numpy arrays with full of zeros and length of sentences and characters with a boolean type
x = numpy.zeros((len(sentences), sequenceLength,
                 len(chars)), dtype=numpy.bool)
y = numpy.zeros((len(sentences),
                 len(chars)), dtype=numpy.bool)

#  convert training data into numerical format
for idx, sentence in enumerate(sentences):
    for t, c in enumerate(sentence):
        x[idx, t, charsToIndex[c]] = 1
    y[idx, charsToIndex[nextChar[idx]]] = 1

# build a neural network model
model = Sequential()
model.add(LSTM(128,
               input_shape=(sequenceLength,
                            len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01), metrics=['accuracy'])

hist = model.fit(x, y, batch_size=256, epochs=4, callbacks=[earlystopping])

model.save('twitter_models/annajordanous/model_twitter_anna.h5', hist)
print("Done")
