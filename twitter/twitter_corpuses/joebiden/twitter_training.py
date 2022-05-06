import functools
import pickle
import numpy
import pandas as pd
import tensorflow
import re

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

from keras import callbacks

earlystopping = callbacks.EarlyStopping(monitor="loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)

corpus = pd.read_csv('joebiden.csv', encoding='utf-8-sig')
pd.options.display.max_colwidth = 1000

corpus.drop(columns=['id', 'timestamp', 'url', 'replies', 'retweets', 'quotes', 'likes'], inplace=True)


# clean tweets from links and mentions
def cleanTweets(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"www.\S+", "", tweet)
    tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)
    tweet = re.sub("#[A-Za-z0-9_]+", "", tweet)
    tweet = " ".join(tweet.split())
    return tweet


# get rid rows  with empty tweet

corpus.tweet = corpus['tweet'].apply(cleanTweets)

# get rid rows  with empty tweet
for i, text in enumerate(corpus['tweet']):
    if not (len(text.strip())):
        corpus.drop([i], inplace=True)

corpus = corpus.reset_index()
corpus.drop(['index'], axis=1, inplace=True)

corpus.to_csv('joebiden.txt', sep='\n', index=False, header=False)

tweetsTxt = open('joebiden.txt', 'rb').read().decode(encoding='utf-8').lower()

chars = sorted(set(tweetsTxt))

pickle.dump(chars, open('chars_joebiden.pkl', 'wb'))

charsToIndex = dict((c, i) for i, c in enumerate(chars))
indexToChars = dict((i, c) for i, c in enumerate(chars))

sequenceLength = 40
stepLength = 3

sentences = []
nextChar = []

for i in range(0, len(tweetsTxt) - sequenceLength, stepLength):
    sentences.append(tweetsTxt[i: i + sequenceLength])
    nextChar.append(tweetsTxt[i + sequenceLength])

x = numpy.zeros((len(sentences), sequenceLength,
                 len(chars)), dtype=numpy.bool)
y = numpy.zeros((len(sentences),
                 len(chars)), dtype=numpy.bool)

for idx, sentence in enumerate(sentences):
    for t, c in enumerate(sentence):
        x[idx, t, charsToIndex[c]] = 1
    y[idx, charsToIndex[nextChar[idx]]] = 1

model = Sequential()
model.add(LSTM(128,
               input_shape=(sequenceLength,
                            len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01), metrics=['accuracy'])

hist = model.fit(x, y, batch_size=256, epochs=  100, callbacks=[earlystopping])

model.save('model_joebiden.h5', hist)
print("Done")
