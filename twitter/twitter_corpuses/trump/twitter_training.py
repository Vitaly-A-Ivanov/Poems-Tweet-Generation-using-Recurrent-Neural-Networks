import functools
import pickle
import numpy
import pandas as pd
import tensorflow
import re

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

from keras import callbacks

earlystopping = callbacks.EarlyStopping(monitor="loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)

corpus = pd.read_csv('trump-archive.csv', encoding='utf-8-sig')
pd.options.display.max_colwidth = 1000

corpus.drop(columns=['id', 'device', 'isDeleted', 'retweets', 'date', 'isFlagged', 'favorites'], inplace=True)



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
for i, text in enumerate(corpus['isRetweet']):
    if text == 't':
        corpus.drop([i], inplace=True)
corpus = corpus.reset_index()
corpus.drop('index', axis=1, inplace=True)

corpus.text = corpus['text'].apply(cleanTweets)


# get rid rows  with empty tweet
for i, text in enumerate(corpus['text']):
    if not (len(text.strip())):
        corpus.drop([i], inplace=True)

corpus = corpus.reset_index()
corpus.drop(['index', 'isRetweet'], axis=1, inplace=True)

corpus.to_csv('trump.txt', sep='\n', index=False, header=False)

tweetsTxt = open('trump.txt', 'rb').read().decode(encoding='utf-8').lower()

chars = sorted(set(tweetsTxt))

pickle.dump(chars, open('chars_trump.pkl', 'wb'))

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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


model = Sequential()
model.add(LSTM(128,
               input_shape=(sequenceLength,
                            len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01), metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=256, epochs=1, callbacks=[earlystopping])
loss, accuracy = model.evaluate(
    X_test, y_test
)
print('Loss: ', loss)
print('Accuracy: ', loss)

# model.save('model_twitter_trump.h5', hist)
print("Done")
