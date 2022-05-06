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

c1 = pd.read_csv('covid_1.csv', encoding='utf-8-sig')
c2 = pd.read_csv('covid_2.csv', encoding='utf-8-sig')
pd.options.display.max_colwidth = 1000

frames = [c1, c2]

corpus = pd.concat(frames)

corpus.drop(columns=['id', 'created_at', 'source', 'lang', 'favorite_count', 'retweet_count', 'original_author', 'hashtags', 'user_mentions', 'place', 'original_text', 'compound', 'neg', 'neu', 'pos', 'sentiment'], inplace=True)

corpus.to_csv('covid.txt', sep='\n', index=False, header=False)

tweetsTxt = open('covid.txt', 'rb').read().decode(encoding='utf-8').lower()

chars = sorted(set(tweetsTxt))

pickle.dump(chars, open('covid.pkl', 'wb'))

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

hist = model.fit(x, y, batch_size=256, epochs=4, callbacks=[earlystopping])

model.save('model_covid.h5', hist)
print("Done")
