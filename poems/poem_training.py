import pickle
import numpy
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

from keras import callbacks

earlystopping = callbacks.EarlyStopping(monitor="loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)

path = tensorflow.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path, 'rb').read().decode(encoding='utf-8').lower()

chars = sorted(set(text))

pickle.dump(chars, open('poem_model/chars.pkl', 'wb'))

charsToIndex = dict((c, i) for i, c in enumerate(chars))
indexToChars = dict((i, c) for i, c in enumerate(chars))



sequenceLength = 40
stepLength = 3

sentences = []
nextChar = []

for i in range(0, len(text) - sequenceLength, stepLength):
    sentences.append(text[i: i + sequenceLength])
    nextChar.append(text[i + sequenceLength])

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

hist = model.fit(x, y, batch_size=256, epochs=100, callbacks=[earlystopping])

model.save('poem_model/model_poem.h5', hist)
print("Done")


