import pickle
import random

import pandas as pd
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import re
stop_words = set(stopwords.words('english'))

steamer = WordNetLemmatizer()

corpus = pd.read_csv('corpus/text_emotion.csv', encoding='utf-8-sig')
corpus.drop(['tweet_id', 'author'], axis=1, inplace=True)


words = []
classes = []
documents = []

punctuation = ['!', '#', '$', '%', '&', "'", "''", "'-", "_"]


def cleanSentences(sentence):
    sentence = re.sub(r"http\S+|www\S+|https\S+", '', sentence, flags=re.MULTILINE)
    sentence = str(sentence.encode("ascii", "ignore"))
    sentence = ''.join([i for i in sentence if not i.isdigit()])
    if sentence.find('@') != -1:
        n = sentence.count('@')
        for j in range(n):
            if sentence.find('@') == -1:
                break
            sub_string = ''
            start_position = sentence.index('@')
            end_position = sentence.find(' ', start_position, )
            if end_position == -1:
                sentence = sentence[:start_position]
                break
            for c in sentence[start_position: end_position]:
                sub_string += c
            sentence = sentence.replace(sub_string, '')
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence


corpus.content = corpus['content'].apply(cleanSentences)

#  get rid rows  with empty content
for i, text in enumerate(corpus['content']):
    if text == '':
        corpus.drop([i], inplace=True)

# reset indexing
corpus.reset_index(inplace=True)

# get rid rows  with empty sentiment
for i, text in enumerate(corpus['sentiment']):
    if text == 'empty':
        corpus.drop([i], inplace=True)

# reset indexing
corpus.drop('index', axis=1, inplace=True)

corpus.to_csv('../corpus/text_emotion_clean.csv', index=False)
print('done')


for idx in corpus.index:
    wordList = nltk.word_tokenize(corpus['content'][idx])
    words.extend(wordList)
    documents.append((wordList, corpus['sentiment'][idx]))
    if corpus['sentiment'][idx] not in classes:
        classes.append(corpus['sentiment'][idx])

words = [steamer.lemmatize(word) for word in words if word not in stop_words]
words = sorted(set(words))

classes = sorted(set(classes))
print(classes)

pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

######################################################
# Data preprocessing, Prepare Training Data
######################################################

# Concept below is to represent all words as are numerical values

# "Bag of Words implementation".
# Sets individual words indices/values to either 0 or 1 depending if it's occurring in the particular pattern.
training = []
# template of zero's
output_empty = [0] * len(classes)
# Convert the document data to the training list in order to train the neural network
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [steamer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Creates a simple sequential model
model = Sequential()
# Creates the input layer (Dense layer with 128 neurones and
# an input shape that is dependent on the size of the training data)
# Activation function to be a rectified linear unit (relu)
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
# Prevents over fitting
model.add(Dropout(0.5))
# Creates the input layer (Dense layer with 64 neurones)
# Activation function to be a rectified linear unit (relu)
model.add(Dense(64, activation='relu'))
# Prevents over fitting
model.add(Dropout(0.5))
# Creates the input layer (Dense layer with the many neurons as classes)
# Activation function is to be a softmax function
model.add(Dense(len(train_y[0]), activation='softmax'))
# Stochastic gradient descent optimizer.
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model compiling
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Feeding the model with prepared data, 200 times with the medium amount of information(verbose = 1)
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('../model/nn/chatbotmodel.h5', hist)
print("Done")