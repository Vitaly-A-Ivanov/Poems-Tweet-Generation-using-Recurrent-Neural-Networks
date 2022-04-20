import pickle
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()
corpus = pd.read_csv('../corpus/text_emotion_clean.csv')

words = pickle.load(open('../models/nn/words.pkl', 'rb'))
classes = pickle.load(open('../models/nn/classes.pkl', 'rb'))
model = load_model('../models/nn/chatbotmodel.h5')

# function for cleaning up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# function for getting the bag of words.
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# function for predicting the class based on the sentences (result based on a bag of words)
def predict_class(sentence):
    bow = bag_of_words(sentence)
    respond = model.predict(np.array([bow]))[0]
    # threshold to avoid too much uncertainty
    ERROR_THRESHOLD = 0.25
    # enumerates all the results
    results = [[i, r] for i, r in enumerate(respond) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'sentiment': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(sentiment_list, dataset):
    result = ''
    tag = sentiment_list[0]['sentiment']
    list_of_intents = dataset['sentiment']
    for i in list_of_intents:
        if i == tag:
            result = i
            break
    return result

print('Bot is running')

while True:
    message = input('')
    ints = predict_class(message)
    res = get_response(ints, corpus)
    print(res)