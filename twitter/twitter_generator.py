import random
import pickle
import numpy

from tensorflow.keras.models import load_model

# read file with clean data
text = open('twitter_corpuses/annajordanous.txt', 'rb').read().decode(encoding='utf-8').lower()

#  read file with characters from the text
chars = pickle.load(open('twitter_models/annajordanous/chars.pkl', 'rb'))

#  load model
model = load_model('twitter_models/annajordanous/model_twitter_anna.h5')

# create two dictionaries and map each character to its index and other way round to represent them as a numerical
charsToIndex = dict((c, i) for i, c in enumerate(chars))
indexToChars = dict((i, c) for i, c in enumerate(chars))

# length of the sentence
sequenceLength = 40
#  number of characters as step to start the next sentence
stepLength = 3

#  helper function to predict the next character
def sample(preds, temperature=1.0):
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)

# function to generate the text
def generate_text(length, temperature):
    # random start index for the initial sentence from the text
    start_index = random.randint(0, len(text) - sequenceLength - 1)

    #  generated text
    generated = ''

    # initial sentence from the text using the random start index
    sentence = text[start_index: start_index + sequenceLength]
    generated += sentence

    # sentence creation
    for i in range(length):
        x_predictions = numpy.zeros((1, sequenceLength, len(chars)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, charsToIndex[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                            temperature)
        next_character = indexToChars[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

#  results
print('-------0.2-----------')
print(generate_text(300, 0.2))
print('-------0.8-----------')
print(generate_text(300, 0.8))
