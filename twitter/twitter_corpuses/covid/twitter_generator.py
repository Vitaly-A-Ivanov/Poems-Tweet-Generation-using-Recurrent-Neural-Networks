import random
import pickle
import numpy
import tensorflow

from tensorflow.keras.models import load_model


text = open('covid.txt', 'rb').read().decode(encoding='utf-8').lower()
chars = pickle.load(open('covid.pkl', 'rb'))

model = load_model('model_covid.h5')

charsToIndex = dict((c, i) for i, c in enumerate(chars))
indexToChars = dict((i, c) for i, c in enumerate(chars))

sequenceLength = 40
stepLength = 3

def sample(preds, temperature=1.0):
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)


def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - sequenceLength - 1)
    generated = ''

    sentence = text[start_index: start_index + sequenceLength]
    generated += sentence
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

print('-------0.2-----------')
print(generate_text(300,0.2))
print('-------0.8-----------')
print(generate_text(300,0.8))

