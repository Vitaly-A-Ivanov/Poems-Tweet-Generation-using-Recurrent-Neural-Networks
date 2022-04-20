import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn import preprocessing

corpus = pd.read_csv('corpus/text_emotion.csv')


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


# TODO get rid of twitter names occurrences from a text
for i, text in enumerate(corpus.content):
    if text.find('@') == -1:
        continue
    n = text.count('@')
    for j in range(n):
        if text.find('@') == -1:
            break
        sub_string = ''
        start_position = text.index('@')
        end_position = text.find(' ', start_position, )
        if end_position == -1:
            text = text[:start_position]
            corpus.at[i, 'content'] = text
            break
        for c in text[start_position: end_position]:
            sub_string += c
        text = text.replace(sub_string, '')
        corpus.at[i, 'content'] = text

# TODO get rid of links occurrences from a text
corpus['content'] = corpus['content'].str.replace(r'\s*https?://\S+(\s+|$)', ' ', regex=True).str.strip()

# TODO get rid rows  with empty content
for i, text in enumerate(corpus['content']):
    if text == '':
        corpus.drop([i], inplace=True)

# reset indexing
corpus.reset_index(inplace=True)

# TODO get rid rows  with empty santiment
for i, text in enumerate(corpus['sentiment']):
    if text == 'empty':
        corpus.drop([i], inplace=True)

# reset indexing
corpus.reset_index(inplace=True)

y = corpus.sentiment
x = corpus.content

emotion_encoding = preprocessing.LabelEncoder()
Y = emotion_encoding.fit_transform(y)

tf_vector = get_feature_vector(np.array(x).ravel())
X = tf_vector.transform(np.array(x).ravel())

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf = SVC(kernel='linear', C=3)
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

# save the model to disk
filename = 'models/svm_model.sav'
joblib.dump(clf, filename)
