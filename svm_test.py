import pickle

import joblib
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

clf = joblib.load('models/svm_model.sav')

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

x = ['hey']

tf_vector = get_feature_vector(np.array(x).ravel())
X = tf_vector.transform(np.array(x).ravel())

a = clf.predict(X)
print(a)