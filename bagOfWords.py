from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def bowProcess(tweetList):
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(tweetList).todense()
    return np.array(bow, dtype=object).tolist()