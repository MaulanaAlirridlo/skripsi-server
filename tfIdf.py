from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def tfIdfProcess(tweetList):
  tfIdfVectorizer=TfidfVectorizer(use_idf=True)
  tfIdf = tfIdfVectorizer.fit_transform(tweetList).todense()
  return np.array(tfIdf, dtype=object).tolist()