from preprocessing import caseFolding, dataCleaning, stopwordRemoval, stemmer, preprocessing
from translate import translate
from tfIdf import tfIdfProcess
from lexiconBasedFeatures import lbfProcess
from bagOfWords import bowProcess
from ensembleFeatures import efProcess
from makeArrayFlat import flatArray
from supportVectorMachine import svmProcess

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import time

def cpuTimeCountC(df, label, random):
  start_time = time.process_time()
  cf = caseFolding(df['tweet'])
  dc = dataCleaning(cf)
  sr = stopwordRemoval(dc)
  stem = stemmer(sr)
  preprocessingData = pd.DataFrame({
                        'caseFolding' : cf,
                        'dataCleaning' : dc,
                        'stopwordRemoval' : sr,
                        'stemming' : stem
                      })
  cleanTweet = stem

  tfIdf = tfIdfProcess(cleanTweet)

  ef = efProcess(df['tweet'])

  # tf-idf + ensemble features
  c = [tfIdf, ef]
  c = np.array(c, dtype=object)
  c = c.T
  newC = flatArray(c)
  xTrainC, xTestC, yTrainC, yTestC = train_test_split(newC, label, test_size=0.3, random_state=random)
  tweetTrainC = [cleanTweet[i] for i in np.where(np.isin(newC, xTrainC).all(1))[0]]
  tweetTestC = [cleanTweet[i] for i in np.where(np.isin(newC, xTestC).all(1))[0]]

  svmC = svmProcess(xTrainC, yTrainC, xTestC)
  end_time = time.process_time()
  return end_time - start_time

def allTimeCountC(df, label, random):
  start_time = time.time()
  cf = caseFolding(df['tweet'])
  dc = dataCleaning(cf)
  sr = stopwordRemoval(dc)
  stem = stemmer(sr)
  preprocessingData = pd.DataFrame({
                        'caseFolding' : cf,
                        'dataCleaning' : dc,
                        'stopwordRemoval' : sr,
                        'stemming' : stem
                      })
  cleanTweet = stem

  tfIdf = tfIdfProcess(cleanTweet)

  ef = efProcess(df['tweet'])

  # tf-idf + ensemble features
  c = [tfIdf, ef]
  c = np.array(c, dtype=object)
  c = c.T
  newC = flatArray(c)
  xTrainC, xTestC, yTrainC, yTestC = train_test_split(newC, label, test_size=0.3, random_state=random)
  tweetTrainC = [cleanTweet[i] for i in np.where(np.isin(newC, xTrainC).all(1))[0]]
  tweetTestC = [cleanTweet[i] for i in np.where(np.isin(newC, xTestC).all(1))[0]]

  svmC = svmProcess(xTrainC, yTrainC, xTestC)
  end_time = time.time()
  return end_time - start_time