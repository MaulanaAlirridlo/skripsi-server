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

def cpuTimeCountB(df, label, random):
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
  
  bow = bowProcess(cleanTweet)

  # tf-idf + bag of word
  b = [tfIdf, bow]
  b = np.array(b, dtype=object)
  b = np.transpose(b, (1, 0, 2))
  newB = flatArray(b)
  xTrainB, xTestB, yTrainB, yTestB = train_test_split(newB, label, test_size=0.3, random_state=random)
  tweetTrainB = [cleanTweet[i] for i in np.where(np.isin(newB, xTrainB).all(1))[0]]
  tweetTestB = [cleanTweet[i] for i in np.where(np.isin(newB, xTestB).all(1))[0]]

  svmB = svmProcess(xTrainB, yTrainB, xTestB)
  end_time = time.process_time()
  return end_time - start_time

def allTimeCountB(df, label, random):
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

  bow = bowProcess(cleanTweet)

  # tf-idf + bag of word
  b = [tfIdf, bow]
  b = np.array(b, dtype=object)
  b = np.transpose(b, (1, 0, 2))
  newB = flatArray(b)
  xTrainB, xTestB, yTrainB, yTestB = train_test_split(newB, label, test_size=0.3, random_state=random)
  tweetTrainB = [cleanTweet[i] for i in np.where(np.isin(newB, xTrainB).all(1))[0]]
  tweetTestB = [cleanTweet[i] for i in np.where(np.isin(newB, xTestB).all(1))[0]]

  svmB = svmProcess(xTrainB, yTrainB, xTestB)
  end_time = time.time()
  return end_time - start_time