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

def cpuTimeCountA(df, engCleanTweet, label, random):
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
  
  neg = []
  pos = []
  neu = []
  for s in engCleanTweet:
    lbf = lbfProcess(s)
    neg.append(lbf['neg'])
    pos.append(lbf['pos'])
    neu.append(lbf['neu'])

  # tf-idf + lexicon
  a = [tfIdf, neg, pos, neu]
  a = np.array(a, dtype=object)
  a = a.T
  newA = flatArray(a)
  xTrainA, xTestA, yTrainA, yTestA = train_test_split(newA, label, test_size=0.3, random_state=random)
  tweetTrainA = [cleanTweet[i] for i in np.where(np.isin(newA, xTrainA).all(1))[0]]
  tweetTestA = [cleanTweet[i] for i in np.where(np.isin(newA, xTestA).all(1))[0]]

  svmA = svmProcess(xTrainA, yTrainA, xTestA)
  end_time = time.process_time()
  return end_time - start_time

def allTimeCountA(df, engCleanTweet, label, random):
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

  neg = []
  pos = []
  neu = []
  for s in engCleanTweet:
    lbf = lbfProcess(s)
    neg.append(lbf['neg'])
    pos.append(lbf['pos'])
    neu.append(lbf['neu'])

  # tf-idf + lexicon
  a = [tfIdf, neg, pos, neu]
  a = np.array(a, dtype=object)
  a = a.T
  newA = flatArray(a)
  xTrainA, xTestA, yTrainA, yTestA = train_test_split(newA, label, test_size=0.3, random_state=random)
  tweetTrainA = [cleanTweet[i] for i in np.where(np.isin(newA, xTrainA).all(1))[0]]
  tweetTestA = [cleanTweet[i] for i in np.where(np.isin(newA, xTestA).all(1))[0]]

  svmA = svmProcess(xTrainA, yTrainA, xTestA)
  end_time = time.time()
  return end_time - start_time