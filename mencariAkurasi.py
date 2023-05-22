from preprocessing import caseFolding, dataCleaning, stopwordRemoval, stemmer, preprocessing
from translate import translate
from tfIdf import tfIdfProcess
from lexiconBasedFeatures import lbfProcess
from bagOfWords import bowProcess
from ensembleFeatures import efProcess
from makeArrayFlat import flatArray
from supportVectorMachine import svmProcess
from timeCount.a import cpuTimeCountA, allTimeCountA
from timeCount.b import cpuTimeCountB, allTimeCountB
from timeCount.c import cpuTimeCountC, allTimeCountC

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

#baca data
print('baca data')
df = pd.read_csv('./data/cek.csv')

print('preprocessing')
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

print('tf-idf')
tfIdf = tfIdfProcess(cleanTweet)

print('translate')
engCleanTweet = []
for s in cleanTweet:
  engCleanTweet.append(translate(s))

print('lbf')
neg = []
pos = []
neu = []
for s in engCleanTweet:
  lbf = lbfProcess(s)
  neg.append(lbf['neg'])
  pos.append(lbf['pos'])
  neu.append(lbf['neu'])

print('bow')
bow = bowProcess(cleanTweet)

print('ef')
ef = efProcess(df['tweet'])

print('mencari data yang pas')
arrAccuracyA = []
arrAccuracyB = []
arrAccuracyC = []
for j in range(1000):
  # tf-idf + lexicon
  a = [tfIdf, neg, pos, neu]
  a = np.array(a, dtype=object)
  a = a.T
  newA = flatArray(a)
  xTrainA, xTestA, yTrainA, yTestA = train_test_split(newA, df['label'], test_size=0.3, random_state=j)
  tweetTrainA = [cleanTweet[i] for i in np.where(np.isin(newA, xTrainA).all(1))[0]]
  tweetTestA = [cleanTweet[i] for i in np.where(np.isin(newA, xTestA).all(1))[0]]

  # tf-idf + bag of word
  b = [tfIdf, bow]
  b = np.array(b, dtype=object)
  b = np.transpose(b, (1, 0, 2))
  newB = flatArray(b)
  xTrainB, xTestB, yTrainB, yTestB = train_test_split(newB, df['label'], test_size=0.3, random_state=j)
  tweetTrainB = [cleanTweet[i] for i in np.where(np.isin(newB, xTrainB).all(1))[0]]
  tweetTestB = [cleanTweet[i] for i in np.where(np.isin(newB, xTestB).all(1))[0]]

  # tf-idf + ensemble features
  c = [tfIdf, ef]
  c = np.array(c, dtype=object)
  c = c.T
  newC = flatArray(c)
  xTrainC, xTestC, yTrainC, yTestC = train_test_split(newC, df['label'], test_size=0.3, random_state=j)
  tweetTrainC = [cleanTweet[i] for i in np.where(np.isin(newC, xTrainC).all(1))[0]]
  tweetTestC = [cleanTweet[i] for i in np.where(np.isin(newC, xTestC).all(1))[0]]

  svmA = svmProcess(xTrainA, yTrainA, xTestA)
  svmB = svmProcess(xTrainB, yTrainB, xTestB)
  svmC = svmProcess(xTrainC, yTrainC, xTestC)

  # matrixA = confusion_matrix(yTestA, svmA)
  accuracyA = accuracy_score(yTestA, svmA)
  arrAccuracyA.append(accuracyA)
  # precisionA = precision_score(yTestA, svmA, average='macro')
  # recallA = recall_score(yTestA, svmA, average='macro')
  # f1A = f1_score(yTestA, svmA, average='macro')
  print(f"lexicon random state {j}, akurasi = {accuracyA}")

  # matrixB = confusion_matrix(yTestB, svmB)
  accuracyB = accuracy_score(yTestB, svmB)
  arrAccuracyB.append(accuracyB)
  # precisionB = precision_score(yTestB, svmB, average='macro')
  # recallB = recall_score(yTestB, svmB, average='macro')
  # f1B = f1_score(yTestB, svmB, average='macro')
  print(f"bow random state {j}, akurasi = {accuracyB}")

  # matrixC = confusion_matrix(yTestC, svmC)
  accuracyC = accuracy_score(yTestC, svmC)
  arrAccuracyC.append(accuracyC)
  # precisionC = precision_score(yTestC, svmC, average='macro')
  # recallC = recall_score(yTestC, svmC, average='macro')
  # f1C = f1_score(yTestC, svmC, average='macro')
  print(f"ef random state {j}, akurasi = {accuracyC}")

data = pd.DataFrame([arrAccuracyA, arrAccuracyB, arrAccuracyC])
data.to_excel('data.xlsx', index=False)