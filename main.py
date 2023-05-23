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

random = 293

# baca data
print('baca data')
df = pd.read_csv('./data/cek.csv')

print('preprocessing')
cf = caseFolding(df['tweet'])
dc = dataCleaning(cf)
sr = stopwordRemoval(dc)
stem = stemmer(sr)
preprocessingData = pd.DataFrame({
    'caseFolding': cf,
    'dataCleaning': dc,
    'stopwordRemoval': sr,
    'stemming': stem
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

print('mengolah data')
# tf-idf + lexicon
a = [tfIdf, neg, pos, neu]
a = np.array(a, dtype=object)
a = a.T
newA = pd.DataFrame(flatArray(a))
xTrainA, xTestA, yTrainA, yTestA = train_test_split(
    newA, df['label'], test_size=0.3, random_state=random)
tweetTrainA = [cleanTweet[i] for i in xTrainA.index]
tweetTestA = [cleanTweet[i] for i in xTestA.index]

# tf-idf + bag of word
b = [tfIdf, bow]
b = np.array(b, dtype=object)
b = np.transpose(b, (1, 0, 2))
newB = pd.DataFrame(flatArray(b))
xTrainB, xTestB, yTrainB, yTestB = train_test_split(
    newB, df['label'], test_size=0.3, random_state=random)
tweetTrainB = [cleanTweet[i] for i in xTrainB.index]
tweetTestB = [cleanTweet[i] for i in xTestB.index]

# tf-idf + ensemble features
c = [tfIdf, ef]
c = np.array(c, dtype=object)
c = c.T
newC = pd.DataFrame(flatArray(c))
xTrainC, xTestC, yTrainC, yTestC = train_test_split(
    newC, df['label'], test_size=0.3, random_state=random)
tweetTrainC = [cleanTweet[i] for i in xTrainC.index]
tweetTestC = [cleanTweet[i] for i in xTestC.index]

print('svm')
svmA = svmProcess(xTrainA, yTrainA, xTestA)
svmB = svmProcess(xTrainB, yTrainB, xTestB)
svmC = svmProcess(xTrainC, yTrainC, xTestC)

print('membuat matrix')
matrixA = confusion_matrix(yTestA, svmA)
accuracyA = accuracy_score(yTestA, svmA)
precisionA = precision_score(yTestA, svmA, average='macro')
recallA = recall_score(yTestA, svmA, average='macro')
f1A = f1_score(yTestA, svmA, average='macro')

matrixB = confusion_matrix(yTestB, svmB)
accuracyB = accuracy_score(yTestB, svmB)
precisionB = precision_score(yTestB, svmB, average='macro')
recallB = recall_score(yTestB, svmB, average='macro')
f1B = f1_score(yTestB, svmB, average='macro')

matrixC = confusion_matrix(yTestC, svmC)
accuracyC = accuracy_score(yTestC, svmC)
precisionC = precision_score(yTestC, svmC, average='macro')
recallC = recall_score(yTestC, svmC, average='macro')
f1C = f1_score(yTestC, svmC, average='macro')


# print(matrixA)
# print(accuracyA)
# print(precisionA)
# print(recallA)
# print(f1A)

# print(svmA)
# print(yTestA)
# print(f"akurasi A {np.mean(svmA == yTestA)}")
# print(svmB)
# print(yTestB)
# print(f"akurasi B {np.mean(svmB == yTestB)}")
# print(svmC)
# print(yTestC)
# print(f"akurasi C {np.mean(svmC == yTestC)}")

print('mulai waktu A')
cpuTimeA = cpuTimeCountA(df, engCleanTweet, df['label'], random)
allTimeA = allTimeCountA(df, engCleanTweet, df['label'], random)
print('mulai waktu B')
cpuTimeB = cpuTimeCountB(df, df['label'], random)
allTimeB = allTimeCountB(df, df['label'], random)
print('mulai waktu C')
cpuTimeC = cpuTimeCountC(df, df['label'], random)
allTimeC = allTimeCountC(df, df['label'], random)

# API API API API API
app = Flask(__name__)
CORS(app)
@app.route('/', methods=['GET'])
def getAll():
    return jsonify({
        'status': 200,
        'initData': df.to_dict(orient='records'),
        'preprocessing': preprocessingData.to_dict(orient='records'),
        'cleanTweet': cleanTweet.tolist(),
        'tfIdf': tfIdf,
        'engCleanTweet': engCleanTweet,
        'neg': neg,
        'pos': pos,
        'neu': neu,
        'bow': bow,
        'ef': ef,
        'newA': newA.values.tolist(),
        'tweetTrainA': tweetTrainA,
        'tweetTestA': tweetTestA,
        'tweetTrainB': tweetTrainB,
        'tweetTestB': tweetTestB,
        'tweetTrainC': tweetTrainC,
        'tweetTestC': tweetTestC,
        'xTrainA': xTrainA.values.tolist(),
        'xTestA': xTestA.values.tolist(),
        'yTrainA': yTrainA.tolist(),
        'yTestA': yTestA.tolist(),
        'svmA': svmA.tolist(),
        'matrixA': matrixA.tolist(),
        'accuracyA': accuracyA.tolist(),
        'precisionA': precisionA.tolist(),
        'recallA': recallA.tolist(),
        'f1A': f1A.tolist(),
        'newB': newB.values.tolist(),
        'xTrainB': xTrainB.values.tolist(),
        'xTestB': xTestB.values.tolist(),
        'yTrainB': yTrainB.tolist(),
        'yTestB': yTestB.tolist(),
        'svmB': svmB.tolist(),
        'matrixB': matrixB.tolist(),
        'accuracyB': accuracyB.tolist(),
        'precisionB': precisionB.tolist(),
        'recallB': recallB.tolist(),
        'f1B': f1B.tolist(),
        'newC': newC.values.tolist(),
        'xTrainC': xTrainC.values.tolist(),
        'xTestC': xTestC.values.tolist(),
        'yTrainC': yTrainC.tolist(),
        'yTestC': yTestC.tolist(),
        'svmC': svmC.tolist(),
        'matrixC': matrixC.tolist(),
        'accuracyC': accuracyC.tolist(),
        'precisionC': precisionC.tolist(),
        'recallC': recallC.tolist(),
        'f1C': f1C.tolist(),
        'cpuTimeA': cpuTimeA,
        'allTimeA': allTimeA,
        'cpuTimeB': cpuTimeB,
        'allTimeB': allTimeB,
        'cpuTimeC': cpuTimeC,
        'allTimeC': allTimeC,
    })

if __name__ == '__main__':
    app.run()
