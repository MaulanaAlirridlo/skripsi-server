from preprocessing import caseFolding, dataCleaning, stopwordRemoval, stemmer, preprocessing
from translate import translate
from tfIdf import tfIdfProcess
from lexiconBasedFeatures import lbfProcess
from bagOfWords import bowProcess
from ensembleFeatures import efProcess
from makeArrayFlat import flatArray
from supportVectorMachine import svmProcess
from KFold import kFoldProcess
from timeCount.a import cpuTimeCountA, allTimeCountA
from timeCount.c import cpuTimeCountC, allTimeCountC

from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

random = 42
k = 3

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
engTweet = []
for i, s in enumerate(cleanTweet):
    engCleanTweet.append(translate(s))
    engTweet.append(translate(df['tweet'][i]))

print('lbf')
neg = []
pos = []
neu = []
for s in engCleanTweet:
    lbf = lbfProcess(s)
    neg.append(lbf['neg'])
    pos.append(lbf['pos'])
    neu.append(lbf['neu'])

print('ef')
ef = efProcess(df['tweet'], engTweet)

print('mengolah data')
# tf-idf + lexicon
a = [tfIdf, neg, pos, neu]
a = np.array(a, dtype=object)
a = a.T
newA = pd.DataFrame(flatArray(a))

# tf-idf + ensemble features
c = [tfIdf, ef]
c = np.array(c, dtype=object)
c = c.T
newC = pd.DataFrame(flatArray(c))

clf = svm.SVC(kernel='rbf')
scoresA = cross_val_score(clf, newA, df['label'], cv=k)
scoresC = cross_val_score(clf, newC, df['label'], cv=k)
mean_scoreA = scoresA.mean()
mean_scoreC = scoresC.mean()

print('mulai waktu A')
cpuTimeA = cpuTimeCountA(df, engCleanTweet, df['label'], random)
allTimeA = allTimeCountA(df, engCleanTweet, df['label'], random)

print('mulai waktu C')
cpuTimeC = cpuTimeCountC(df, df['label'], random, engTweet)
allTimeC = allTimeCountC(df, df['label'], random, engTweet)

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
        'ef': ef,
        'newA': newA.values.tolist(),
        'newC': newC.values.tolist(),
        'scoresA': scoresA.tolist(),
        'scoresC': scoresC.tolist(),
        'mean_scoreA': mean_scoreA,
        'mean_scoreC': mean_scoreC,
        'cpuTimeA': cpuTimeA,
        'allTimeA': allTimeA,
        'cpuTimeC': cpuTimeC,
        'allTimeC': allTimeC,
    })

if __name__ == '__main__':
    app.run()
