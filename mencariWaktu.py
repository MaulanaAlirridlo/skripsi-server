from preprocessing import caseFolding, dataCleaning, stopwordRemoval, stemmer, preprocessing
from translate import translate
from timeCount.a import cpuTimeCountA, allTimeCountA
from timeCount.b import cpuTimeCountB, allTimeCountB
from timeCount.c import cpuTimeCountC, allTimeCountC
import pandas as pd


df = pd.read_csv('./data/cek.csv')

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

engCleanTweet = []
for s in cleanTweet:
  engCleanTweet.append(translate(s))

cpuA = []
allA = []
cpuB = []
allB = []
cpuC = []
allC = []
for i in range(334,339):
  try:
    cpuTimeA = cpuTimeCountA(df, engCleanTweet, df['label'],i)
    cpuA.append(cpuTimeA)
    allTimeA = allTimeCountA(df, engCleanTweet, df['label'],i)
    allA.append(allTimeA)
    print(f"{i} cpuA: {cpuTimeA} allA: {allTimeA}")
    cpuTimeB = cpuTimeCountB(df, df['label'],i)
    cpuB.append(cpuTimeB)
    allTimeB = allTimeCountB(df, df['label'],i)
    allB.append(allTimeB)
    print(f"{i} cpuB: {cpuTimeB} allB: {allTimeB}")
    cpuTimeC = cpuTimeCountC(df, df['label'],i)
    cpuC.append(cpuTimeC)
    allTimeC = allTimeCountC(df, df['label'],i)
    allC.append(allTimeC)
    print(f"{i} cpuC: {cpuTimeC} allC: {allTimeC}")
  except:
    data = pd.DataFrame([cpuA,allA,cpuB,allB,cpuC,allC])
    data.to_excel('data waktu.xlsx', index=False)
    break
