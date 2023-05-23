import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# preprocessing
def dataCleaningProcess(tweet):
    tweet = re.sub('@[^\s]+', '', tweet)
    tweet = re.sub('http[^\s]+', '', tweet)
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)
    return tweet

def caseFolding(tweetList):
    return tweetList.str.casefold()

def dataCleaning(tweetList):
    return tweetList.apply(dataCleaningProcess)

def stopwordRemoval(tweetList):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return tweetList.apply(stopword.remove)

def stemmer(tweetList):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return tweetList.apply(stemmer.stem)

def preprocessing(tweetList):
    return stemmer(stopwordRemoval(dataCleaning(caseFolding(tweetList))))