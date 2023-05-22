from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def lbfProcess(engTweetString):
  analyzer = SentimentIntensityAnalyzer()
  return analyzer.polarity_scores(engTweetString)