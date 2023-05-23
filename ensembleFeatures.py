from translate import translate
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tag import CRFTagger
import pycrfsuite
import re
import spacy
import emoji

analyzer = SentimentIntensityAnalyzer()

def F1(doc):
    return 1 if re.findall(r"#\w+", doc) else 0

def F2(doc):
    return 1 if doc.startswith("RT") else 0

def F3(doc):
    return 1 if re.findall(re.compile('@[^\s]'), doc) else 0

def F4(doc):
    return 1 if re.findall(re.compile('http[^\s]'), doc) else 0

def F5(doc):
    return len(doc.split())

def F6(doc):
    return len(doc)/len(doc.split())

def F7(doc):
    return doc.count("?")

def F8(doc):
    return doc.count("!")

def F9(doc):
    return len(re.findall('"([^"]*)"', doc))

def F10(doc):
    return len(re.findall(r'\b[A-Z]\w*\b', doc))

def F11(doc):
    has_emoji = False
    for character in doc:
        if len(list(set(emoji.distinct_emoji_list(doc)))) > 0:
            has_emoji = True
            break
    if has_emoji:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(doc)
        if sentiment_scores['compound'] > 0:
            return 1
        else:
            return 0
    else:
        return 0

def F12(doc):
    has_emoji = False
    for character in doc:
        if len(list(set(emoji.distinct_emoji_list(doc)))) > 0:
            has_emoji = True
            break
    if has_emoji:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(doc)
        if sentiment_scores['compound'] < 0:
            return 1
        else:
            return 0
    else:
        return 0

ct = CRFTagger()
ct.set_model_file('./data/all_indo_man_tag_corpus_model.crf.tagger')

def F13(doc):
    pos_tag = ct.tag_sents([doc.split()])
    return len([word for word, tag in pos_tag[0] if tag.startswith('N')])

def F14(doc):
    pos_tag = ct.tag_sents([doc.split()])
    return len([word for word, tag in pos_tag[0] if tag.startswith('J')])

def F15(doc):
    pos_tag = ct.tag_sents([doc.split()])
    return len([word for word, tag in pos_tag[0] if tag.startswith('V')])

def F16(doc):
    pos_tag = ct.tag_sents([doc.split()])
    return len([word for word, tag in pos_tag[0] if tag.startswith('RB')])

def F17(doc):
    pos_tag = ct.tag_sents([doc.split()])
    return len([word for word, tag in pos_tag[0] if tag == 'UH'])

def F18(doc):
    return (F13(doc)/len(doc.split()))*100

def F19(doc):
    return (F14(doc)/len(doc.split()))*100

def F20(doc):
    return (F15(doc)/len(doc.split()))*100

def F21(doc):
    return (F16(doc)/len(doc.split()))*100

def F22(doc):
    return (F17(doc)/len(doc.split()))*100

# init spacy
sp = spacy.load('en_core_web_sm')

def F23(doc):
    positive_count = 0
    for word in doc.split():
        scores = analyzer.polarity_scores(word)
        if scores['pos'] > scores['neg']:
            positive_count += 1
    return positive_count

def F24(doc):
    negative_count = 0
    for word in doc.split():
        scores = analyzer.polarity_scores(word)
        if scores['pos'] < scores['neg']:
            negative_count += 1
    return negative_count

def F25(doc):
    pos = sp(doc)
    count = 0
    for v in pos:
        if v.pos_ == 'ADJ':
            scores = analyzer.polarity_scores(v.text)
            if scores['pos'] > scores['neg']:
                count += 1
    return count

def F26(doc):
    pos = sp(doc)
    count = 0
    for v in pos:
        if v.pos_ == 'ADJ':
            scores = analyzer.polarity_scores(v.text)
            if scores['pos'] < scores['neg']:
                count += 1
    return count

def F27(doc):
    pos = sp(doc)
    count = 0
    for v in pos:
        if v.pos_ == 'VERB':
            scores = analyzer.polarity_scores(v.text)
            if scores['pos'] > scores['neg']:
                count += 1
    return count

def F28(doc):
    pos = sp(doc)
    count = 0
    for v in pos:
        if v.pos_ == 'VERB':
            scores = analyzer.polarity_scores(v.text)
            if scores['pos'] < scores['neg']:
                count += 1
    return count

def F29(doc):
    pos = sp(doc)
    count = 0
    for v in pos:
        if v.pos_ == 'ADV':
            scores = analyzer.polarity_scores(v.text)
            if scores['pos'] > scores['neg']:
                count += 1
    return count

def F30(doc):
    pos = sp(doc)
    count = 0
    for v in pos:
        if v.pos_ == 'ADV':
            scores = analyzer.polarity_scores(v.text)
            if scores['pos'] < scores['neg']:
                count += 1
    return count

def F31(doc):
    f23 = F23(doc)
    return (F25(doc)/f23)*100 if f23 != 0 else 0

def F32(doc):
    f24 = F24(doc)
    return (F26(doc)/f24)*100 if f24 != 0 else 0

def F33(doc):
    f23 = F23(doc)
    return (F27(doc)/f23)*100 if f23 != 0 else 0

def F34(doc):
    f24 = F24(doc)
    return (F28(doc)/f24)*100 if f24 != 0 else 0

def F35(doc):
    f23 = F23(doc)
    return (F29(doc)/f23)*100 if f23 != 0 else 0

def F36(doc):
    f24 = F24(doc)
    return (F28(doc)/f24)*100 if f24 != 0 else 0

def F37(doc):
    pos = sp(doc)
    count = 0
    for v in pos:
        if v.pos_ == 'ADV' and re.match(r'\b\w+\b', v.text):
            count += 1
    return count

def efProcess(tweetList):
    ensemble = []
    for doc in tweetList:
        enDoc = translate(doc)
        ensemble.append([F1(doc), F2(doc), F3(doc), F4(doc), F5(doc), F6(doc),
                         F7(doc), F8(doc), F9(doc), F10(
                             doc), F11(enDoc), F12(enDoc),
                         F13(doc), F14(doc), F15(doc), F16(
                             doc), F17(doc), F18(doc),
                         F19(doc), F20(doc), F21(doc), F22(
                             doc), F23(enDoc), F24(enDoc),
                         F25(enDoc), F26(enDoc), F27(enDoc), F28(
                             enDoc), F29(enDoc), F30(enDoc),
                         F31(doc), F32(doc), F33(doc), F34(
                             doc), F35(doc), F36(doc),
                         F37(enDoc)])
    return ensemble