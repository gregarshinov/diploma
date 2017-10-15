import pandas as pd
from tqdm import tqdm
import string
from sklearn.model_selection import train_test_split
import numpy as np
from pymystem3 import Mystem

def read_files(sentences_fname,
               labels_fname,
               stopwords_fname):
    sentences = pd.read_csv(sentences_fname, sep='\n')
    labels = pd.read_csv(labels_fname, sep='\n')

    stop_words = pd.read_csv(stopwords_fname, sep='\n')
    stop_words.columns=['word']

    printout_class_proportions(labels)

    sentences.columns = ['sentences']
    labels.columns = ['labels']

    return sentences, labels, stop_words

def prepare_y_vector(labels):
    y = labels['labels'].replace(['positive',
                                  'neutral',
                                  'negative'], [1,0,-1])
    return y

def printout_class_proportions(labels):
    pos, neutral, neg = 0,0,0
    for l in labels.values:
        if l == 'positive':
            pos+=1
        elif l == 'negative':
            neg += 1
        elif l == 'neutral':
            neutral += 1
    print("Positive = {} / Neutral = {} / Negative = {}\n".format(pos, neutral, neg))

def normalization_russian_sentences(sentences_df, stopwords_df):
    myStemInstance = Mystem()
    normalized_sentences_list = []
    for c_sentence in tqdm(sentences_df['sentences']):
        tokens = myStemInstance.lemmatize(c_sentence)
        tokens_without_stopwords = list(filter(lambda token : (token not in stopwords_df['word'].values) and
                                                         token.isalpha(), tokens))
        normalized_sentences_list.append(' '.join(tokens_without_stopwords).strip("\n"))
    return normalized_sentences_list

def separation_by_train_and_test(sentences_df, answers_df, stopwords_df):
    sentences_list = normalization_russian_sentences(sentences_df, stopwords_df)
    y = prepare_y_vector(answers_df)

    X_train, X_test, y_train, y_test = train_test_split(sentences_list, y,
                                                        test_size=0.3, random_state=45)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    sentences_fname = 's.txt'
    labels_fname = 'polarity.csv'
    stopwords_fname = 'stop-words-russian.txt'

    sentences_df, labels_df, stopwords_df = read_files(sentences_fname, labels_fname, stopwords_fname)
    X_train, X_test, y_train, y_test = separation_by_train_and_test(sentences_df, labels_df, stopwords_df)

    f=open('x_train_rus.txt', 'w', encoding='utf-8')
    for x in X_train:
        f.write(x+ '\n')
    f.close()
    f=open('x_test_rus.txt', 'w', encoding='utf-8')
    for x in X_test:
        f.write(x+ '\n')
    f.close()
    f=open('y_train_rus.txt', 'w', encoding='utf-8')
    for yy in y_train:
        f.write(str(yy)+'\n')
    f.close()
    f=open('y_test_rus.txt', 'w', encoding='utf-8')
    for yy in y_test:
        f.write(str(yy)+'\n')
    f.close()