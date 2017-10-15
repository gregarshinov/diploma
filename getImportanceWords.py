from __future__ import print_function

from collections import defaultdict
import math
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Bidirectional, Dropout, GRU
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import math
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split


MAX_FEATURES = 20000
MAX_LEN = 80
BATCH_SIZE = 32

def loading_dataset(X_train_fname, y_train_fname,
                    X_test_fname, y_test_fname, count_of_classes):
    # Загрузка как csv в dataframe
    X_train = pd.read_csv(X_train_fname, sep='\n')
    X_train.columns = ['sentences']
    X_test = pd.read_csv(X_test_fname, sep='\n')
    X_test.columns = ['sentences']
    y_train = pd.read_csv(y_train_fname, sep='\n')
    y_train.columns = ['label']
    y_test = pd.read_csv(y_test_fname, sep='\n')
    y_test.columns = ['label']

    # Преобразование классов в категориальную форму
    y_train = to_categorical(y_train, num_classes=count_of_classes)
    y_test = to_categorical(y_test, num_classes=count_of_classes)

    return X_train, y_train, X_test, y_test

def linguistic_preprocession(X_train, X_test):
    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(X_train.sentences)
    X_train_tokenized = tokenizer.texts_to_sequences(X_train.sentences)
    X_test_tokenized = tokenizer.texts_to_sequences(X_test.sentences)

    print('Padding sentences (sample X time)')
    X_train_sequence = sequence.pad_sequences(X_train_tokenized)
    X_test_sequence = sequence.pad_sequences(X_test_tokenized)

    print('X_train shape: {}'.format(X_train_sequence.shape))
    print('X_test shape: {}'.format(X_test_sequence.shape))
    return X_train_sequence, X_test_sequence, tokenizer

def generate_LSTM_model(number_classes):

    model = Sequential()
    model.add(Embedding(MAX_FEATURES, 128))
    model.add(LSTM(32,
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   activation='tanh',
                   return_sequences=True))
    model.add(LSTM(64,
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   activation='tanh'))
    model.add(Dense(number_classes, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer = 'rmsprop',
                  metrics=['accuracy'])

    return model

def generate_GRU_mode(number_classes):
    model = Sequential()
    model.add(Embedding(MAX_FEATURES, 128))
    model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(number_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def generate_BiLSTM_model(number_classes):
    model = Sequential()
    model.add(Embedding(MAX_FEATURES, 128))
    # model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2, activation='tanh', return_sequences=True)))
    model.add(Bidirectional(LSTM(64, activation='tanh'), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Dense(number_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    return model

def train_and_test_model( neural_model,
        X_train_sequence, y_train,
        X_test_sequence, y_test):
    print('Training model:')
    neural_model.fit(X_train_sequence, y_train,
                     batch_size=BATCH_SIZE,
                     epochs=10,
                     validation_data=(X_test_sequence, y_test))
    score, accuracy = neural_model.evaluate(X_test_sequence, y_test,
                                            batch_size=BATCH_SIZE)
    print('\n')
    print('Test score: {0:7.6f} \n Test accuracy: {1:7.6f}'.format(score, accuracy))

def create_dictionary(tokenizer_instance):
    res_dictionary = tokenizer_instance.word_index

    w_list = []
    for key in sorted(res_dictionary,
                      key = res_dictionary.get,
                      reverse = False):
        w_list.append(key)
    print("Length of word dictionary : {}".format(len(w_list)))
    return w_list

def calculate_frequency_in_test_corpora(X_test, y_test,
                                        word_list):
    test = X_test
    test['label'] = y_test.tolist()
    frequency_counts = {}
    for word in word_list:
        frequency_tmp = (test[test['sentences'].str.contains(word)]).shape[0]
        #print('{} : {}'.format(word, frequency_tmp))
        if frequency_tmp > 3 and len(word) > 2:
            frequency_counts[word] = frequency_tmp
    print("Volume of efficient dictionary: {}".format(len(frequency_counts)))
    return test, frequency_counts

def get_score_accuracy( neural_model,
                        tokenizer_instance,
                        sub_sentences_list,
                        sub_y_list):
    tokenized_sub_sentences = tokenizer_instance.texts_to_sequences(sub_sentences_list)
    seq_sub_sentences = sequence.pad_sequences(tokenized_sub_sentences)
    score, accuracy = neural_model.evaluate(seq_sub_sentences, sub_y_list,
                                            batch_size=BATCH_SIZE)
    return score, accuracy, seq_sub_sentences

def test_score_accuracy_with_preprocessed(neural_model,
                                          sentences_list,
                                          answers_list):
    score, accuracy = neural_model.evaluate(sentences_list, answers_list,
                                            batch_size=BATCH_SIZE)
    return score, accuracy

def get_doesnt_affected_words(
        neural_model,
        tokenizer_instance,
        word_frequency_dictionary,
        X_test):
    result_list_of_words_doest_affected_to_nn_model = []
    words_affected_to_model = defaultdict(dict)

    for word in word_frequency_dictionary:
        #sentences_with_word = (X_test[X_test['sentences'].str.contains(word)])
        sentences_with_word = X_test
        X_with_word = sentences_with_word['sentences']
        y_with_word = sentences_with_word['label'].tolist()

        score_with_word, accuracy_with_word, seq_sub_sentences_with_word = get_score_accuracy(neural_model,
                                                                                              tokenizer_instance,
                                                                                              X_with_word, y_with_word)
        words_affected_to_model[word]['enabled'] = accuracy_with_word
        X_without_word = sentences_with_word['sentences'].str.replace(word, '')
        score_without_word, accuracy_without_word, seq_sub_sentences_without_word = get_score_accuracy(neural_model,
                                                                                                       tokenizer_instance,
                                                                                                       X_without_word,
                                                                                                       y_with_word)
        words_affected_to_model[word]['disabled'] = accuracy_without_word
        words_affected_to_model[word]['delta'] = accuracy_with_word - accuracy_without_word
        if math.isclose(accuracy_with_word, accuracy_without_word):
            print("\nDoesn\'t affect to model: {}".format(word))
            result_list_of_words_doest_affected_to_nn_model.append(word)
        else:
            print("\nWord: {0} affected to model with delta accuracy = {1:7.5f}".format(word,
                                                                                     math.fabs(accuracy_with_word-accuracy_without_word)))
    return words_affected_to_model, result_list_of_words_doest_affected_to_nn_model

def check_with_deleted_specific_words(
        neural_model,
        tokenizer_instance,
        X_test,
        target_words
):
    sentences_with_word = X_test
    X_with_word = sentences_with_word['sentences']
    y_with_word = sentences_with_word['label'].tolist()

    tokenized_sub_sentences_with_word = tokenizer_instance.texts_to_sequences(X_with_word)
    seq_sub_sentences_with_word = sequence.pad_sequences(tokenized_sub_sentences_with_word)
    _, accuracy_with_words = test_score_accuracy_with_preprocessed(neural_model,
                                                                                seq_sub_sentences_with_word,
                                                                                y_with_word)

    X_without_word = sentences_with_word['sentences']
    for target_deleted_word in target_words:
        X_without_word = X_without_word.str.replace(target_deleted_word, '')
    tokenized_sub_sentences_without_word = tokenizer_instance.texts_to_sequences(X_without_word)
    seq_sub_sentences_without_word = sequence.pad_sequences(tokenized_sub_sentences_without_word)
    _, accuracy_without_words = test_score_accuracy_with_preprocessed(  neural_model,
                                                                                        seq_sub_sentences_without_word,
                                                                                        y_with_word)
    print('Accuracy with word list: {:7.5f}'.format(accuracy_with_words))
    print('Accuracy without word list: {:7.5f}'.format(accuracy_without_words))

    return accuracy_with_words, accuracy_without_words

def get_probabilities_gold(probability_model, y):
    probabilities_gold = []
    for x,y in zip(probability_model, y):
        proba = (max(x*y))
        probabilities_gold.append(proba)
    return probabilities_gold

def analyze_importance_words(neural_model,
                             tokenizer_instance,
                             word_frequency_dictionary,
                             X_test):
    dictionary_of_importante_words = {}
    for word in word_frequency_dictionary:
        #sentences_with_word = X_test[X_test['sentences'].str.contains(word)]
        sentences_with_word = X_test
        X_with_word = sentences_with_word['sentences']
        y_with_word = sentences_with_word['label'].tolist()


        tokenized_sub_sentences_with_word = tokenizer_instance.texts_to_sequences(X_with_word)
        seq_sub_sentences_with_word = sequence.pad_sequences(tokenized_sub_sentences_with_word)

        classes_with_word = neural_model.predict_classes(seq_sub_sentences_with_word)
        probability_with_word = neural_model.predict_proba(seq_sub_sentences_with_word)
        probabilities_gold_with_word = get_probabilities_gold(probability_with_word, y_with_word)

        X_without_word = sentences_with_word['sentences'].str.replace(word, '')
        tokenized_sub_sentences_without_word = tokenizer_instance.texts_to_sequences(X_without_word)
        seq_sub_sentences_without_word = sequence.pad_sequences(tokenized_sub_sentences_without_word)

        classes_without_word = neural_model.predict_classes(seq_sub_sentences_without_word)
        probability_without_word = neural_model.predict_proba(seq_sub_sentences_without_word)
        probabilities_gold_without_word = get_probabilities_gold(probability_without_word, y_with_word)

        importance_summ = 0

        for proba_gold_with_word, proba_gold_without_word in zip(probabilities_gold_with_word, probabilities_gold_without_word):
            #print("Probabilities gold with word: {} \n Probabilities gold without word: {}".format(proba_gold_with_word,
            #                                                                                       proba_gold_without_word))
            importance_summ += 1 - (math.log(proba_gold_without_word)/math.log(proba_gold_with_word))
        importance_of_word = importance_summ / len(y_with_word)
        dictionary_of_importante_words[word] = importance_of_word
        #print('Importances of word {} equal : {}'.format(word, importance_of_word))
    return dictionary_of_importante_words


def main():
#    file_name_x_train = 'x_train_e.txt'
#    file_name_x_test = 'x_test_e.txt'
#    file_name_y_train = 'y_train_e.txt'
#    file_name_y_test = 'y_test_e.txt'

    file_name_x_train = 'x_train_rus.txt'
    file_name_x_test = 'x_test_rus.txt'
    file_name_y_train = 'y_train_rus.txt'
    file_name_y_test = 'y_test_rus.txt'


    #neural_model = generate_GRU_mode(number_classes=3)
    neural_model = generate_LSTM_model(number_classes=3)

    X_train, y_train, X_test, y_test = loading_dataset(X_train_fname = file_name_x_train, y_train_fname = file_name_y_train,
                    X_test_fname = file_name_x_test, y_test_fname = file_name_y_test, count_of_classes=3)
    X_train_sequence, X_test_sequence, tokenizer_instance = linguistic_preprocession(X_train, X_test)
    train_and_test_model(neural_model, X_train_sequence, y_train,
                         X_test_sequence, y_test)
    word_list = create_dictionary(tokenizer_instance)
    test_corpora, word_frequency_dictionary = calculate_frequency_in_test_corpora(X_test, y_test,
                                                           word_list)

    words_affected_to_model, result_list_of_words_doest_affected_to_nn_model = get_doesnt_affected_words(
        neural_model,
        tokenizer_instance,
        word_frequency_dictionary,
        test_corpora)

    listOfAffectedWords = []
    for affected_word in words_affected_to_model:
        listOfAffectedWords.append((affected_word, words_affected_to_model[affected_word]['delta']))

    file_affected_words = open('LSTM_MOD_Affected_words_RUS.txt', 'w', encoding='utf-8')
    # for affected_word in words_affected_to_model:
    #     line = '{} : {}\n'.format(affected_word,
    #                             words_affected_to_model[affected_word]['enabled'] - words_affected_to_model[affected_word]['disabled'])
    #     file_affected_words.write(line)
    for (affected_word, delta_of_accuracy) in sorted(listOfAffectedWords, key = lambda x : x[1], reverse = True):
        line = '{} : {:6.5f} (Before: {:6.5f}, After: {:6.5f})\n'.format(affected_word, delta_of_accuracy,
                                                                         words_affected_to_model[affected_word]['enabled'],
                                                                         words_affected_to_model[affected_word]['disabled'])
        file_affected_words.write(line)
    file_affected_words.close()

    dictionary_of_importante_words = analyze_importance_words(neural_model,
                             tokenizer_instance,
                             word_frequency_dictionary,
                             test_corpora)

    negative_importance_word_list = []
    positive_important_word_list = []
    doesnt_importance_word_list = []

    for potential_word in dictionary_of_importante_words:
        if math.isclose(dictionary_of_importante_words[potential_word], 0.0):
            doesnt_importance_word_list.append((potential_word, dictionary_of_importante_words[potential_word]))
        elif dictionary_of_importante_words[potential_word] < 0.0:
            negative_importance_word_list.append((potential_word, dictionary_of_importante_words[potential_word]))
        elif dictionary_of_importante_words[potential_word] > 0.0:
            positive_important_word_list.append((potential_word, dictionary_of_importante_words[potential_word]))

    print('Negative importance words (count = {})\n'.format(len(negative_importance_word_list)))
    accuracy_with_negative_important_words, accuracy_without_negative_importance_words = check_with_deleted_specific_words(neural_model,
                                                                                    tokenizer_instance,
                                                                                    test_corpora,
                                                                                    list(map(lambda x: x[0], negative_importance_word_list)))

    print('Positive importance words (count = {})\n'.format(len(positive_important_word_list)))
    accuracy_with_positive_importance_words, accuracy_without_positive_importance_words = check_with_deleted_specific_words(neural_model,
                                                                                    tokenizer_instance,
                                                                                    test_corpora,
                                                                                    list(map(lambda x : x[0], positive_important_word_list)))

    print('Doesnt affected (importance) words (count = {})\n'.format(len(doesnt_importance_word_list)))
    accuracy_with_doesnt_affected_important_words, accuracy_without_doesnt_affected_important_words = check_with_deleted_specific_words(neural_model,
                                                                                    tokenizer_instance,
                                                                                    test_corpora,
                                                                                    list(map(lambda x: x[0], doesnt_importance_word_list)))


    file_importance_words = open('LSTM_MOD_Important_words_RUS.txt', 'w', encoding='utf-8')
    file_importance_words.write('Positive importance words: \n')
    file_importance_words.write('\tAccuracy with words: {}\n\tAccuracy without words: {}\n\n'.format(accuracy_with_positive_importance_words, accuracy_without_positive_importance_words))
    for positive_important_word in sorted(positive_important_word_list, key = lambda x : x[1], reverse = True):
        line = '\t\t{} : (importance = {:7.6f})\n'.format(positive_important_word[0], positive_important_word[1])
        file_importance_words.write(line)

    file_importance_words.write('\n\nNegative importance words: \n')
    file_importance_words.write('\tAccuracy with words: {}\n\tAccuracy without words: {}\n\n'.format(accuracy_with_negative_important_words, accuracy_without_negative_importance_words))
    for negative_important_word in sorted(negative_importance_word_list, key = lambda x : x[1]):
        line = '\t\t{} : (importance = {:7.6f})\n'.format(negative_important_word[0],
                                                     negative_important_word[1])
        file_importance_words.write(line)

    file_importance_words.write('\n\nDoesn\'t importance words: \n')
    file_importance_words.write('\tAccuracy with words: {}\n\tAccuracy without words: {}\n\n'.format(accuracy_with_doesnt_affected_important_words, accuracy_without_doesnt_affected_important_words))
    for doesnt_important_word in sorted(doesnt_importance_word_list, key = lambda x : x[1], reverse = True):
        line = '\t\t{} : (importance = {:7.6f})\n'.format(doesnt_important_word[0], doesnt_important_word[1])
        file_importance_words.write(line)
    file_importance_words.close()
    print("Important words was processed")


if __name__ == "__main__":
    main()