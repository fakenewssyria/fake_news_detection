import numpy as np
import pandas as pd
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from tensorflow import keras
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, auc, roc_curve, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import load_model
from keras import optimizers
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.layers import Reshape, Conv2D, GlobalMaxPooling2D
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import string
import re


class TextBased:
    def __init__(self, train_df, test_df):

        # training and testing data
        self.X_train = np.array(train_df.loc[:, 'article_content'])
        self.y_train = np.array(train_df.loc[:, 'label'])

        self.X_test = np.array(test_df.loc[:, 'article_content'])
        self.y_test = np.array(test_df.loc[:, 'label'])

        # the glove file
        self.glove_file = open('glove.6B.300d.txt', encoding="utf8")

        self.padded_sentences_train = None
        self.padded_sentences_test = None

        # split training data into: training and validation data
        self.xtrain, self.xval, self.ytrain, self.yval = None, None, None, None

    def pre_process_data(self):

        # stemming for each article
        stemmer = SnowballStemmer("english")
        self.X_train = [stemmer.stem(x) for x in self.X_train]
        self.X_test = [stemmer.stem(x) for x in self.X_test]

        # remove stop words
        stop_words = set(stopwords.words('english'))

        # remove stop words from self.X_train
        for i in range(len(self.X_train)):
            tokens = word_tokenize(self.X_train[i])
            filtered_article = [w for w in tokens if not w in stop_words]
            self.X_train[i] = ' '.join(filtered_article)

        # remove stop words from self.X_test
        for i in range(len(self.X_test)):
            tokens = word_tokenize(self.X_test[i])
            filtered_article = [w for w in tokens if not w in stop_words]
            self.X_test[i] = ' '.join(filtered_article)

        # remove punctuation from text, transform text to lowercase, remove single characters, remove multiple spaces
        for i in range(len(self.X_train)):
            # remove punctuation
            self.X_train[i] = self.X_train[i].translate(str.maketrans('', '', string.punctuation))
            # transform text to lower case
            self.X_train[i] = self.X_train[i].lower()
            # remove single characters
            self.X_train[i] = re.sub(r"\s+[a-zA-Z]\s+", ' ', self.X_train[i])
            # remove multiple spaces
            self.X_train[i] = re.sub(r'\s+', ' ', self.X_train[i])

        for i in range(len(self.X_test)):
            # remove punctuation
            self.X_test[i] = self.X_test[i].translate(str.maketrans('', '', string.punctuation))
            # transform text to lower case
            self.X_test[i] = self.X_test[i].lower()
            # remove single characters
            self.X_test[i] = re.sub(r"\s+[a-zA-Z]\s+", ' ', self.X_test[i])
            # remove multiple spaces
            self.X_test[i] = re.sub(r'\s+', ' ', self.X_test[i])

        #self.y_train = pd.get_dummies(self.y_train).to_numpy()
        #self.y_test = pd.get_dummies(self.y_test).to_numpy()

    def text_to_sequences(self):
        # tokenize (fitting) on the training data (set oov_token to True so new words in
        # X_test are not ignored)
        word_tokenizer = Tokenizer(oov_token=True)
        word_tokenizer.fit_on_texts(self.X_train)

        # length of the vocabulary
        self.vocab_length = len(word_tokenizer.word_index) + 1

        # text_to_sequences on both the training and the testing
        embedded_sentences_train = word_tokenizer.texts_to_sequences(self.X_train)
        embedded_sentences_test = word_tokenizer.texts_to_sequences(self.X_test)

        word_count = lambda sentence: len(word_tokenize(sentence))
        longest_sentence = max(self.X_train, key=word_count)
        self.length_long_sentence = len(word_tokenize(longest_sentence))

        self.padded_sentences_train = pad_sequences(embedded_sentences_train, self.length_long_sentence, padding='post')
        self.padded_sentences_test = pad_sequences(embedded_sentences_test, self.length_long_sentence, padding='post')

        embeddings_dictionary = dict()
        for line in self.glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions

        self.glove_file.close()

        # embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_length, 300))
        for word, index in word_tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index] = embedding_vector

    def train_val_split(self):
        # split the training data into 90% training and 10% validation
        self.xtrain, self.xval, self.ytrain, self.yval = train_test_split(self.padded_sentences_train, self.y_train
                                                        ,test_size=0.1, random_state=42)

    def build_model_LSTM(self):

        ''' LSTM model '''

        model = Sequential()
        embedding_layer = Embedding(self.vocab_length, 300, weights=[self.embedding_matrix], input_length=self.length_long_sentence, trainable=False)
        model.add(embedding_layer)
        model.add(LSTM(84, kernel_regularizer=l2(0.1), dropout=0.2))
        model.add(Dense(64, kernel_regularizer=l2(0.1), activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def build_model_FeedForward(self):

        ''' Feed forward model '''

        model = Sequential()
        embedding_layer = Embedding(self.vocab_length, 300, weights=[self.embedding_matrix], input_length=self.length_long_sentence, trainable=False)
        model.add(embedding_layer)
        model.add(Flatten())
        model.add(Dense(200, kernel_regularizer=l2(0.01), activation='relu'))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def build_model_CNN(self):

        ''' CNN '''

        model = Sequential()
        embedding_layer = Embedding(self.vocab_length, 300, weights=[self.embedding_matrix], input_length=self.length_long_sentence,
                                    trainable=False)
        model.add(embedding_layer)
        model.add(Conv1D(filters=150, kernel_regularizer=l2(0.01), kernel_size=5, strides=1, padding='valid'))
        model.add(MaxPooling1D(2, padding='valid'))
        model.add(Conv1D(filters=150, kernel_regularizer=l2(0.01), kernel_size=5, strides=1, padding='valid'))
        model.add(MaxPooling1D(2, padding='valid'))
        model.add(Flatten())
        model.add(Dense(80, kernel_regularizer=l2(0.01), activation='relu'))
        model.add(Dense(40, kernel_regularizer=l2(0.01), activation='relu'))
        model.add(Dense(20, kernel_regularizer=l2(0.01), activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
        model.summary()

    def test_model(self, model, model_name, output_folder):

        ''' produce validation & testing error metrics + learning curve '''

        estimator = model.fit(self.xtrain, self.ytrain, epochs=100, batch_size=64, validation_data=(self.xval, self.yval), verbose=1)

        # produce learning curve
        produce_learning_curve(estimator, output_folder, model_name)

        # produce validation error metrics
        predictedLabels = model.predict(self.xval)
        ytClass = np.argmax(self.yval, axis=1)
        ypClass = np.argmax(predictedLabels, axis=1)

        # Validation error metrics
        print('\nValidation error metrics: ')
        get_stats(ytClass, ypClass)

        # re-train on the whole training data
        model.fit(self.padded_sentences_train, self.y_train, epochs=100, batch_size=64, verbose=1)

        # predict on the remaining 20% testing
        predictedLabels = model.predict(self.padded_sentences_test)
        ytClass = np.argmax(self.y_test, axis=1)
        ypClass = np.argmax(predictedLabels, axis=1)

        # Testing error metrics
        print('\nTesting error metrics: ')
        get_stats(ytClass, ypClass)


def produce_learning_curve(estimator, output_folder, model_name):
    ''' method for plotting the learning curve '''
    loss = estimator.history['loss']
    vloss = estimator.history['val_loss']

    plt.plot(loss)
    plt.plot(vloss)

    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    destination = output_folder + '/learning_curve/'
    if not os.path.exists(destination):
        os.makedirs(destination)
    plt.savefig(destination + '%s_learning_curve.png' % model_name)
    plt.close()


def get_stats(ytClass, ypClass):
    ''' method for getting classification error metrics '''
    accuracy = accuracy_score(ytClass, ypClass)
    fmeasure = f1_score(ytClass, ypClass)
    precision = precision_score(ytClass, ypClass)
    recall = recall_score(ytClass, ypClass)
    auc = roc_auc_score(ytClass, ypClass)

    print('Accuracy: %.5f' % accuracy)
    print('Precision: %.5f' % precision)
    print('Recall: %.5f' % recall)
    print('F-measure: %.5f' % fmeasure)
    print('AUC: %.5f' % auc)

    tn, fp, fn, tp = confusion_matrix(ytClass, ypClass, labels=[0, 1]).ravel()

    print("----------------------\n")
    print("| tp = %d   fp = %d |\n" % (tp, fp))
    print("----------------------\n")
    print("| fn = %d   tn = %d |\n" % (fn, tn))
    print("----------------------\n")

