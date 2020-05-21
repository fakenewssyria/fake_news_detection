import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from tensorflow import keras
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class TextBased:
    
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 0
    
    def run(self, data_path, glove_file_path):
        
        self.read_data(data_path)
        self.preprocess_data()
        self.text_to_sequences()
        self.glove_embeddings(glove_file_path)
        self.set_embedding_matrix()
        self.compile_lstm()
        self.split_train_test()
        self.fit_lstm()
        self.test_lstm()
        
    def read_data(self, data_path):
        
        ''' read dataset '''
        
        df = pd.read_csv(data_path, encoding='latin-1')   
        self.X = np.array(df.loc[:, 'article_content'])  # text
        self.Y = np.array(df.loc[:, 'label'])  # label

    def preprocess_data(self):
        
        '''stemming'''
        stemmer = SnowballStemmer("english")
        self.X = [stemmer.stem(x) for x in self.X] # stem each article
        
        '''removing stopwords'''
        stop_words = set(stopwords.words('english'))

        for i in range(len(self.X)):
            tokens = word_tokenize(self.X[i]) # transform article to tokens
            filtered_article = [w for w in tokens if not w in stop_words] # filter out the stop words from the article
            self.X[i] = ' '.join(filtered_article) # transform the filtered article back to a string
  
    def glove_embeddings(self, glove_file):
        
        '''Extract embeddings from glove file'''
        self.embeddings_index = {}
        f = open(glove_file, encoding = 'utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
            
        f.close()
    
    def text_to_sequences(self):

        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(self.X) # tokenize the articles
        
        self.word_index = tokenizer.word_index # number of unique words
        self.vocab_size = len(tokenizer.word_index) + 1

        sequences = tokenizer.texts_to_sequences(self.X)
        
        for sequence in sequences:
            if(len(sequence) > self.MAX_SEQUENCE_LENGTH):
                self.MAX_SEQUENCE_LENGTH = len(sequence)
        
        self.X = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post') #pad word sequences
            
    def set_embedding_matrix(self):
        
        '''map our data's words to embeddings found'''
        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM)) # init embedding matrix to 0's
        for word, i in self.word_index.items():
            if i >= self.vocab_size:
                continue
            self.embedding_vector = self.embeddings_index.get(word)
            if self.embedding_vector is not None: # if word exists in embedding file, update embedding matrix
                self.embedding_matrix[i] = self.embedding_vector 
    
    def split_train_test(self):
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2)
        
    def compile_lstm(self):

        embedding_layer = keras.layers.Embedding(self.vocab_size, 
                                         self.EMBEDDING_DIM, 
                                         weights=[self.embedding_matrix], 
                                         input_length=self.MAX_SEQUENCE_LENGTH,
                                         trainable=False) # embedding layer using the pretrained embeddings
        
        '''create the layers'''
    
        self.model = keras.models.Sequential()
        self.model.add(embedding_layer)
        self.model.add(keras.layers.LSTM(50))  
        self.model.add(keras.layers.Dense(1, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        
    def fit_lstm(self):
        self.model.fit(self.X_train, self.Y_train, batch_size=32, epochs= 10)
        
    def test_lstm(self):
        predicted_y = self.model.predict(self.X_test)
        print(f1_score(self.Y_test, predicted_y))
        
        
        
        
        
        