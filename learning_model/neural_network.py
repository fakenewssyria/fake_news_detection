from tensorflow import keras
from random import randint
from random import uniform
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    
    def __init__(self, X_train, Y_train, X_test, Y_test):
        
        ''' set X_train, Y_train, X_test, Y_test values '''
        
#         scaler = MinMaxScaler()       
#         scaler.fit(X_train) 

        self.X_train = X_train
        self.Y_train = keras.utils.to_categorical(np.asarray(Y_train))
        
        self.X_test = X_test
        self.Y_test = keras.utils.to_categorical(np.asarray(Y_test))
    
    def get_min_max_values(self, features):
        
        min_features = np.zeros(len(features[0]))
        max_features = np.zeros(len(features[0]))
        
        for i in range(len(min_features)):
            for j in range(len(features)):
                min_features[i] = min(min_features[i], features[j][i])
                max_features[i] = max(max_features[i], features[j][i])
        
        return (min_features, max_features)

    def min_max_normalization(self, features):
    
        (min_features, max_features) = self.get_min_max_values(features)
        
        normalized_features = np.zeros(shape=(len(features), len(min_features)))
    
        for i in range(len(features)):
            for j in range(len(features[0])):
                normalized_features[i][j] = (features[i][j] - min_features[j]) / (max_features[j] - min_features[j])

        return normalized_features
  
    def find_best_nn(self, iterations=100):      
        
        ''' function that finds best Neural Network model '''
          
        return self.random_search(iterations)        
    
    def random_search(self, iterations):
        
        ''' function that finds the best neural network model using random search '''
        
        self.best_model = None
        self.best_history = None
        self.best_accuracy = 0

        for i in range(iterations):

            rand_batch_size = randint(1, 100)  # generate random batch size
            rand_dropout_rate = uniform(0, 1)  # generate random drop-out rate
            rand_hidden_units = randint(1, 100)  # generate random number of hidden units
            rand_learning_rate = uniform(0, 1)  # generate random learning rate
            rand_learning_rate_decay = uniform(0, 1)  # generate random learning rate decay
            rand_nb_layers = randint(1, 100)  # generate random number of hidden layers
            
            print("\nBest accuracy so far: %f\nIteration # %d:" % (self.best_accuracy, (i + 1)))
            # fit this combination of params
            this_model = self.create_nn_model(rand_hidden_units, rand_learning_rate, rand_learning_rate_decay, rand_dropout_rate, rand_nb_layers)
            this_history = this_model.fit(self.X_train, self.Y_train, batch_size=rand_batch_size, epochs=100, verbose=0, validation_data=(self.X_test, self.Y_test))

            # test this combination of params on the validation training data
            (_, this_accuracy) = this_model.evaluate(self.X_test, self.Y_test, batch_size=rand_batch_size)
            
            # decide if best model 
            if this_accuracy > self.best_accuracy:
                self.best_accuracy = this_accuracy
                self.best_model = this_model
                self.best_batch_size = rand_batch_size 
                self.best_hidden_units = rand_hidden_units 
                self.best_learning_rate = rand_learning_rate
                self.best_decay = rand_learning_rate_decay
                self.best_dropout = rand_dropout_rate 
                self.best_nb_layers = rand_nb_layers
                self.best_history = this_history
        
        self.print_best()  # print params of best NN model found
        self.plot_learning_curve()
        label_proba = self.best_model.predict(self.X_test)
        print(label_proba)
        predicted_y = [1 if label[1] > label[0] else 0 for label in label_proba]
        return predicted_y
    
    def create_nn_model(self, hidden_units, learning_rate, learning_rate_decay, dropout_rate, nb_layers):
        
        ''' function that creates a neural network model with given params '''
        
        input_layer = keras.layers.Input(shape=(len(self.X_train[0]),), dtype='float32')
        dropout_layer = keras.layers.Dropout(dropout_rate)(input_layer)
        hidden_layer = keras.layers.Dense(hidden_units, activation='sigmoid')(dropout_layer)  # add 1 hidden layer
        
        for _ in range(nb_layers):  # add nb_layers - 1 hidden layers
                hidden_layer = keras.layers.Dense(hidden_units, activation='sigmoid')(hidden_layer)
        
        output_layer = keras.layers.Dense(2, activation='sigmoid')(hidden_layer)

        temp = keras.models.Model(input_layer, output_layer)
        
        optimizer = keras.optimizers.Adam(lr=learning_rate, decay=learning_rate_decay)

        temp.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
        
        return temp
    
    def print_best(self):

        '''function that prints the hyper-parameters of the best model'''
        
        print("Best model found:")
        print("accuracy: %.3f" % self.best_accuracy)
        
        print("Best model found of hyper-parameters:")
        print("Learning rate: %.3f" % self.best_learning_rate)
        print("Drop-out: %.3f" % self.best_dropout)
        print("Number of hidden units: %d" % self.best_hidden_units)
        print("Batch size: %d" % self.best_batch_size)
        print("Learning rate decay: %.3f" % self.best_decay)
        print("Number of layers: %d" % self.best_nb_layers)

    def plot_learning_curve(self):
        plt.plot(self.best_history.history['loss'])
        plt.plot(self.best_history.history['val_loss'])
        plt.title('Learning curve for best model')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig("nn_learning_curve.png")
