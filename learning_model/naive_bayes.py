from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score


class NaiveBayes:

    def __init__(self, X_train, Y_train, X_test, Y_test):
        
        ''' set X_train, Y_train, X_test, Y_test values '''
        
        self.X_train = X_train
        self.Y_train = Y_train
    
        self.X_test = X_test
        self.Y_test = Y_test

    def find_best_naive_bayes(self):
        
        ''' function that tests Naive Bayes classifiers to find the best for our dataset '''
        
        naive_bayes_models = [ BernoulliNB, GaussianNB()]
            
        best_accuracy = 0
        for model in naive_bayes_models:
            
            model.fit(X = self.X_train, y = self.Y_train)
            predicted_y = model.predict(self.X_test)  # predict the test y using this model
            accuracy = accuracy_score(self.Y_test, predicted_y)  # calculate accuracy of test data
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
        
        return self.best_model