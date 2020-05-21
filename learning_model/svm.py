import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class SVM:
    
    ''' class that fits SVM to our dataset '''

    def __init__(self, X_train, Y_train, X_test, Y_test):
        
        ''' set X_train, Y_train, X_test, Y_test values '''
        
        self.X_train = X_train
        self.Y_train = Y_train
    
        self.X_test = X_test
        self.Y_test = Y_test
    
    def find_best_svm(self):
        
        ''' function that uses grid search to find the best SVM for our dataset'''

        ''' hyper-parameters to tune '''

        soft_c = np.random.uniform(low=0, high=10, size=10)  # C values for soft-margin SVM
        hard_c = np.random.randint(low=100, high=1000, size=10)  # C values for hard-margin SVM
           
        params = {'C': soft_c,  # np.concatenate((soft_c, hard_c)),
                  'kernel':['poly', 'rbf', 'linear']  # kernel values
                  }  
        
        ''' use grid search to find best SVM params '''
        
        self.svm_grid_search = GridSearchCV(SVC(gamma='scale'), params, cv=10, scoring='f1')  
        
        self.svm_grid_search.fit(self.X_train, self.Y_train)
        
        return self.print_best_params() 
        
    def print_best_params(self):
        
        ''' function that prints the best SVM and returns the labels of the test dataset '''

        self.best_svm = self.svm_grid_search.best_estimator_  # SVM with best hyper-params
        
        print("Hyper-parameters of the best SVM found:", self.svm_grid_search.best_params_)
        print("Support vectors of the best SVM found: ", self.best_svm.n_support_)
        
        return self.best_svm