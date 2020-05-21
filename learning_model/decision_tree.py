import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.tree.tree import DecisionTreeClassifier


class DecisionTree:
    
    ''' class that fits a decision tree classifier to our dataset '''
    
    def __init__(self, X_train, Y_train, X_test, Y_test):
        
        ''' set X_train, Y_train, X_test, Y_test values '''
        
        self.X_train = X_train
        self.Y_train = Y_train
    
        self.X_test = X_test
        self.Y_test = Y_test
    
    def find_best_tree(self):
        
        ''' function that uses grid search to find the best decision tree for our dataset'''
        
        ''' hyper-parameters to tune '''

        max_depths = np.random.randint(low=1, high=100, size=10) 
        min_samples_splits = np.random.uniform(low=0, high=1, size=10)
        min_samples_leafs = np.random.randint(low=1, high=10, size=10)
        max_features = np.array(range(1, 10))
        
        params = {'max_depth': max_depths,
                  'min_samples_split':min_samples_splits,
                  'min_samples_leaf' : min_samples_leafs,
                  'max_features':max_features
                  }
        
        ''' use grid search to find the best params '''        
        
        self.tree_grid_search = GridSearchCV(DecisionTreeClassifier(), params, cv=10)  
        self.tree_grid_search.fit(self.X_train, self.Y_train)

        return self.print_best_params()
       
    def print_best_params(self):
        
        ''' function that prints the parameters the best decision tree and returns the labels of the test dataset'''

        self.best_decision_tree = self.tree_grid_search.best_estimator_ # decision tree with best hyper-parameters
        print(self.tree_grid_search.best_params_)
        
        return self.best_decision_tree
