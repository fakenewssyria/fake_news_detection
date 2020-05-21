from sklearn.linear_model.base import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class Regression:
        
    def __init__(self, X_train, Y_train, X_test, Y_test):
        
        ''' set X_train, Y_train, X_test, Y_test values '''
        
        self.X_train = X_train
        self.Y_train = Y_train
        
        self.X_test = X_test
        self.Y_test = Y_test

    def polynomial_linear_regression(self):
    
            best_accuracy = 0
            best_degree = 0
        
#         for degree in range(2, 10):
            
            degree = 2
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())  # polynomial transformation of this degree 
            model.fit(self.X_train, self.Y_train)  # fit the model
            
            ''' check accuracy using test dataset '''
            
            predicted_y = model.predict(self.X_test)
            predicted_y = [ 1 if(abs(1 - val) < abs(val)) else 0 for val in predicted_y]
            
            accuracy = accuracy_score(self.Y_test, predicted_y)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_degree = degree
                self.best_model = model
                
                print(best_degree)
        
                return model
        