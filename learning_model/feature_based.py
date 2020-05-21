import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from learning_model.neural_network import NeuralNetwork
from learning_model.regression import Regression
from learning_model.decision_tree import DecisionTree
from learning_model.svm import SVM
from learning_model.naive_bayes import NaiveBayes
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, recall_score, precision_score, confusion_matrix
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier


class FeatureBased:
    
    def read_split_dataset(self, training_data_path, test_data_path):
        
        ''' read training dataset '''
        
        self.train_df = pd.read_csv(training_data_path, encoding='latin-1')   

        self.features = ['source_attribution', 'assertive_verbs', 'factive_verbs', 'implicative_verbs', 'hedges', 'report_verbs', 'bias', 'sectarian_language', 'consistency_score']
       
        self.X_train = np.array(self.train_df.loc[:, self.features])  # features
        print(self.X_train)
        self.Y_train = np.array(self.train_df.loc[:, 'label'])  # labels
        
        ''' read testing dataset '''
        
        self.test_df = pd.read_csv(test_data_path, encoding='latin-1')        
        self.X_test = np.array(self.test_df.loc[:, ['source_attribution', 'assertive_verbs', 'factive_verbs', 'implicative_verbs', 'hedges', 'report_verbs', 'bias', 'sectarian_language', 'consistency_score']])  # features
        self.Y_test = np.array(self.test_df.loc[:, 'label'])  # labels 
        
        ''' scaling the features '''
        scaler = preprocessing.StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train) 
        self.X_test = scaler.transform(self.X_test)
        
    def test_nn(self):
        
        ''' function that tests Feed Forward Neural Network on our dataset'''

        print("Neural Networks")
        self.nn = NeuralNetwork(self.X_train, self.Y_train, self.X_test, self.Y_test)
        predicted_y = self.nn.find_best_nn(iterations=1)

        self.print_stats(predicted_y, "nn")

    def test_regression(self):
        
        ''' function that tests a Polynomial Linear Regression on our dataset'''

        print("Linear Regression")
        self.regression = Regression(self.X_train, self.Y_train, self.X_test, self.Y_test)
        model = self.regression.polynomial_linear_regression()
        predicted_y = model.predict(self.X_test)
        predicted_y = [ 1 if(abs(1 - val) < abs(val)) else 0 for val in predicted_y]    
        self.print_stats(predicted_y, "regression")
        self.plot_learning_curve(model, "regression")
        
    def test_svm(self):
        
        ''' function that tests SVM on our dataset '''
        
        print("SVM")
        self.svm = SVM(self.X_train, self.Y_train, self.X_test, self.Y_test)
        model = self.svm.find_best_svm()
        predicted_y = model.predict(self.X_test)
        self.print_stats(predicted_y, "svm")
        self.plot_learning_curve(model, "svm")
        
    def test_decision_tree(self):
        
        ''' function that tests Decision Tree Classifier on our dataset'''

        print("Decision Tree")
        self.decision_tree = DecisionTree(self.X_train, self.Y_train, self.X_test, self.Y_test)
        model = self.decision_tree.find_best_tree()
        predicted_y = model.predict(self.X_test)
        self.print_stats(predicted_y, "tree")
        self.plot_learning_curve(model, "tree")

    def test_naive_bayes(self):

        ''' function that tests Naive Bayes Classifiers on our dataset'''
        
        print("Naive Bayes")
        self.naive_bayes = NaiveBayes(self.X_train, self.Y_train, self.X_test, self.Y_test)
        model = self.naive_bayes.find_best_naive_bayes()
        predicted_y = model.predict(self.X_test)
        self.print_stats(predicted_y, "naive_bayes")
        self.plot_learning_curve(model, "naive_bayes")
        
    def print_stats(self, predicted_y, title):
                
        count0 = 0
        count1 = 0
        
        for y in predicted_y:
            if y == 0:
                count0 += 1
            else:
                count1 += 1
                
        print("%d %d\n" % (count0, count1))
        fpr, tpr, _ = roc_curve(self.Y_test, predicted_y)

        best_auc = auc(fpr, tpr)
        accuracy = accuracy_score(self.Y_test, predicted_y)
        best_f_score = f1_score(self.Y_test, predicted_y)
        best_recall = recall_score(self.Y_test, predicted_y)
        best_precision = precision_score(self.Y_test, predicted_y)
        
        print("Area under the curve %f\n" % best_auc) 
        print("Accuracy %f\n" % accuracy) 
        print("F-score %f\n" % best_f_score)
        print("Precision %f\n" % best_precision)
        print("Recall %f\n" % best_recall)
        
        print("probability of class 0: %f\n" % (count0 / len(predicted_y)))
        print("probability of class 1: %f\n" % (count1 / len(predicted_y)))
        
#         plt.figure()

        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
#         plt.savefig("output/roc_plots/" + title + "_roc.png")
        
        tp, fp, fn, tn = confusion_matrix(self.Y_test, predicted_y, labels=[0, 1]).ravel()

        print("----------------------\n")
        print("| tp = %d   fp = %d |\n" % (tp, fp))
        print("----------------------\n")
        print("| fn = %d   tn = %d |\n" % (fn, tn))
        print("----------------------\n")

#         self.plot_pca(predicted_y, title)

    def plot_pca(self, predicted_y, title):
        
        ''' function that plots the clusters using PCA '''
        
        x = StandardScaler().fit_transform(self.X_test)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        
        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
        
        principalDf["label"] = pd.DataFrame(predicted_y)
        
        fig = plt.figure(figsize=(8, 8))
        
        ax = fig.add_subplot(1, 1, 1) 
        
        targets = [1, 0]
        colors = ['r', 'b']
        
        for target, color in zip(targets, colors):
            indicesToKeep = principalDf['label'] == target
            ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                       , principalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color)

        plt.savefig("output/pca_plots/" + title + "_pca.png", bbox_inches='tight')
    
    def plot_learning_curve(self, model, title):
    
        '''plot the learning curve'''
        
        # set the cross-validation factor to be used in evaluating learning curve
        cv = ShuffleSplit(n_splits=len(self.Y_train), test_size=0.2)  # 20% for validation
        
        train_sizes, train_scores, test_scores = learning_curve(model, self.X_train, self.Y_train, cv=cv)  # calculate learning curve values
        
        # mean of the results of the training and testing
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
         
        '''plot results'''
        
        plt.figure()
        plt.xlabel("Number of Training Points")
        plt.ylabel("Error")
        
        plt.plot(train_sizes, train_scores_mean, color="r", label="Ein")
        plt.plot(train_sizes, test_scores_mean, color="g", label="Eval")
        
        frame = plt.gca()
        frame.axes.yaxis.set_ticklabels([])
        frame.invert_yaxis()
        
        plt.legend(loc="best")
        plt.savefig("output/learning_model_plots/" + title + "_learning_model.png", bbox_inches='tight')
        
    def create_svm(self, best_kernel, best_c):
        svm = SVC(gamma='scale', kernel=best_kernel, C=best_c)
        svm.fit(self.X_train, self.Y_train)
        predicted_y = svm.predict(self.X_test)
        self.print_stats(predicted_y, "svm")
        
    def create_decision_tree(self):
        
        ''' based on experiments our best model was the decision tree model with the following params: '''
        
        tree = DecisionTreeClassifier(max_depth=65, min_samples_split=0.03, min_samples_leaf=3 , max_features=8)
        tree.fit(self.X_train, self.Y_train)
        predicted_y = tree.predict(self.X_test)
        print(predicted_y)
        self.print_stats(predicted_y, "")
        self.test_df['learning_label'] = predicted_y
        self.test_df.to_csv('output/feature_extraction.csv', encoding="latin-1")  # save the training dataset

    def feature_importance(self):

        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=0)
        
        forest.fit(self.X_train, self.Y_train)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        
        # Print the feature ranking
        print("Feature ranking:")
        
        for f in range(self.X_train.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, self.features[indices[f]], importances[indices[f]]))
        
        x_ticks = []
        
        for index in indices:
            x_ticks.append(self.features[index])
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(x_ticks, importances[indices],
               color="r"#, yerr=std[indices]
               , align="center")
        plt.xticks( x_ticks, rotation='vertical')
        plt.savefig("importance.png", bbox_inches='tight', rotation="vertical")

    def select_k_best(self):
        test = SelectKBest(score_func=f_classif, k=4)
        fit = test.fit(self.X_train, self.Y_train)
        mask = test.get_support() #list of booleans
        new_features = [] # The list of your K best features
        
        for bool, feature in zip(mask, self.features):
            if bool:
                new_features.append(feature)
        print(new_features)
        # summarize scores
        print(fit.scores_)
        features = fit.transform(self.X_train)
        # summarize selected features
        print(features[0:5,:])
    
    def majority_class(self):
        predicted = np.ones(len(self.Y_test))
        print(accuracy_score(self.Y_test, predicted))
    