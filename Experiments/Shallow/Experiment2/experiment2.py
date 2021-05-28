import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score,\
    recall_score, precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pickle


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class LearningModel:
    '''
    Class for training picking the best model from FA-KES and testing it on buzzfeed
    '''
    def __init__(self, training_data_path, testing_data_path, standard_scaling=True, minmax_scaling=False):
        self.train_df = pd.read_csv(training_data_path, encoding='latin-1')
        self.train_df = self.train_df.drop(['article_title', 'article_content', 'source', 'source_category', 'unit_id'], axis=1)

        self.test_df = pd.read_csv(testing_data_path, encoding='latin-1')
        self.test_df = self.test_df.drop(['article_content'], axis=1)

        self.X_train = np.array(self.train_df.loc[:, self.train_df.columns != 'label'])
        self.y_train = np.array(self.train_df.loc[:, 'label'])

        self.X_test = np.array(self.test_df.loc[:, self.test_df.columns != 'label'])
        self.y_test = np.array(self.test_df.loc[:, 'label'])

        ''' scaling the features '''
        if standard_scaling:
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        if minmax_scaling:
            scaler = MinMaxScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        self.results_testing = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'auc',
                                                     'TN', 'FP', 'FN', 'TP'])

    def test_model(self, model_path, model_name, output_folder):
        df_test = self.test_df

        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        y_pred = model.predict(X_test)

        accuracy, precision, recall, f_measure, auc = get_stats(y_test, y_pred)

        print('Testing Scores:\nAccuracy: %.5f\nPrecision: %.5f\nRecall: %.5f\nF-measure: %.5f\nAUC: %.5f\n' %
              (accuracy, precision, recall, f_measure, auc))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        print("----------------------\n")
        print("| tp = %d   fp = %d |\n" % (tp, fp))
        print("----------------------\n")
        print("| fn = %d   tn = %d |\n" % (fn, tn))
        print("----------------------\n")

        # produce output vector
        self.produce_output_vector(df_test, y_pred, model_name, output_folder + 'output_vector/')

        self.results_testing = self.results_testing.append({
            'model': model_name,
            'accuracy': '{:.5f}'.format(accuracy),
            'precision': '{:.5f}'.format(precision),
            'recall': '{:.5f}'.format(recall),
            'f1': '{:.5f}'.format(f_measure),
            'auc': '{:.5f}'.format(auc),
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp
        }, ignore_index=True)

        self.save_error_metrics(output_folder='output/')

    def save_error_metrics(self, output_folder):
        errors_folder = os.path.join(output_folder, 'error_metrics/')
        mkdir(errors_folder)
        self.results_testing.sort_values(by=['accuracy'], ascending=False).to_csv(os.path.join(errors_folder, 'testing_errors.csv'), index=False)

    def produce_output_vector(self, df_test, predicted, model_name, output_folder):
        '''
        saves the testing data with the ouput vecctor added as a column
        :param df_test: the testing data
        :param predicted: the output vector
        :param model_name: the name of the model used
        :param output_folder: the folder to save the data frame in.
        :return:
        '''
        # add the predicted value to the df
        df_test['predicted'] = list(predicted)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df_test.to_csv(output_folder + '%s_test.csv' % model_name)


def get_stats(y_true, y_pred):
    '''
    gets the classification error metrics between the actual and the predicted
    :param y_true:
    :param y_pred:
    :return:
    '''
    accuracy = accuracy_score(y_true, y_pred)
    f_measure = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return accuracy, precision, recall, f_measure, auc


if __name__ == '__main__':
    training_data_path = '../input/feature_extraction_train_updated.csv'
    testing_data_path = '../input/buzzfeed_feature_extraction_test_20_updated.csv'

    models = ['decision_tree', 'logistic_regression', 'ridge', 'sgd', 'extra_trees', 'random_forest', 'ada_boost',
              'gradient_boost', 'xg_boost', 'linear_svc', 'nu_svc', 'svc']

    lm = LearningModel(training_data_path, testing_data_path)
    for model in models:
        print('*************** Results for %s ***************' % model)
        model_path = '../Experiment1/all_trained_models/%s.pickle' % model
        lm.test_model(model_path, model, 'output/')



