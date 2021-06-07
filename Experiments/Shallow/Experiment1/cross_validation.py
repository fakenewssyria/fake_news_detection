import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
import os
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import itertools as it
import pickle
from imblearn.over_sampling import SMOTE
from models_hyperparams import models_to_test, non_probabilistic_models_to_test
from models_hyperparams import hyperparameters_per_model as hyperparameters
from sklearn.pipeline import Pipeline


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class LearningModelCrossVal:
    '''
        Class for applying cross validation over a suite of classification algorithms with hyper parameter tuning
        Then, it picks the best sets of hyper parameters optimized by accuracy, then generates a new model
        with winning hyper parameters, trains it on the whole training data, then tests on testing data
        and reports testing error metrics
     '''
    def __init__(self, train_df, test_df, output_folder, cols_drop,
                 over_sample=False, k_neighbors=5, standard_scaling=True, minmax_scaling=False,
                 learning_curve=True):
        '''
        :param train_df: path to **training** data-set
        :param test_df: path to **testing* data-set
        :param output_folder: folder to store results (errors & plots)
        :param cols_drop: list of columns to drop. If none, set to None
        :param over_sample: whether to oversample training data. Default: False
        :param k_neighbors: number of nearset neighbours to consider in oversampling (SMOTE). Only used if over_sample=True
        :param standard_scaling: whether to standardize input data. Default True
        :param minmax_scaling: wehther to Min-Max scale input data. Default False
        :param learning_curve: whether to produce learning curves or not

        NOTE: if you want to scale data, set wither min_max or standard-scaling to True, NOT BOTH TRUE
        '''
        
        # read the datasets
        self.train_df = train_df
        self.test_df = test_df

        # drop un-necessary columns
        self.train_df = self.train_df.drop(cols_drop, axis=1)
        self.test_df = self.test_df.drop(cols_drop, axis=1)

        self.X_train = np.array(self.train_df.loc[:, self.train_df.columns != 'label'])
        self.y_train = np.array(self.train_df.loc[:, 'label'])
        
        self.X_test = np.array(self.test_df.loc[:, self.test_df.columns != 'label'])
        self.y_test = np.array(self.test_df.loc[:, 'label'])
        self.y_test_df = self.test_df.loc[:, self.test_df.columns == 'label']

        self.output_folder = output_folder
        self.over_sample = over_sample
        self.k_neighbors = k_neighbors

        self.standard_scaling = standard_scaling
        self.minmax_scaling = minmax_scaling

        self.learning_curve = learning_curve

        self.results_testing = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'auc',
                                                     'TN', 'FP', 'FN', 'TP'])
        self.results_validation = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'auc'])

        self.results_winning = pd.DataFrame(columns=['model', 'winning hyperparameters'])

    def cross_validation(self, model_to_use, possible_hyperparams, model_name, nb_splits, df_test, nb_repeats=None, probabilistic=True,
                         pos_class_label=1):
        '''
        cross validation function with hyper parameter tuning
        :param model_to_use: sklearn model
        :param possible_hyperparams: dictionary of hyper parameters
        :param model_name: name of the sklearn model (String)
        :param nb_splits: number of folds
        :param df_test: testing data frame
        :param nb_repeats: number of repeats if we want to apply repeated KFold cross validation. By default, None
        :param standard_scaling: whether to standardize or not. By default, True.
        :param minmax_scaling: whether to to MinMax scaling or not. By default, False
        :return:
        '''
        def get_param_grid(dicts):
            '''
            returns all possible combinations of hyperparameters passed. This will be used in the hyperparameters
            loop of the cross validation.
            :param dicts: the possible hyper parameters for a model, passed as a dictionary
            :return: all combinations of hyper parameters
            '''
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        if nb_repeats is None:
            skf = StratifiedKFold(n_splits=nb_splits, random_state=42)
        else:
            skf = RepeatedStratifiedKFold(n_splits=nb_splits, n_repeats=nb_repeats, random_state=42)

        tempModels = []

        print('total nb of combinations of hyper parameters: %d' % len(get_param_grid(possible_hyperparams)))
        for parameter in get_param_grid(possible_hyperparams):

            if probabilistic:
                model = model_to_use(**parameter, random_state=42)
            else:
                model = model_to_use(**parameter)

            accuracy_scores, precision_scores, recall_scores, f_scores, auc_scores = [], [], [], [], []

            count = 1
            for train_index, test_index in skf.split(X_train, y_train):
                # print('split nb %d' % count)
                X_train_inner, X_val = X_train[train_index], X_train[test_index]
                y_train_inner, y_val = y_train[train_index], y_train[test_index]

                if self.standard_scaling:
                    # standardization inside cross validation loop to avoid contamination
                    scaler = StandardScaler()
                    X_train_inner = scaler.fit_transform(X_train_inner)
                    X_val = scaler.transform(X_val)

                if self.minmax_scaling:
                    scaler = MinMaxScaler()
                    X_train_inner = scaler.fit_transform(X_train_inner)
                    X_val = scaler.transform(X_val)

                # over-sampling inside cross validation and AFTER standardization
                if self.over_sample:
                    # print('num of neighbors: %d' % self.k_neighbors)
                    sm = SMOTE(random_state=2, k_neighbors=self.k_neighbors)
                    X_train_res, y_train_res = sm.fit_sample(X_train_inner, y_train_inner)
                    X_train_inner = X_train_res
                    y_train_inner = y_train_res

                model.fit(X_train_inner, y_train_inner)
                y_pred = model.predict(X_val)

                accuracy, precision, recall, f_measure, auc = self.get_stats(y_val, y_pred)

                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f_scores.append(f_measure)
                auc_scores.append(auc)

                count += 1

            tempModels.append(
                [parameter, np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores),
                 np.mean(f_scores), np.mean(auc_scores)])

        # sort the parameters by their accuracy scores (reverse order from highest to lowest)
        tempModels = sorted(tempModels, key=lambda k: k[1], reverse=True)

        # the winning hyper parameters
        winning_hyperparameters = tempModels[0][0]

        print('winning hyper parameters: ', str(winning_hyperparameters))
        print('Best Validation Scores:\nAccuracy: %.5f\nPrecision: %.5f\nRecall: %.5f\nF-measure: %.5f\nAUC: %.5f\n' %
              (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5]))

        self.results_validation = self.results_validation.append({
            'model': model_name,
            'accuracy': '{:.5f}'.format(tempModels[0][1]),
            'precision': '{:.5f}'.format(tempModels[0][2]),
            'recall': '{:.5f}'.format(tempModels[0][3]),
            'f1': '{:.5f}'.format(tempModels[0][4]),
            'auc': '{:.5f}'.format(tempModels[0][5]),
        }, ignore_index=True)

        wh = ''
        i = 0
        max = len(winning_hyperparameters)
        for k, v in winning_hyperparameters.items():
            wh += '{}:{}'.format(str(k), str(v))
            if i <= max - 1:
                wh += ', '
            i += 1
        self.results_winning = self.results_winning.append({
            'model': model_name,
            'winning hyperparameters': wh
        }, ignore_index=True)

        if probabilistic:
            model = model_to_use(**winning_hyperparameters, random_state=42)
        else:
            model = model_to_use(**winning_hyperparameters)

        if self.standard_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if self.minmax_scaling:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if self.over_sample:
            sm = SMOTE(random_state=2, k_neighbors=self.k_neighbors)
            # print('num of neighbors: %d' % self.k_neighbors)
            X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
            X_train = X_train_res
            y_train = y_train_res

        model.fit(X_train, y_train)

        # save the tuned and trained model as pickle
        model_file_name = '%s.pickle' % model_name
        output_folder = self.output_folder
        destination = output_folder + 'trained_models/'

        # create the destination folder (where to save the model) if it does not already exist
        mkdir(destination)

        # dump the model as a pickle file in the destination folder
        print('saving model ...')
        with open(destination + model_file_name, 'wb') as file:
            pickle.dump(model, file)

        y_pred = model.predict(X_test)

        # generate probability of the positive class
        if probabilistic:
            probas = model.predict_proba(X_test)[:, pos_class_label]

        accuracy, precision, recall, f_measure, auc = self.get_stats(y_test, y_pred)

        print('Testing Scores:\nAccuracy: %.5f\nPrecision: %.5f\nRecall: %.5f\nF-measure: %.5f\nAUC: %.5f\n' %
              (accuracy, precision, recall, f_measure, auc))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        print("----------------------\n")
        print("| tp = %d   fp = %d |\n" % (tp, fp))
        print("----------------------\n")
        print("| fn = %d   tn = %d |\n" % (fn, tn))
        print("----------------------\n")

        # save testing error metrics
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

        # saving both validation and testing error metrics
        self.save_error_metrics()

        # produce output vector
        self.produce_output_vector(df_test, y_pred, model_name, output_folder + 'output_vector/')

        print('X_train: ', X_train.shape)
        print('y_train: ', y_train.shape)
        # produce learning curve
        # self.produce_learning_curve(X_train, y_train, model=model_to_use, model_name=model_name, parameters=winning_hyperparameters, nb_splits=10, output_folder=output_folder + 'learning_curves/', nb_repeats=10)

        if self.learning_curve:
            self.produce_learning_curve(model=model_to_use, model_name=model_name, parameters=winning_hyperparameters, nb_splits=10, output_folder=output_folder + 'learning_curves/', nb_repeats=10)

        # produce a risk data frame for advanced ml evaluation
        if probabilistic:
            self.generate_risk_dataframe(probas, y_test, y_pred, model_name, pos_class_label)
        return model

    def save_error_metrics(self):
        errors_folder = os.path.join(self.output_folder, 'error_metrics/')
        mkdir(errors_folder)

        self.results_validation.sort_values(by=['accuracy'], ascending=False).to_csv(os.path.join(errors_folder, 'validation_errors.csv'), index=False)
        self.results_testing.sort_values(by=['accuracy'], ascending=False).to_csv(os.path.join(errors_folder, 'testing_errors.csv'), index=False)
        self.results_winning.to_csv(os.path.join(errors_folder, 'winning_hyperparameters.csv'), index=False)

    def generate_risk_dataframe(self, probas, y_test, y_pred, model_name, pos_class_label=1):
        test_indexes = list(self.y_test_df.index)
        self.test_indexes = test_indexes
        risk_scores = probas
        nb_bins = 10

        # create a new dataframe of indices & their risk
        risk_df = pd.DataFrame(np.column_stack((test_indexes, y_test, y_pred, risk_scores)),
                               columns=['test_indices', 'y_test', 'y_pred', 'risk_scores'])
        risk_df = risk_df.sort_values(by='risk_scores', ascending=False)

        # # # create 4 bins of the data (like in the paper)
        # # risk_df['quantiles'] = pd.qcut(risk_df['risk_scores'], q=4, duplicates='drop')
        # risk_df['quantiles'] = pd.cut(risk_df['risk_scores'], self.nb_bins)
        # print(pd.cut(risk_df['risk_scores'], self.nb_bins).value_counts())
        items_per_bin = len(risk_df) // nb_bins
        bin_category = [0] * len(risk_df)
        for i in range(nb_bins):
            lower = i * items_per_bin
            if i != nb_bins - 1:
                upper = (i + 1) * items_per_bin
                bin_category[lower:upper] = [i] * (upper - lower)
            else:
                bin_category[lower:] = [i] * (len(range(lower, len(risk_df))))

        risk_df['quantiles'] = list(reversed(bin_category))
        # calculate mean empirical risk
        # which is the fraction of students from that bin who actually (as per ground truth)
        # failed to graduate on time (actually positive)
        mean_empirical_risk = []
        quantiles_sorted = sorted(list(risk_df['quantiles'].unique()))
        for quantile in quantiles_sorted:
            df = risk_df[risk_df['quantiles'] == quantile]
            # test_indices_curr = df['test_indices']
            # ground_truth = self.y_test_df.loc[test_indices_curr, :]
            ground_truth = df['y_test']
            # mean_empirical_risk.append(list(ground_truth[self.target_variable]).count(1)/len(ground_truth))
            mean_empirical_risk.append(list(ground_truth).count(pos_class_label) / len(ground_truth))
        print('quantiles: {}'.format(quantiles_sorted))
        print('mean empirical risk: {}'.format(mean_empirical_risk))
        self.save_risk_dataframe(risk_df, model_name)

    def save_risk_dataframe(self, risk_df, trained_model_name):
        ''' saves risk data frame for advanced ML evaluation '''
        risk_df.to_csv(os.path.join(self.output_folder + 'trained_models/',
                                    'risk_df_{}.csv'.format(trained_model_name)), index=False)

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

    def produce_learning_curve(self, model, model_name, nb_splits, output_folder, parameters=None, nb_repeats=None):

        '''
        produce learning curve of a certain model, using either KFold or repeated KFold cross validation
        :param model: the model
        :param model_name: name of the model, string.
        :param nb_splits: number of splits in KFold
        :param output_folder: path to output folder. If doesn't exist, will be created at runtime
        :param nb_repeats: number of repeats in case of RepeatedKFold. By defualt None. If None,
        KFold will be used instead
        :return: saves the learning curve
        '''

        X_train, y_train = self.X_train, self.y_train

        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('model', model(**parameters, random_state=42))
        ])

        print('Inside Learning curve')
        print('X_train: ', self.X_train.shape)
        print('y_train: ', self.y_train.shape)

        if nb_repeats is None:
            cv = StratifiedKFold(n_splits=nb_splits, random_state=42)
        else:
            cv = RepeatedStratifiedKFold(n_splits=nb_splits, n_repeats=nb_repeats, random_state=42)

        # train_sizes, train_scores, test_scores = learning_curve(model(**parameters), X_train, y_train, cv=cv, scoring='accuracy')  # calculate learning curve values

        train_sizes, train_scores, test_scores = learning_curve(pipe, X_train, y_train, cv=cv, scoring='accuracy')  # calculate learning curve values

        # mean of the results of the training and testing
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure()
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Accuracy")

        plt.plot(train_sizes, train_scores_mean, label="training")
        plt.plot(train_sizes, test_scores_mean, label="validation")
        plt.legend()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + '%s_learning_curve.png' % model_name)
        plt.close()

    def get_stats(self, y_true, y_pred):
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
    testing_data_path = '../input/feature_extraction_test_updated.csv'

    train_df = pd.read_csv(training_data_path, encoding='latin-1')
    test_df = pd.read_csv(testing_data_path, encoding='latin-1')

    cols_drop = ['article_title', 'article_content', 'source', 'source_category', 'unit_id']
    lm = LearningModelCrossVal(train_df, test_df, output_folder='output_probabilistic/',
                               cols_drop=cols_drop,
                               over_sample=False,
                               standard_scaling=True,
                               minmax_scaling=False,
                               learning_curve=False)

    for model in models_to_test:
        model_name = models_to_test[model]
        print('\n********** Results for %s **********' % model_name)
        t0 = time.time()
        lm.cross_validation(model, hyperparameters[model_name], model_name, 10, lm.test_df, 10, probabilistic=True)
        t1 = time.time()
        print('function took %.3f min\n' % (float(t1 - t0) / 60))

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Non - Probabilistic Models')
    lm = LearningModelCrossVal(train_df, test_df, output_folder='output_non_probabilistic/',
                               cols_drop=cols_drop,
                               over_sample=False,
                               standard_scaling=True,
                               minmax_scaling=False,
                               learning_curve=False)
    for model in non_probabilistic_models_to_test:
        model_name = non_probabilistic_models_to_test[model]
        print('\n********** Results for %s **********' % model_name)
        t0 = time.time()
        lm.cross_validation(model, hyperparameters[model_name], model_name, 10, lm.test_df, 10, probabilistic=False)
        t1 = time.time()
        print('function took %.3f min\n' % (float(t1 - t0) / 60))

