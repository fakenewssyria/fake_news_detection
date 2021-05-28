import pandas as pd
import numpy as np
import os
import random
from tensorflow.python.platform import flags
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
FLAGS = flags.FLAGS


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class DataGenerator(object):
    """
    Data Generator capable of generating batches of any 2D dataset given as input.
    """
    def __init__(self, num_samples_per_class, batch_size):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.meta_batchsz = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = FLAGS.num_classes

        # training and testing data
        self.training_path = FLAGS.training_data_path
        self.testing_path = FLAGS.testing_data_path
        self.target_variable = FLAGS.target_variable
        self.cols_drop = FLAGS.cols_drop
        self.special_encoding = FLAGS.special_encoding

        # for dropping un-wanted columns
        if self.cols_drop is not None:
            if self.special_encoding:
                self.df_train = pd.read_csv(self.training_path, encoding=self.special_encoding).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path, encoding=self.special_encoding).drop(self.cols_drop, axis=1)
            else:
                self.df_train = pd.read_csv(self.training_path).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path).drop(self.cols_drop, axis=1)
        else:
            if self.special_encoding is not None:
                self.df_train = pd.read_csv(self.training_path, encoding=self.special_encoding)
                self.df_test = pd.read_csv(self.testing_path, encoding=self.special_encoding)
            else:
                self.df_train = pd.read_csv(self.training_path)
                self.df_test = pd.read_csv(self.testing_path)

        # create combined dataset for FP growth
        self.df = pd.concat([self.df_train, self.df_test])

        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable])

        self.dim_input = self.X_train.shape[1]
        self.dim_output = self.num_classes

        # scaling the data
        if FLAGS.scaling is not None:
            if FLAGS.scaling == 'min-max':
                scaler = MinMaxScaler()
            elif FLAGS.scaling == 'z-score':
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

    def sample_tasks(self, train):
        all_idxs, all_labels = [], []
        if train:
            x, y = self.X_train, self.y_train
        else:
            x, y = self.X_test, self.y_test

        for i in range(self.num_classes):
            idxs = [idx for idx in range(len(x)) if y[idx] == i]
            idxs_chosen = random.sample(idxs, self.num_samples_per_class)
            labels_curr = [i] * len(idxs_chosen)
            labels_curr = np.array([labels_curr, -(np.array(labels_curr) - 1)]).T

            all_idxs.extend(idxs_chosen)
            all_labels.extend(labels_curr)

        zipped = list(zip(all_idxs, all_labels))
        random.shuffle(zipped)
        all_idxs, all_labels = zip(*zipped)

        return x[all_idxs, :], np.array(all_labels)

    def make_data_tensor(self, train=True):
        if train:
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 1000
        else:
            num_total_batches = 600

        all_data, all_labels = [], []

        for ifold in range(num_total_batches):
            data, labels = self.sample_tasks(train=train)
            all_data.extend(data)
            all_labels.extend(labels)

        examples_per_batch = self.num_classes * self.num_samples_per_class  # 2*16 = 32

        all_data_batches, all_label_batches = [], []
        for i in range(self.meta_batchsz):
            data_batch = all_data[i * examples_per_batch:(i + 1) * examples_per_batch]
            labels_batch = all_labels[i * examples_per_batch: (i + 1) * examples_per_batch]

            all_data_batches.append(np.array(data_batch))
            all_label_batches.append(np.array(labels_batch))

        all_image_batches = np.array(all_data_batches)
        all_label_batches = np.array(all_label_batches)

        return all_image_batches, all_label_batches