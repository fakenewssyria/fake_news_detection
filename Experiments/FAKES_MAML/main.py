"""
Usage Instructions:
    10-shot sinusoid:
        python main_fn.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main_fn.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main_fn.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main_fn.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main_fn.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main_fn.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main_fn.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.

    Note that better sinusoid results can be achieved by using a larger network.
"""
import csv
import numpy as np
import pandas as pd
import pickle
import random
import os
import tensorflow as tf
from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from sklearn.metrics import *

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_string('training_data_path', 'input/feature_extraction_train_updated.csv', 'path to training data')
flags.DEFINE_string('testing_data_path', 'input/feature_extraction_test_updated.csv', 'path to testing data')
flags.DEFINE_string('target_variable', 'label', 'name of the target variable column')
flags.DEFINE_list('cols_drop', ['article_title', 'article_content', 'source', 'source_category', 'unit_id'], 'list of column to drop from data, if any')

flags.DEFINE_string('special_encoding', 'latin-1', 'special encoding needed to read the data, if any')
flags.DEFINE_string('scaling', 'z-score', 'scaling done to the dataset, if any')

flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 1000, 'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 32, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.1, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 32, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 0.001, 'step size alpha for inner gradient update.')
flags.DEFINE_integer('num_updates', 4, 'number of inner gradient updates during training.')

## Base model hyper parameters
flags.DEFINE_string('dim_hidden', '128, 64', 'number of neurons in each hidden layer')
flags.DEFINE_string('dim_name', 'dim0', 'unique index name for the list of hidden layers (above)')
flags.DEFINE_string('activation_fn', 'relu', 'activation function used')
flags.DEFINE_integer('model_num', 18, 'model number to store trained model. Better for tracking')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', False, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'trained_models/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot


def compute_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1score = f1_score(labels, predictions)
    roc = roc_auc_score(labels, predictions)

    return accuracy, precision, recall, f1score, roc


def unstack_prediction_probabilities(support_predicted, support_actual, support_probas,
                                     query_predicted, query_actual, query_probas,
                                     exp_string):
    """ creates the risk data frame needed for advanced ML Evaluation """
    risk_df = pd.DataFrame()
    y_test, y_pred, probas = [], [], []
    for mini_batch in range(len(support_predicted)):
        y_pred.extend(support_predicted[mini_batch])
        y_test.extend(support_actual[mini_batch])
        # add the probability of the positive class
        probas.extend(support_probas[mini_batch][:, 1])

    for num_update in range(len(query_predicted)):
        for mini_batch in range(len(query_predicted[num_update])):
            y_pred.extend(query_predicted[num_update][mini_batch])
            y_test.extend(query_actual[num_update][mini_batch])
            # add the probability of the positive class
            probas.extend(query_probas[num_update][mini_batch][:, 1])

    risk_df['test_indices'] = list(range(len(y_test)))
    risk_df['y_test'] = y_test
    risk_df['y_pred'] = y_pred
    risk_df['risk_scores'] = probas

    # sort by ascending order of risk score
    risk_df = risk_df.sort_values(by='risk_scores', ascending=False)
    risk_df.to_csv(os.path.join(FLAGS.logdir + '/' + exp_string, 'risk_df.csv'), index=False)


def evaluate(support_predicted, support_actual, query_predicted, query_actual):
    support_accuracies = []
    support_precisions, support_recalls, support_f1s, support_aucs = [], [], [], []

    query_total_accuracies = []
    query_total_precisions, query_total_recalls, query_total_f1s, query_total_aucs = [], [], [], []

    for i in range(len(support_predicted)):
        accuracy, precision, recall, f1score, auc = compute_metrics(predictions=np.int64(support_predicted[i]),
                                                                    labels=np.int64(support_actual[i]))

        support_accuracies.append(accuracy)
        support_precisions.append(precision)
        support_recalls.append(recall)
        support_f1s.append(f1score)
        support_aucs.append(auc)

    support_accuracy = np.mean(support_accuracies)
    support_precision = np.mean(support_precisions)
    support_recall = np.mean(support_recalls)
    support_f1 = np.mean(support_f1s)
    support_auc = np.mean(support_aucs)

    for k in range(len(query_predicted)):
        query_accuracies = []
        query_precisions, query_recalls, query_f1s, query_rocs = [], [], [], []
        mini_batch = query_predicted[k]
        for i in range(len(mini_batch)):
            accuracy, precision, recall, f1score, roc = compute_metrics(predictions=np.int64(query_predicted[k][i]),
                                                                        labels=np.int64(query_actual[k][i]))
            query_accuracies.append(accuracy)
            query_precisions.append(precision)
            query_recalls.append(recall)
            query_f1s.append(f1score)
            query_rocs.append(roc)

        query_total_accuracies.append(np.mean(query_accuracies))
        query_total_precisions.append(np.mean(query_precisions))
        query_total_recalls.append(np.mean(query_recalls))
        query_total_f1s.append(np.mean(query_f1s))
        query_total_aucs.append(np.mean(query_rocs))

    results = {
        'accuracy': [support_accuracy] + query_total_accuracies,
        'precision': [support_precision] + query_total_precisions,
        'recall': [support_recall] + query_total_recalls,
        'f1': [support_f1] + query_total_f1s,
        'roc': [support_auc] + query_total_aucs
    }

    return results


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000

    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            feed_dict = {}
            input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates - 1], model.summ_op]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 600


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []
    metaval_accuracies2, metaval_precisions, metaval_recalls, metaval_f1s, metaval_aucs = [], [], [], [], []

    for _ in range(NUM_TEST_POINTS):

        feed_dict = {}
        feed_dict = {model.meta_lr: 0.0}

        result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        support_predicted, support_actual = sess.run([model.pred1, model.actual1], feed_dict)
        query_predicted, query_actual = sess.run([model.pred2, model.actual2], feed_dict)
        support_probabilities, query_probabilities = sess.run([model.proba1, model.proba2], feed_dict)
        metrics = evaluate(support_predicted, support_actual, query_predicted, query_actual)
        unstack_prediction_probabilities(support_predicted, support_actual, support_probabilities,
                                         query_predicted, query_actual, query_probabilities,
                                         exp_string)

        metaval_accuracies.append(result)
        metaval_accuracies2.append(metrics['accuracy'])
        metaval_precisions.append(metrics['precision'])
        metaval_recalls.append(metrics['recall'])
        metaval_f1s.append(metrics['f1'])
        metaval_aucs.append(metrics['roc'])

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    results_final = {
        'accuracy': metaval_accuracies2,
        'precision': metaval_precisions,
        'recall': metaval_recalls,
        'f1': metaval_f1s,
        'roc': metaval_aucs,
    }

    results_save = {}
    stds_save = {}
    print('\n============================ Results -- Evaluation ============================ ')
    for metric in results_final:
        means = np.mean(results_final[metric], 0)
        stds = np.std(results_final[metric], 0)
        ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

        print('\nMetric: {}'.format(metric))
        print('[support_t0, query_t0 - \t\t\tK] ')
        print('mean:', means)
        print('mean of all {}: {} +- {}'.format(metric, np.mean(means), np.mean(ci95)))
        results_save[metric] = '{:.5f}'.format(np.mean(means))
        stds_save[metric] = '{:.5f}'.format(np.std(means))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

    path_to_save = FLAGS.logdir + '/' + exp_string + '/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save, exist_ok=True)

    # save dictionary of results
    with open(os.path.join(path_to_save, 'error_metrics.p'), 'wb') as f:
        pickle.dump(results_save, f, pickle.HIGHEST_PROTOCOL)

    # save dictionary of standard deviations
    with open(os.path.join(path_to_save, 'std_metrics.p'), 'wb') as f:
        pickle.dump(stds_save, f, pickle.HIGHEST_PROTOCOL)


def main():
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir, exist_ok=True)

    test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    data_generator = DataGenerator(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size)

    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input

    tf_data_load = True
    num_classes = data_generator.num_classes

    if FLAGS.train:  # only construct training model if needed
        random.seed(5)
        image_tensor, label_tensor = data_generator.make_data_tensor()
        inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    random.seed(6)
    image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
    inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
    inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
    labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
    labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'model_{}'.format(FLAGS.model_num)

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    FLAGS.train = True
    train(model, saver, sess, exp_string, data_generator, resume_itr)
    FLAGS.train = False
    test(model, saver, sess, exp_string, data_generator, test_num_updates)


if __name__ == "__main__":
    main()
