import random

import csv
import numpy as np
import tensorflow as tf

from maml import MAML
import os, pickle
import pandas as pd
from sklearn.metrics import *

tf.set_random_seed(1234)
from data_generator import DataGenerator
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'plainmulti', '2D or plainmulti or artmulti')
flags.DEFINE_integer('test_dataset', -1, 'which data to be test, plainmulti: 0-3, artmulti: 0-11, -1: random select')
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_test_task', 600, 'number of test tasks.')
flags.DEFINE_integer('test_epoch', -1, 'test epoch, only work when test start')

## Dataset - I/O related
flags.DEFINE_string('training_data_path', 'input/feature_extraction_train_updated.csv', 'path to training data')
flags.DEFINE_string('testing_data_path', 'input/feature_extraction_test_updated.csv', 'path to testing data')
flags.DEFINE_string('target_variable', 'label', 'name of the target variable column')
flags.DEFINE_list('cols_drop', ['article_title', 'article_content', 'source', 'source_category', 'unit_id'], 'list of column to drop from data, if any')
flags.DEFINE_string('special_encoding', 'latin-1', 'special encoding needed to read the data, if any')
flags.DEFINE_string('scaling', 'z-score', 'scaling done to the dataset, if any')

## Training options
flags.DEFINE_integer('metatrain_iterations', 1000, 'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 32, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.1, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 32, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_integer('update_batch_size_eval', 10, 'number of examples used for inner gradient test (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-1, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 4, 'number of inner gradient updates during training.')
flags.DEFINE_integer('num_updates_test', 20, 'number of inner gradient updates during training.')
flags.DEFINE_integer('sync_group_num', 6, 'the number of different groups in sync dataset')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, or None')
flags.DEFINE_integer('hidden_dim', 40, 'output dimension of task embedding')
flags.DEFINE_integer('num_filters', 64, '32 for plainmulti and artmulti')
flags.DEFINE_integer('sync_filters', 40, 'number of dim when combine sync functions.')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_float('emb_loss_weight', 0.0, 'the weight of autoencoder')
flags.DEFINE_integer('task_embedding_num_filters', 32, 'number of filters for task embedding')
flags.DEFINE_integer('num_vertex', 4, 'number of vertex in the first layer')
# flags.DEFINE_integer('num_vertex', 6, 'number of vertex in the first layer')

flags.DEFINE_string('dim_hidden', '128', 'number of neurons in each hidden layer')
flags.DEFINE_string('dim_name', 'dim0', 'unique index name for the list of hidden layers (above)')
flags.DEFINE_string('activation_fn', 'relu', 'activation function used')
flags.DEFINE_integer('model_num', 1, 'model number to store trained model. Better for tracking')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'fake_news/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('test_set', True, 'Set to true to evaluate on the the test set, False for the validation set.')


def compute_metrics(predictions, labels):
	'''compute metrics - regular and cos sensitive  '''
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

    # 'test_indices', 'y_test', 'y_pred', 'risk_scores'
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
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 100

    print('Done initializing, starting training.')

    prelosses, postlosses, embedlosses = [], [], []

    num_classes = data_generator.num_classes

    for itr in range(resume_itr, FLAGS.metatrain_iterations):
        feed_dict = {}
        if FLAGS.datasource == '2D':
            batch_x, batch_y, para_func, sel_set = data_generator.generate_2D_batch()

            inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
            labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}

        input_tensors = [model.metatrain_op, model.total_embed_loss, model.total_loss1,
                         model.total_losses2[FLAGS.num_updates - 1]]
        if model.classification:
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]])

        result = sess.run(input_tensors, feed_dict)

        prelosses.append(result[-2])
        postlosses.append(result[-1])
        embedlosses.append(result[2])

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration {}'.format(itr)
            std = np.std(postlosses, 0)
            ci95 = 1.96 * std / np.sqrt(PRINT_INTERVAL)
            print_str += ': preloss: ' + str(np.mean(prelosses)) + ', postloss: ' + str(
                np.mean(postlosses)) + ', embedding loss: ' + str(np.mean(embedlosses)) + ', confidence: ' + str(ci95)

            print(print_str)
            prelosses, postlosses, embedlosses = [], [], []

        if (itr != 0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir, exist_ok=True)
    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


def test(model, sess, exp_string, data_generator):
    num_classes = data_generator.num_classes

    metaval_accuracies = []
    metaval_accuracies2, metaval_precisions, metaval_recalls, metaval_f1s, metaval_aucs = [], [], [], [], []

    print(FLAGS.num_test_task)

    for test_itr in range(FLAGS.num_test_task):
        if FLAGS.datasource == '2D':
            batch_x, batch_y, para_func, sel_set = data_generator.generate_2D_batch()

            inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
            labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                         model.meta_lr: 0.0}
        else:
            feed_dict = {model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
            support_predicted, support_actual = sess.run([model.pred1, model.actual1], feed_dict)
            query_predicted, query_actual = sess.run([model.pred2, model.actual2], feed_dict)
            support_probabilities, query_probabilities = sess.run([model.proba1, model.proba2], feed_dict)
            metrics = evaluate(support_predicted, support_actual, query_predicted, query_actual)
            unstack_prediction_probabilities(support_predicted, support_actual, support_probabilities,
                                             query_predicted, query_actual, query_probabilities,
                                             exp_string)
        else:
            result = sess.run([model.metaval_total_loss1] + model.metaval_total_losses2, feed_dict)

        metaval_accuracies.append(result)
        metaval_accuracies2.append(metrics['accuracy'])
        metaval_precisions.append(metrics['precision'])
        metaval_recalls.append(metrics['recall'])
        metaval_f1s.append(metrics['f1'])
        metaval_aucs.append(metrics['roc'])

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(FLAGS.num_test_task)

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
        ci95 = 1.96 * stds / np.sqrt(FLAGS.num_test_task)

        print('\nMetric: {}'.format(metric))
        print('[support_t0, query_t0 - \t\t\tK] ')
        print('mean:', means)
        # print('stds:', stds)
        # print('ci95:', ci95)
        print('mean of all {}: {} +- {}'.format(metric, np.mean(means), np.mean(ci95)))
        results_save[metric] = '{:.5f}'.format(np.mean(means))
        stds_save[metric] = '{:.5f}'.format(np.std(means))

    out_filename = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(
        FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(
        FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update' + str(i) for i in range(len(means))])
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
    sess = tf.InteractiveSession()
    if FLAGS.train:
        test_num_updates = FLAGS.num_updates # test_num_updates: 1
    else:
        test_num_updates = FLAGS.num_updates_test

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        FLAGS.meta_batch_size = 1

    data_generator = DataGenerator(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size)

    dim_output = data_generator.dim_output # dim_output: 5
    dim_input = data_generator.dim_input # dim_input: 21168

    if FLAGS.datasource in ['plainmulti', 'artmulti']:
        num_classes = data_generator.num_classes # num_classes 5
        if FLAGS.train:
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1]) # (25, 25, 21168)
            inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1]) # (25, 75, 21168)
            labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1]) # (25, 25, 5)
            labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1]) # (25, 75, 5)
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        input_tensors = None
        metaval_input_tensors=None

    model = MAML(sess, dim_input, dim_output, test_num_updates=test_num_updates)

    if FLAGS.train:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    # else:
    #     model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=60)

    if FLAGS.train == False:
        FLAGS.meta_batch_size = orig_meta_batch_size

    exp_string = 'model_{}'.format(FLAGS.model_num)

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        # model_file = '{0}/{2}/model{1}'.format(FLAGS.logdir, FLAGS.test_epoch, exp_string)
        model_file = '{}/{}/model{}'.format(FLAGS.logdir, exp_string, FLAGS.metatrain_iterations - 1)
        if model_file:
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
    resume_itr = 0

    FLAGS.train = True
    train(model, saver, sess, exp_string, data_generator, resume_itr)
    FLAGS.train = False
    test(model, sess, exp_string, data_generator)


if __name__ == "__main__":
    main()
