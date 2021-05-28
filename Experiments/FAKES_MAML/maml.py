""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import xent, conv_block, normalize
FLAGS = flags.FLAGS

# dictionary of activation functions to use in hyper parameter tuning
activations = {
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
    'softmax': tf.nn.softmax,
    'swish': tf.nn.swish
}


class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.test_num_updates = test_num_updates
        self.classification = True
        self.loss_func = xent

        self.dim_hidden = list(map(int, list(FLAGS.dim_hidden.split(", "))))
        self.forward = self.forward_fc
        self.construct_weights = self.construct_fc_weights

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float64)
            self.inputb = tf.placeholder(tf.float64)
            self.labela = tf.placeholder(tf.float64)
            self.labelb = tf.placeholder(tf.float64)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                # add actual & predicted for evaluation
                query_pred_evals, query_actual_evals = [], []
                # add prediction probabilities for advanced evaluation
                query_proba_evals = []

                if self.classification:
                    task_accuraciesb = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                # task_outputa are the logits
                # logits are the values of the last layer
                # before applying softmax to them (before turning them into probabilities)
                # for cost sensitive, multiply logits
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1),
                                                                 tf.argmax(labela, 1))
                    # add actual & predicted for evaluation
                    support_pred_eval = tf.argmax(tf.nn.softmax(task_outputa), 1)
                    support_actual_eval = tf.argmax(labela, axis=1)
                    # add prediction probabilities for advanced evaluation
                    support_proba_eval = tf.nn.softmax(task_outputa)

                    for j in range(num_updates):
                        task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                        # add actual & predicted for evaluation
                        query_pred_evals.append(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1))
                        query_actual_evals.append(tf.argmax(labelb, 1))
                        # add prediction probabilities for advanced evaluation
                        query_proba_evals.append(tf.nn.softmax(task_outputbs[j]))

                    task_output.extend([task_accuracya, task_accuraciesb])
                    # add actual & predicted for evaluation
                    task_output.extend([support_pred_eval, support_actual_eval])
                    task_output.extend([query_pred_evals, query_actual_evals])
                    # add prediction probabilities for advanced evaluation
                    task_output.extend([support_proba_eval, query_proba_evals])

                # return task_output

                task_output_mod = []
                for out in task_output:
                    if isinstance(out, list):
                        out_mod = []
                        for sub in out:
                            out_mod.append(tf.cast(sub, tf.float64))
                        task_output_mod.append(out_mod)
                    else:
                        task_output_mod.append(tf.cast(out, tf.float64))

                return task_output_mod

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float64, [tf.float64]*num_updates, tf.float64, [tf.float64]*num_updates]
            if self.classification:
                out_dtype.extend([tf.float64, [tf.float64]*num_updates])
                # add actual & predicted for evaluation
                out_dtype.extend([tf.float64, tf.float64])
                out_dtype.extend([[tf.float64]*num_updates, [tf.float64]*num_updates])
                # add prediction probabilities for advanced evaluation
                out_dtype.extend([tf.float64, [tf.float64] * num_updates])

            # Hiyam adding the 2 lines below
            self.labela = tf.cast(self.labela, tf.float64)
            self.labelb = tf.cast(self.labelb, tf.float64)

            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb,\
                predsa, actualsa, predsb, actualsb, \
                probasa, probasb = result
            else:
                outputas, outputbs, lossesa, lossesb  = result

        ## Performance & Optimization
        meta_batch_siz64 = tf.cast(tf.to_float(FLAGS.meta_batch_size), tf.float64)
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / meta_batch_siz64
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / meta_batch_siz64 for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / meta_batch_siz64
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / meta_batch_siz64 for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                # if FLAGS.datasource == 'miniimagenet':
                #     gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / meta_batch_siz64
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / meta_batch_siz64 for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / meta_batch_siz64
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / meta_batch_siz64 for j in range(num_updates)]

                # Hiyam adding outputs here:
                self.outputas = outputas
                self.outputbs = outputbs

                self.pred1 = predsa
                self.actual1 = actualsa
                self.pred2 = predsb
                self.actual2 = actualsb
                self.proba1 = probasa
                self.proba2 = probasb

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        ac_fn = activations[FLAGS.activation_fn]
        for key in weights:
            weights[key] = tf.cast(weights[key], tf.float64)
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=ac_fn, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = tf.cast(hidden, tf.float64)
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=ac_fn, reuse=reuse, scope=str(i+1))
            hidden = tf.cast(hidden, tf.float64)
        if len(self.dim_hidden) == 1:
            hidden = tf.cast(hidden, tf.float64)
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]


