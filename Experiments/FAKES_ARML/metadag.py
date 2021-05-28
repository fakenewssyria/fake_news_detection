import ipdb
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class GraphConvolution(object):

    def __init__(self, hidden_dim, name=None, sparse_inputs=False, act=tf.nn.tanh, bias=True, dropout=0.0):
        self.act = act # tanh
        self.dropout = dropout # 0.0
        self.sparse_inputs = sparse_inputs # False
        self.hidden_dim = hidden_dim # 40
        self.bias = bias # True

        with tf.variable_scope('{}_vars'.format(name), tf.AUTO_REUSE):
            self.gcn_weights = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim], dtype=tf.float32),
                                           name='gcn_weight') # tf.Variable (40, 40)
            if self.bias:
                self.gcn_bias = tf.Variable(tf.constant(0.0, shape=[self.hidden_dim],
                                                        dtype=tf.float32), name='gcn_bias') # (40,)

    def model(self, feat, adj):
        x = feat
        x = tf.nn.dropout(x, 1 - self.dropout)

        node_size = tf.shape(adj)[0]
        I = tf.eye(node_size)
        adj = adj + tf.cast(I, tf.float64)
        D = tf.diag(tf.reduce_sum(adj, axis=1))
        adj = tf.matmul(tf.linalg.inv(D), adj)
        pre_sup = tf.matmul(x, tf.cast(self.gcn_weights, tf.float64))
        output = tf.matmul(adj, pre_sup)

        if self.bias:
            output += tf.cast(self.gcn_bias, tf.float64)
        if self.act is not None:
            return self.act(output)
        else:
            return output

class MetaGraph(object):
    def __init__(self, hidden_dim, input_dim):
        self.input_dim = input_dim # 40
        self.hidden_dim = hidden_dim # 40
        self.proto_num = FLAGS.num_classes # 5
        self.node_cluster_center, self.nodes_cluster_bias = [], []
        for i in range(FLAGS.num_vertex): # (num_vertex is 4, number of vertex in the first layer)
            self.node_cluster_center.append(tf.get_variable(name='{}_node_cluster_center'.format(i),
                                                            shape=(1, input_dim))) # an element of this list is tf.Variable with shape (1, 40)
            self.nodes_cluster_bias.append(
                tf.get_variable(name='{}_nodes_cluster_bias'.format(i), shape=(1, hidden_dim))) # (1, 40)

        self.vertex_num = FLAGS.num_vertex # 4

        self.adj_mlp_weight = tf.Variable(tf.truncated_normal([self.hidden_dim, 1], dtype=tf.float32),
                                          name='adj_mlp_weight') # tf.Variable, shape (1, 40)
        self.adj_mlp_bias = tf.Variable(tf.constant(0.1, shape=[1],
                                                    dtype=tf.float32), name='adj_mlp_bias') # tf.Variable (1,)

        self.GCN = GraphConvolution(self.hidden_dim, name='data_gcn')

    def model(self, inputs): # inputs: (5, 40) the 'squeezed'
        if FLAGS.datasource in ['plainmulti', 'artmulti']:
            sigma = 8.0 # this
        elif FLAGS.datasource in ['2D']:
            sigma = 2.0

        cross_graph = tf.nn.softmax(
            (-tf.reduce_sum(tf.square(inputs - self.node_cluster_center), axis=-1) / (2.0 * sigma)), axis=0) # model/transpose_2:0 (4, 5)
        cross_graph = tf.transpose(cross_graph, perm=[1, 0]) # (5, 4)

        meta_graph = []
        for idx_i in range(self.vertex_num):
            tmp_dist = []
            for idx_j in range(self.vertex_num):
                if idx_i == idx_j:
                    dist = tf.squeeze(tf.zeros([1])) # model/squeeze_1:0, shape: ()
                else:
                    dist = tf.squeeze(tf.sigmoid(tf.layers.dense(
                        tf.abs(self.node_cluster_center[idx_i] - self.node_cluster_center[idx_j]), units=1,
                        name='meta_dist'))) # model/squeeze_2:0, shape: ()
                tmp_dist.append(dist)
            meta_graph.append(tf.stack(tmp_dist))
        meta_graph = tf.stack(meta_graph) # meta graph is a list, each element is model/stack_2:0 shape: (4,)

        proto_graph = [] # meta graph now: model/stack_6:0 shape: (4,4)
        for idx_i in range(self.proto_num):
            tmp_dist = []
            for idx_j in range(self.proto_num):
                if idx_i == idx_j:
                    dist = tf.squeeze(tf.zeros([1]))
                else:
                    dist = tf.squeeze(tf.sigmoid(tf.layers.dense(
                        tf.abs(tf.expand_dims(inputs[idx_i] - inputs[idx_j], axis=0)), units=1, name='proto_dist')))
                tmp_dist.append(tf.cast(dist, tf.float64))
            proto_graph.append(tf.stack(tmp_dist))
        proto_graph = tf.stack(proto_graph) # model/stack_12:0 shape: (5,5)

        adj = tf.concat((tf.concat((proto_graph, cross_graph), axis=1),
                         tf.concat((tf.transpose(cross_graph, perm=[1, 0]), tf.cast(meta_graph, tf.float64)), axis=1)), axis=0) # model/concat_5:0 (9, 9)

        feat = tf.concat((inputs, tf.cast(tf.squeeze(tf.stack(self.node_cluster_center)), tf.float64)), axis=0) # model/concay_6:0 (9, 40)

        repr = self.GCN.model(feat, adj) # model/Tanh:0 (9, 40)

        return repr[0:self.proto_num]
