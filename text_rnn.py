#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

class TextRNN(object):
    """
    RNN with Attention mechanism for text classification
    """
    def __init__(self, vocab_size, embedding_size, sequence_length, rnn_size, num_layers,
        attention_size, num_classes, learning_rate, grad_clip):
        """
        - vocab_size : vocabulary size
        - embedding_size: word embedding dimension
        - sequence_length : sequence length after sentence padding
        - rnn_size : hidden layer dimension
        - num_layers : number of rnn layers
        - attention_size : attention layer dimension
        - num_classes : number of target labels
        - learning_rate : initial learning rate
        - grad_clip : gradient clipping threshold
        """

        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Define Forward RNN Cell
        with tf.name_scope('fw_rnn'):
            fw_basic_cell = tf.contrib.rnn.GRUCell(rnn_size)
            # fw_basic_cell = tf.contrib.rnn.LSTMCell(rnn_size)
            fw_rnn_cell = tf.contrib.rnn.MultiRNNCell([fw_basic_cell for _ in range(num_layers)])
            fw_rnn_cell = tf.contrib.rnn.DropoutWrapper(fw_rnn_cell, output_keep_prob=self.keep_prob)

        # Define Backward RNN Cell
        with tf.name_scope('bw_rnn'):
            bw_basic_cell = tf.contrib.rnn.GRUCell(rnn_size)
            # bw_basic_cell = tf.contrib.rnn.LSTMCell(rnn_size)
            bw_rnn_cell = tf.contrib.rnn.MultiRNNCell([bw_basic_cell for _ in range(num_layers)])
            bw_rnn_cell = tf.contrib.rnn.DropoutWrapper(bw_rnn_cell, output_keep_prob=self.keep_prob)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True, name='W')
            # self.input_x shape: (batch_size, sequence_length)
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('bi_rnn'):
            # embedding_inputs shape: (batch_size, sequence_length, embedding_size)
            # rnn_output, _ = tf.nn.dynamic_rnn(fw_rnn_cell, inputs=embedding_inputs, sequence_length=self.seq_len, dtype=tf.float32)
            rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, inputs=embedding_inputs, sequence_length=self.seq_len, dtype=tf.float32)

        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs
        if isinstance(rnn_output, tuple):
            rnn_output = tf.concat(rnn_output, 2)

        # Attention Layer
        with tf.name_scope('attention'):
            input_shape = rnn_output.shape # (batch_size, sequence_length, hidden_size)
            sequence_size = input_shape[1].value  # the length of sequences processed in the RNN layer
            hidden_size = input_shape[2].value  # hidden size of the RNN layer

            attention_w = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.1), name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            attention_u = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name='attention_u')
            z_list = []
            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(rnn_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
                z_list.append(z_t)
            # Transform to batch_size * sequence_size
            attention_z = tf.concat(z_list, axis=1)
            self.alpha = tf.nn.softmax(attention_z)
            # Transform to batch_size * sequence_size * 1 , same rank as rnn_output
            attention_output = tf.reduce_sum(rnn_output * tf.reshape(self.alpha, [-1, sequence_size, 1]), 1)

        # Add dropout
        with tf.name_scope('dropout'):
            # attention_output shape: (batch_size, hidden_size)
            self.final_output = tf.nn.dropout(attention_output, self.keep_prob)

        # Fully connected layer
        with tf.name_scope('output'):
            fc_w = tf.Variable(tf.truncated_normal([hidden_size, num_classes], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.predictions = tf.argmax(self.logits, 1, name='predictions')

        # Calculate cross-entropy loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        # Create optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
            self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    model = TextRNN(vocab_size=8000, embedding_size=150, sequence_length=100, rnn_size=100, num_layers=2,
            attention_size=50, num_classes=30, learning_rate=0.001, grad_clip=5.0)
