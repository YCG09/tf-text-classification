#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    CNN for text classification
    """
    def __init__(self, vocab_size, embedding_size, sequence_length, filter_sizes, num_filters,
            num_classes, learning_rate, grad_clip, l2_reg_lambda=0.0):
        """
        - vocab_size : vocabulary size
        - embedding_size: word embedding dimension
        - sequence_length : sequence length after sentence padding
        - filter_sizes : comma-separated filter sizes
        - num_filters : number of filters per filter size
        - num_classes : number of target labels
        - learning_rate : initial learning rate
        - grad_clip : gradient clipping threshold
        - l2_reg_lambda : l2 regularization lambda
        """

        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True, name='W')
            # self.input_x shape: (batch_size, sequence_length)
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            embedding_inputs_expanded = tf.expand_dims(embedding_inputs, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size) as scope:
                # Convolution Layer, embedding_inputs_expanded shape: (batch_size, sequence_length, embedding_size, 1)
                # filter_shape: (filter_size, embedding_size, 1, num_filters)
                conv = tf.layers.conv2d(embedding_inputs_expanded, filters=num_filters, kernel_size=[filter_size, embedding_size],
                        strides=[1, 1], padding='valid', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        bias_initializer=tf.constant_initializer(0.1), name=scope+'conv')
                # Maxpooling Layer, conv shape: (batch_size, sequence_length - filter_size + 1, 1, num_filters)
                # N = (W-F+2P)/S+1
                pooled = tf.layers.max_pooling2d(conv, pool_size=[sequence_length - filter_size + 1, 1], strides=[1, 1], padding='valid', name=scope+'pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features, pooled shape: (batch_size, 1, 1, num_filters)
        pooled_concat = tf.concat(pooled_outputs, 3)
        # pooled_concat shape: (batch_size, 1, 1, num_filters * len(filter_sizes))
        pooled_concat_flat = tf.squeeze(pooled_concat, [1, 2])

        # Add dropout
        with tf.name_scope('dropout'):
            # pooled_concat_flat shape: (batch_size, num_filters * len(filter_sizes))
            self.final_output = tf.contrib.layers.dropout(pooled_concat_flat, keep_prob=self.keep_prob, is_training=self.is_training)

        # Fully connected layer
        with tf.name_scope('output'):
            fc_w = tf.get_variable('fc_w', shape=[self.final_output.shape[1].value, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.predictions = tf.argmax(self.logits, 1, name='predictions')

        # Calculate cross-entropy loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            l2_loss += tf.nn.l2_loss(fc_w)
            l2_loss += tf.nn.l2_loss(fc_b)
            self.loss = tf.reduce_mean(cross_entropy) + l2_reg_lambda * l2_loss

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
    model = TextCNN(vocab_size=8000, embedding_size=150, sequence_length=100, filter_sizes=list(map(int, "3,4,5".split(","))), num_filters=128,
            num_classes=30, learning_rate=0.001, grad_clip=5.0, l2_reg_lambda=0.01)
