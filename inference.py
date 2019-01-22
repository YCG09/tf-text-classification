#-*- coding:utf-8 -*-
import os
import time
import json
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
import preprocess.data_helpers as data_helpers
from text_classifier.text_cnn import TextCNN
from text_classifier.text_rnn import TextRNN

# Parameters
# ==================================================
tf.flags.DEFINE_string('model_type', None, "The type of model used to evaluate, CNN or RNN")
tf.flags.DEFINE_string('checkpoint_dir', None, "The directory of checkpoints")

FLAGS = tf.flags.FLAGS


class Model:
    def __init__(self):
        if FLAGS.checkpoint_dir == None or not os.path.exists(FLAGS.checkpoint_dir):
            raise IOError("checkpoint_dir not found")

        if FLAGS.model_type == None or not FLAGS.model_type in ['CNN', 'RNN']:
            raise ValueError("model_type must be CNN or RNN")

        self.output_dir = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, '..')) + '/'

        # Load parameters
        print("Loading parameters...\n")
        self.params = json.loads(open(self.output_dir + 'parameters.json').read())
        self.label_transformer = joblib.load(os.path.join(self.output_dir, 'label_transformer.pkl'))

        # Model initialization
        if FLAGS.model_type == 'CNN':
            self.model = TextCNN(
                vocab_size=self.params['vocab_size'],
                embedding_size=self.params['embedding_size'],
                sequence_length=self.params['sequence_length'],
                filter_sizes=list(map(int, self.params['filter_sizes'].split(","))),
                num_filters=self.params['num_filters'],
                num_classes=self.params['num_classes'],
                learning_rate=self.params['learning_rate'],
                grad_clip=self.params['grad_clip'],
                l2_reg_lambda=self.params['l2_reg_lambda'])

        elif FLAGS.model_type == 'RNN':
             self.model = TextRNN(
                vocab_size=self.params['vocab_size'],
                embedding_size=self.params['embedding_size'],
                sequence_length=self.params['sequence_length'],
                rnn_size=self.params['rnn_size'],
                num_layers=self.params['num_layers'],
                attention_size=self.params['attention_size'],
                num_classes=self.params['num_classes'],
                learning_rate=self.params['learning_rate'],
                grad_clip=self.params['grad_clip'])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Restore all variables from checkpoint
        print("Loading model...\n")
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            best_model_path = os.path.join('/'.join(ckpt.model_checkpoint_path.split("/")[:-1]), 'best_model')
            saver.restore(self.sess, best_model_path)
        else:
            raise ValueError("Check model_checkpoint_path in checkpoint file")


    def inference(self, sentence):
        labels = []
        sentences = [sentence.strip()]
        x, y = data_helpers.process_data(None, sentences, labels, self.params['sequence_length'], \
                        self.params['vocab_size'], 'inference', self.output_dir)
        
        feed_dict = {
            self.model.keep_prob: 1.0,
            self.model.input_x: x
        }

        if FLAGS.model_type == 'CNN':
            feed_dict[self.model.is_training] = False
        elif FLAGS.model_type == 'RNN':
            feed_dict[self.model.seq_len] = data_helpers.real_len(x)

        y_pred, y_pred_softmax = self.sess.run([self.model.logits, self.model.logits_softmax], feed_dict=feed_dict)
        y_pred_original = self.label_transformer.inverse_transform(np.array(y_pred))[0]
        probability = np.max(y_pred_softmax)

        return y_pred_original, probability


if __name__ == '__main__':
    model = Model()
    welcome = 'Input sentence, enter q exit:\n'
    while True:
        get_str = input(welcome)
        if get_str == 'q':
            break
        begin_time = time.time()
        inference_result = model.inference(get_str)
        print("Inference label: {}, probability: {:.3f}, time spent: {:.3f}s".format(inference_result[0], inference_result[1], time.time() - begin_time))
