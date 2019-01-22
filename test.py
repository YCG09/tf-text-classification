#-*- coding:utf-8 -*-
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import preprocess.data_helpers as data_helpers
from text_classifier.text_cnn import TextCNN
from text_classifier.text_rnn import TextRNN

warnings.filterwarnings("ignore")

# Parameters
# ==================================================

# Data loading parameters
tf.flags.DEFINE_string('test_data', './data/test_data', "Data source for the test data")
tf.flags.DEFINE_string('test_mode', 'evaluation', "In evaluation mode, the data should have labels, \
                        which is contrary to that of prediction mode (default: evaluation)")

# Evaluating parameters
tf.flags.DEFINE_integer('batch_size', 256, "Batch size (default: 256)")
tf.flags.DEFINE_string('model_type', None, "The type of model used to evaluate, CNN or RNN")
tf.flags.DEFINE_string('checkpoint_dir', None, "The directory of checkpoints")

# Tensorflow parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, "Allow device soft device placement (default: True)")
tf.flags.DEFINE_boolean('log_device_placement', False, "Log placement of ops on devices (default: False)")
tf.flags.DEFINE_boolean('gpu_allow_growth', True, "GPU memory allocation mode (default: True)")

FLAGS = tf.flags.FLAGS


def test():
    if FLAGS.checkpoint_dir == None or not os.path.exists(FLAGS.checkpoint_dir):
        raise IOError("checkpoint_dir not found")

    if FLAGS.model_type == None or not FLAGS.model_type in ['CNN', 'RNN']:
        raise ValueError("model_type must be CNN or RNN")

    # Load parameters
    print("Loading parameters...\n")
    base_dir = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, '..')) + '/'
    params = json.loads(open(base_dir + 'parameters.json').read())

    # Load data
    print("Prepareing data...\n")
    test_data = os.path.abspath(FLAGS.test_data)
    x_test, y_test = data_helpers.load_data(test_data, params['sequence_length'], mode=FLAGS.test_mode, output_dir=base_dir)

    # Testing
    # ==================================================
    with tf.Graph().as_default():
        tf_config = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        tf_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth

        with tf.Session(config=tf_config).as_default() as sess:
            # Model initialization
            if FLAGS.model_type == 'CNN':
                model = TextCNN(
                    vocab_size=params['vocab_size'],
                    embedding_size=params['embedding_size'],
                    sequence_length=params['sequence_length'],
                    filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                    num_filters=params['num_filters'],
                    num_classes=params['num_classes'],
                    learning_rate=params['learning_rate'],
                    grad_clip=params['grad_clip'],
                    l2_reg_lambda=params['l2_reg_lambda'])
                feed_dict = {model.keep_prob: 1.0, model.is_training: False}

            elif FLAGS.model_type == 'RNN':
                model = TextRNN(
                    vocab_size=params['vocab_size'],
                    embedding_size=params['embedding_size'],
                    sequence_length=params['sequence_length'],
                    rnn_size=params['rnn_size'],
                    num_layers=params['num_layers'],
                    attention_size=params['attention_size'],
                    num_classes=params['num_classes'],
                    learning_rate=params['learning_rate'],
                    grad_clip=params['grad_clip'])
                feed_dict = {model.keep_prob: 1.0}

            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())

            # Restore all variables from checkpoint
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                best_model_path = os.path.join('/'.join(ckpt.model_checkpoint_path.split("/")[:-1]), 'best_model')
                saver.restore(sess, best_model_path)
            else:
                raise ValueError("Check model_checkpoint_path in checkpoint file")

            # Testing on batch
            print("Start testing...\n")
            y_logits = []
            start = time.time()
            data_size = len(x_test)
            # Generate test batches
            test_batches = data_helpers.batch_iter(x_test, FLAGS.batch_size, shuffle=False)
            for x_batch in test_batches:
                feed_dict[model.input_x] = x_batch
                if FLAGS.model_type == 'RNN':
                    feed_dict[model.seq_len] = data_helpers.real_len(x_batch)
                batch_predictions = sess.run(model.logits, feed_dict=feed_dict)
                y_logits.extend(batch_predictions)
            print("Mission complete, total number of test examples: {}, testing speed: {:.0f} examples/sec\n".format(
                data_size, data_size / (time.time() - start)))
            label_transformer = joblib.load(os.path.join(base_dir, 'label_transformer.pkl'))
            y_logits_original = label_transformer.inverse_transform(np.array(y_logits))

            # Print accuracy if test examples have labels
            if FLAGS.test_mode == 'evaluation':
                columns = ['real_label', 'content']
                df = pd.DataFrame([line.strip().split("\t", 1) for line in open(test_data, 'r', encoding='utf-8').readlines()
                    if len(line.strip().split("\t", 1)) == 2], columns=columns)[list(reversed(columns))]
                y_test_original = label_transformer.inverse_transform(y_test)
                test_accuracy = sum(y_logits_original == y_test_original) / data_size
                print("Testing Accuracy: {:.3f}\n".format(test_accuracy))
                print("Precision, Recall and F1-Score:\n\n", classification_report(y_test_original, y_logits_original))
            else:
                df = pd.DataFrame([line.strip() for line in open(test_data, 'r', encoding='utf-8').readlines()
                    if line.strip()], columns=['content'])

            # Save result
            timestamp = str(int(time.time()))
            output_dir = os.path.join(base_dir, 'test')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            save_path = os.path.join(output_dir, 'test_result_' + timestamp + '.csv')
            df['predicted_label'] = y_logits_original
            print("Writing result to {}...\n".format(save_path))
            df.to_csv(save_path, header=True, index=False, sep='\t', encoding='utf-8')


if __name__ == '__main__':
    test()
