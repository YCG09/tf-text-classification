#-*- coding:utf-8 -*-
import os
import time
import json
import warnings
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import preprocess.data_helpers as data_helpers
from text_classifier.text_cnn import TextCNN

warnings.filterwarnings("ignore")

# Parameters
# ==================================================

# Data loading parameters
tf.flags.DEFINE_string('data_file', './data/train_data', "Data source for the text data")
tf.flags.DEFINE_integer('num_classes', None, "Number classes of labels in data")
tf.flags.DEFINE_float('test_size', 0.05, "Percentage of data to use for validation and test (default: 0.05)")
tf.flags.DEFINE_integer('vocab_size', 9000, "Select words to build vocabulary, according to term frequency (default: 9000)")
tf.flags.DEFINE_integer('sequence_length', 500, "Padding sentences to same length, cut off when necessary (default: 500)")

# Model hyperparameters
tf.flags.DEFINE_integer('embedding_size', 128, "Dimension of word embedding (default: 128)")
tf.flags.DEFINE_string('filter_sizes', '3,4,5', "Filter sizes to use in convolution layer, comma-separated (default: '3,4,5')")
tf.flags.DEFINE_integer('num_filters', 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float('learning_rate', 0.001, "Learning rate for model training (default: 0.001)")
tf.flags.DEFINE_float('grad_clip', 3.0, "Gradients clipping threshold (default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer('batch_size', 128, "Batch size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")
tf.flags.DEFINE_string('init_embedding_path', None, "Using pre-trained word embedding, npy file format")
tf.flags.DEFINE_string('init_model_path', None, "Continue training from saved model at this path")
tf.flags.DEFINE_integer('evaluate_every', 50, "Evaluate model on val set after this many steps (default: 50)")

# Tensorflow parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, "Allow device soft device placement (default: True)")
tf.flags.DEFINE_boolean('log_device_placement', False, "Log placement of ops on devices (default: False)")
tf.flags.DEFINE_boolean('gpu_allow_growth', True, "GPU memory allocation mode (default: True)")

FLAGS = tf.flags.FLAGS
print("\nParameters:")
for param, value in sorted(FLAGS.flag_values_dict().items()):
    print("{} = {}".format(param.upper(), value))
print("")


def train_cnn():
    # Data Preparation
    # ==================================================

    if FLAGS.init_embedding_path is not None:
        embedding = np.load(FLAGS.init_embedding_path)
        print("Using pre-trained word embedding which shape is {}\n".format(embedding.shape))
        FLAGS.vocab_size = embedding.shape[0]
        FLAGS.embedding_size = embedding.shape[1]

    if FLAGS.init_model_path is not None:
        assert os.path.isdir(FLAGS.init_model_path), "init_model_path must be a directory\n"
        ckpt = tf.train.get_checkpoint_state(FLAGS.init_model_path)
        assert ckpt, "No checkpoint found in {}\n".format(FLAGS.init_model_path)
        assert ckpt.model_checkpoint_path, "No model_checkpoint_path found in checkpoint\n"

    # Create output directory for models and summaries
    timestamp = str(int(time.time()))
    output_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', 'textcnn', 'trained_result_' + timestamp))
    os.makedirs(output_dir)

    # Load data
    print("Prepareing data...\n")
    data = os.path.abspath(FLAGS.data_file)
    x, y = data_helpers.load_data(data, FLAGS.sequence_length, FLAGS.vocab_size, mode='train', output_dir=output_dir)
    FLAGS.num_classes = len(y[0])

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=FLAGS.test_size, stratify=y, random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

    # Training
    # ==================================================
    with tf.Graph().as_default():
        tf_config = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        tf_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth

        with tf.Session(config=tf_config).as_default() as sess:
            cnn = TextCNN(
                vocab_size=FLAGS.vocab_size,
                embedding_size=FLAGS.embedding_size,
                sequence_length=FLAGS.sequence_length,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
            	num_classes=FLAGS.num_classes,
                learning_rate=FLAGS.learning_rate,
                grad_clip=FLAGS.grad_clip,
            	l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Summaries for loss and accuracy
            tf.summary.scalar("loss", cnn.loss)
            tf.summary.scalar("accuracy", cnn.accuracy)
            merged_summary = tf.summary.merge_all()

            # Summaries dictionary
            print("Writing to {}...\n".format(output_dir))
            train_summary_dir = os.path.join(output_dir, 'summaries', 'train')
            val_summary_dir = os.path.join(output_dir, 'summaries', 'val')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Checkpoint directory, will not create itself
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model.ckpt')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Using pre-trained word embedding
            if FLAGS.init_embedding_path is not None:
                sess.run(cnn.embedding.assign(embedding))
                del embedding

            # Continue training from saved model
            if FLAGS.init_model_path is not None:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Training start
            print("Start training...\n")
            best_at_step = 0
            best_val_accuracy = 0
            for epoch in range(FLAGS.num_epochs):
                # Generate train batches
                train_batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size)
                start = time.time()
                for batch in train_batches:
                    # Training model on x_batch and y_batch
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.keep_prob: FLAGS.dropout_keep_prob, cnn.is_training: True}
                    _, global_step, train_summaries, train_loss, train_accuracy = sess.run([cnn.train_op, cnn.global_step,
                        merged_summary, cnn.loss, cnn.accuracy], feed_dict=feed_dict)

                    # Evaluates model on val set
                    if global_step % FLAGS.evaluate_every == 0:
                        end = time.time()
                        train_summary_writer.add_summary(train_summaries, global_step)
                        feed_dict = {cnn.input_x: x_val, cnn.input_y: y_val, cnn.keep_prob: FLAGS.dropout_keep_prob, cnn.is_training: False}
                        val_summaries, val_loss, val_accuracy = sess.run([merged_summary, cnn.loss, cnn.accuracy], feed_dict=feed_dict)
                        val_summary_writer.add_summary(val_summaries, global_step)
                        print("Epoch: {}, global step: {}, training speed: {:.3f}sec/batch".format(epoch,
                            global_step, (end - start) / FLAGS.evaluate_every))
                        print("train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f}\n".format(train_loss,
                            train_accuracy, val_loss, val_accuracy))
                        # If improved, save the model
                        if val_accuracy > best_val_accuracy:
                            print("Get a best val accuracy at step {}, model saving...\n".format(global_step))
                            saver.save(sess, checkpoint_prefix, global_step=global_step)
                            best_val_accuracy = val_accuracy
                            best_at_step = global_step
                        start = time.time()

            # Rename the checkpoint
            best_model_prefix = checkpoint_prefix + '-' + str(best_at_step)
            os.rename(best_model_prefix + '.index', os.path.join(checkpoint_dir, 'best_model.index'))
            os.rename(best_model_prefix + '.meta', os.path.join(checkpoint_dir, 'best_model.meta'))
            os.rename(best_model_prefix + '.data-00000-of-00001', os.path.join(checkpoint_dir, 'best_model.data-00000-of-00001'))

            # Testing on test set
            print("\nTraining complete, testing the best model on test set...\n")
            saver.restore(sess, os.path.join(checkpoint_dir, 'best_model'))
            feed_dict = {cnn.input_x: x_test, cnn.input_y: y_test, cnn.keep_prob: FLAGS.dropout_keep_prob, cnn.is_training: False}
            y_logits, test_accuracy = sess.run([cnn.logits, cnn.accuracy], feed_dict=feed_dict)
            print("Testing Accuracy: {:.3f}\n".format(test_accuracy))
            label_transformer = joblib.load(os.path.join(output_dir, 'label_transformer.pkl'))
            y_test_original = label_transformer.inverse_transform(y_test)
            y_logits_original = label_transformer.inverse_transform(y_logits)
            print("Precision, Recall and F1-Score:\n\n", classification_report(y_test_original, y_logits_original))

            # Save parameters
            print("Parameters saving...\n")
            params = {}
            for param, value in FLAGS.flag_values_dict().items():
                params[param] = value
            with open(os.path.join(output_dir, 'parameters.json'), 'w') as outfile:
                json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

            # Save word embedding
            print("Word embedding saving...\n")
            np.save(os.path.join(output_dir, 'embedding.npy'), sess.run(cnn.embedding))


if __name__ == '__main__':
    train_cnn()
