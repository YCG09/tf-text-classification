#-*- coding:utf-8 -*-
import json
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.externals import joblib
from .seg_words import text_preprocess, seg_words


def build_vocab(sentences, save_path, vocab_size):
    """
    Build vocabulary, for future use
    """
    all_words = []
    for sentence in sentences:
        all_words.extend(sentence)

    word_counts = Counter(all_words)
    vocabulary = [pairs[0] for pairs in word_counts.most_common(vocab_size - 1)]  # Reserved a position for padding zero
    words_index = {word: index + 1 for index, word in enumerate(vocabulary)}
    with open(save_path + 'words_index.json', 'w', encoding = 'utf-8') as outfile:
        json.dump(words_index, outfile, indent = 4, ensure_ascii=False)

    return words_index


def read_vocab(vocab_path):
    """
    Read vocabulary
    """
    words_index = json.loads(open(vocab_path + 'words_index.json', 'r', encoding = 'utf-8').read())

    return words_index


def pad_sentences(sentences_indexed, sequence_length):
    """
    Pad setences to same length
    """
    sentences_padded = []
    for sentence in sentences_indexed:
        num_padding = sequence_length - len(sentence)
        sentences_padded.append(sentence[:sequence_length - 1] + [0] * max(num_padding, 1))

    return sentences_padded


def process_data(data_file, sentences, labels, sequence_length, vocab_size, mode, output_dir):
    """
    Transform words and labels to index
    """
    # Use the directory of data_file instead of output_dir when the latter is empty
    if output_dir == None:
        output_dir = '/'.join(data_file.split("/")[:-1]) + '/'
    elif not output_dir[-1] == '/':
        output_dir += '/'

    if mode != 'inference':
        sentences = text_preprocess(sentences)
    else:
        sentences = [seg_words(sentences[0])]
    
    words_index = {}
    label_transformer = preprocessing.LabelBinarizer()
    if mode == 'train':
        words_index = build_vocab(sentences, output_dir, vocab_size)
        label_transformer.fit(labels)
        joblib.dump(label_transformer, output_dir + 'label_transformer.pkl')
    else:
        words_index = read_vocab(output_dir)
        label_transformer = joblib.load(output_dir + 'label_transformer.pkl')

    x, y = [], []
    for sentence in sentences:
        x.append([words_index[word] for word in sentence if word in words_index])
    x = pad_sentences(x, sequence_length)
    if len(labels) > 0:
        y = label_transformer.transform(labels)

    return x, y


def load_data(data_file, sequence_length, vocab_size=10000, mode='train', output_dir=None):
    """
    Load data from files
    """
    sentences = []
    labels = []
    data = open(data_file, 'r', encoding='utf-8')

    # In prediction mode, data have no labels
    if mode == 'train' or mode == 'evaluation':
        parts = [line.strip().split("\t", 1) for line in data if len(line.strip().split("\t", 1)) == 2]
        labels, sentences = [elem[0] for elem in parts], [elem[1] for elem in parts]
    elif mode == 'prediction':
        sentences = [line.strip() for line in data if line.strip()]
    else:
        raise ValueError("mode must be train, evaluation or prediction")

    # Convert words and labels
    x, y = process_data(data_file, sentences, labels, sequence_length, vocab_size, mode, output_dir)

    return x, y


def batch_iter(data, batch_size, shuffle=True):
    """
    Generate a batch iterator for dataset
    """
    data_size = len(data)
    data = np.array(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    # Shuflle the data in training
    if shuffle:
        indices_shuffled = np.random.permutation(np.arange(data_size))
        data_shuffled = data[indices_shuffled]
    else:
        data_shuffled = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data_shuffled[start_index:end_index]


def real_len(x_batch):
    """
    Get actual lengths of sequences
    """
    return np.array([list(x).index(0) + 1 for x in x_batch])


if __name__ == "__main__":
    x, y = load_data('../data/train_data', 200, 10000, mode='train')  # train
    # x, y = load_data('../data/test_data', 200, mode='evaluation')  # test
    print("First indexed sentence: {}\nLength of sentence: {}\nFirst onehot label: {}\nLength of label: {}".format(x[0], len(x[0]), y[0], len(y[0])))
