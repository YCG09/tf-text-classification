#-*- coding:utf-8 -*-
import json
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.externals import joblib


def build_vocab(text, save_path, vocab_size):
    """
    Build vocabulary, for the next use
    """
    all_words = []
    for sentence in text:
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


def process_data(data_file, root_dir, sentences, labels, sequence_length, vocab_size, is_training):
    """
    Transform words and labels to index
    """
    # Use the directory of data_file instead of root_dir when the latter is empty
    if root_dir == None:
        root_dir = '/'.join(data_file.split("/")[:-1]) + '/'
    elif not root_dir[-1] == '/':
        root_dir += '/'

    words_index = {}
    label_transformer = preprocessing.LabelBinarizer()
    if is_training == True:
        words_index = build_vocab(sentences, root_dir, vocab_size)
        label_transformer.fit(labels)
        joblib.dump(label_transformer, root_dir + 'label_transformer.pkl')
    else:
        words_index = read_vocab(root_dir)
        label_transformer = joblib.load(root_dir + 'label_transformer.pkl')

    x, y = [], []
    for sentence in sentences:
        x.append([words_index[word] for word in sentence if word in words_index])
    x = pad_sentences(x, sequence_length)
    # In inference mode, data has no labels
    if len(labels) > 0:
        y = label_transformer.transform(labels)

    return x, y


def load_data(data_file, sequence_length, vocab_size=10000, root_dir=None, has_label=True, is_training=True):
    """
    Load data from files
    """
    sentences = []
    labels = []
    data = open(data_file, 'r', encoding='utf-8')

    # Whether the inference mode
    if has_label==True:
        for parts in [line.strip().split("\t") for line in data]:
            if len(parts) == 2:
                sentences.append(list(parts[0].split(" ")))
                labels.append(parts[1])
    else:
        for sentence in [line.strip() for line in data if line.strip()]:
            sentences.append(list(sentence.split(" ")))

    # Convert words and labels
    x, y = process_data(data_file, root_dir, sentences, labels, sequence_length, vocab_size, is_training)

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
    x, y = load_data('./data/traindata', 200, 10000)  # train
    # x, y = load_data('./data/testdata', 200, is_training=False)  # test
    print("First indexed sentence: {}\nLength of sentence: {}\nFirst onehot label: {}\nLength of label: {}".format(x[0], len(x[0]), y[0], len(y[0])))
