#-*- coding:utf-8 -*-
import os
import re
import time
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


def load_dict(path):
    """
    Load dictionary
    """
    jieba.load_userdict(path + 'default.dic')

    stop_words = open(path + 'stop.dic', 'r', encoding='utf-8').readlines()
    stop_dic = {}.fromkeys([line.strip() for line in stop_words])

    single_words = open(path + 'single.dic', 'r', encoding='utf-8').readlines()
    single_dic = {}.fromkeys([line.strip() for line in single_words])

    synonym_words = open(path + 'synonym.dic', 'r', encoding='UTF-8').readlines()
    synonym_dic = dict([line.strip().split(" ", 1) for line in synonym_words])

    return stop_dic, single_dic, synonym_dic


if os.path.exists('./data/dict/'):
    dict_path = './data/dict/'
else:
    dict_path = '../data/dict/'

stop_dic, single_dic, synonym_dic = load_dict(path=dict_path)


def clean_sentence(sentence):
    """
    Clean sentence
    """
    sentence = re.sub(r'http[s]?://[a-z0-9./?=_-]+', '', sentence.strip().lower())
    sentence = re.sub(r'[0-9_a-z]+([-+.][0-9_a-z]+)*@[0-9_a-z]+([-.][0-9_a-z]+)*\.[0-9_a-z]+([-.][0-9_a-z]+)*', '', sentence)
    sentence = re.sub(r'[0-9]{7,}', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def remove_words(words_list):
    """
    Remove stop words
    """
    words_list = [pair.word for pair in words_list if pair.word not in stop_dic and
                  pair.flag not in ['w', 'ns', 'nr', 't', 'r', 'u', 'e', 'y', 'o']]
    words_list = [word for word in words_list if len(word) >= 2 or word in single_dic]

    return words_list


def replace_words(words_list):
    """
    Replace synonym words
    """
    for idx, word in enumerate(words_list):
        if word in synonym_dic:
            words_list[idx] = synonym_dic[word]
    return words_list


def seg_words(sentence):
    """
    Cut sentence into words
    """
    sentence = clean_sentence(sentence)
    words_list = list(pseg.cut(sentence))
    words_list = remove_words(words_list)
    words_list = replace_words(words_list)

    return words_list


def text_preprocess(sentences):
    """
    Multiprocessing
    """
    pool = Pool(int(cpu_count() / 2))
    text_processed = pool.map(seg_words, sentences)
    pool.close()
    pool.join()
    return text_processed


if __name__ == '__main__':
    # Load data
    data = open('../data/train_data', 'r', encoding='utf-8').readlines()
    parts = np.array([line.strip().split("\t", 1) for line in data])
    labels, sentences = parts[:, 0], parts[:, 1]
    # Word segment
    print("Mission start...")
    start = time.time()
    seg_results = text_preprocess(sentences)
    seg_results = [' '.join(words_list) for words_list in seg_results]
    end = time.time()
    print("Mission complete, it took {:.3f}s".format(end - start))
    # Save result
    save_path = os.path.abspath('../data/prepared_data')
    print("Writing result to {}...".format(save_path))
    df = pd.DataFrame(seg_results, columns=['content'])
    df['label'] = labels
    df.to_csv(save_path, header=False, index=False, sep='\t', encoding='utf-8')
    print("Done!")
