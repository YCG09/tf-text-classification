#-*- coding:utf-8 -*-
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

    return stop_dic, single_dic


def clean_sentence(sentence):
    """
    Cleaning sentence
    """
    sentence = re.sub(r'http[s]?://[a-z0-9./?=_-]+', '', sentence.strip().lower())
    sentence = re.sub(r'[0-9_a-z]+([-+.][0-9_a-z]+)*@[0-9_a-z]+([-.][0-9_a-z]+)*\.[0-9_a-z]+([-.][0-9_a-z]+)*', '', sentence)
    sentence = re.sub(r'[0-9]{7,}', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def remove_words(words_list):
    words_list = [pair.word for pair in words_list if pair.word not in stop_dic and
                  pair.flag not in ['w', 'ns', 'nr', 't', 'r', 'u', 'e', 'y', 'o']]
    words_list = [word for word in words_list if len(word) >= 2 or word in single_dic]

    return words_list


def seg_words(sentence):
    """
    Cut sentence into words
    """
    sentence = clean_sentence(sentence)
    words_list = list(pseg.cut(sentence))
    words_list = remove_words(words_list)

    return ' '.join(words_list)


if __name__ == '__main__':
    start = time.time()
    stop_dic, single_dic = load_dict(path='./data/dict/')
    # Load data
    data = open('./data/rawdata', 'r', encoding='utf-8').readlines()
    parts = np.array([line.strip().split("\t", 1) for line in data])
    labels, sentences = parts[:, 0], parts[:, 1]
    # Create process pool
    pool = Pool(cpu_count())
    print("Mission start...")
    seg_results = pool.map(seg_words, sentences)
    pool.close()
    pool.join()
    end = time.time()
    print("Mission complete, it took {:.3f}s".format(end - start))
    # Save result
    save_path = './data/traindata'
    print("Writing result to {}...".format(save_path))
    df = pd.DataFrame(seg_results, columns=['content'])
    df['label'] = labels
    df.to_csv(save_path, header=False, index=False, sep='\t', encoding='utf-8')
    print("Done!")
