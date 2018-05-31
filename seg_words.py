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

    stopWords = open(path + 'stop.dic', 'r', encoding='UTF-8').readlines()
    stop = {}.fromkeys([line.strip() for line in stopWords])

    singleWord = open(path + 'single.dic', 'r', encoding='UTF-8').readlines()
    single = {}.fromkeys([line.strip() for line in singleWord])

    return stop, single

def clean_str(string):
    """
    Cleaning sentence
    """
    string = re.sub(r'http[s]?://[a-z0-9./?=_-]+', '', string)
    string = re.sub(r'[0-9_a-z]+([-+.][0-9_a-z]+)*@[0-9_a-z]+([-.][0-9_a-z]+)*\.[0-9_a-z]+([-.][0-9_a-z]+)*', '', string)
    string = re.sub(r'\s+', ' ', string)

    return string

def seg_words(sentence):
    """
    Cut sentence into words
    """
    words_list = []
    for seg in pseg.cut(clean_str(sentence)):
        if seg.word not in stop and seg.flag not in ['w', 'ns', 'nr', 't', 'r', 'u', 'e', 'y', 'o']:
            if len(seg.word) >= 2 or seg.word in single:
                words_list.append(seg.word)

    return " ".join(words_list)

if __name__ == '__main__':
    start = time.time()
    stop, single = load_dict(path='./data/dict/')
    # Load data
    data = open('./data/rawdata', 'r', encoding='UTF-8').readlines()
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
