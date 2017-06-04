#-*- encoding:utf-8 -*-
#/usr/bin/ipython3
import sys,os
from gensim.models.word2vec import *

MODEL = 'output/word2vec.model'
PREFS = 'input/user_prefs.txt'
SEP = ' '

# {user:{dt1:item1, dt2:item2, ...}}
def read_prefs(path=PREFS):
    prefs = {}
    with open(path) as f:
        for line in f.readlines():
            parts = line.rstrip().split(',')
            if len(parts) == 3:
                user = parts[0]
                time = parts[1]
                item = parts[2]
                pref = prefs[user] if user in prefs else {}
                pref.update({time:item})
                prefs.update({user:pref})
    return prefs

def sents_from_prefs(prefs):
    sents = []
    for user in prefs:
        sent = ''
        for time in sorted(prefs[user].keys()):
            word = prefs[user][time]
            sent += SEP + word.replace(SEP, '')
        sents.append(sent)
    return sents

def flatmap(vocab):
    ret = []
    for i in vocab:
        if type(i) == type('a'):
            ret.append(i)
        elif type(i) == type([]):
            for j in i:
                ret.append(j)
    return ret

def train_model(prefs):
    sents = sents_from_prefs(prefs)
    vocab = [s.split(SEP) for s in sents]
    model = Word2Vec(vocab, size=100, window=5, min_count=1, workers=4)
    return model

def save_model(model, path=MODEL):
    model.save_word2vec_format(path, binary=False)

def load_model(path=MODEL):
    return Word2Vec.load_word2vec_format(path, binary=False)

def calc_item_cf(model):
    print('基于书籍的 word2vec 协同过滤推荐')
    for item in flatmap(model.vocab):
        print('\n根据 %s 推荐：' % item)
        for item_score in model.most_similar(positive=[item]):
            item, score = item_score
            print('\t%s %.2f' % (item, score))

def print_vocabs(vocabs):
    print('choose:')
    for i in range(len(vocabs)):
        print('\t', i, vocabs[i])

def recomend(model, pos, neg):
    print('recommend:')
    for item_score in model.most_similar(positive=list(pos), negative=list(neg)):
        item, score = item_score
        print('\t%s %.2f' % (item, score))
    print('')

def choose(model):
    vocabs = list(model.vocab.keys())[:]
    pos = set()
    neg = set()
    while True:
        print_vocabs(vocabs)
        line = input('like:')
        if line.isdigit():
            vocab = vocabs[int(line)]
            pos.add(vocab)
            print('\t + %s = %s' % (vocab, pos))

        line = input('hate:')
        if line.isdigit():
            vocab = vocabs[int(line)]
            neg.add(vocab)
            print('\t + %s = %s' % (vocab, neg))

        recomend(model, pos, neg)


def main():
    prefs = read_prefs()
    model = train_model(prefs)
    save_model(model)
    model = load_model()
    # calc_item_cf(model)
    choose(model)
    
if __name__ == '__main__':
    main()