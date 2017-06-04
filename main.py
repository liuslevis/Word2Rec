#-*- encoding:utf-8 -*-
#/usr/bin/ipython3
import sys,os
from gensim.models.word2vec import *

MODEL = 'output/word2vec.model'
PREFS = 'input/user_prefs.txt'
SEP   = ' '
MAX_CHOICE = 12
MAX_RECOMM = 12

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
    for vocab in flatmap(model.vocab):
        print('\n根据 %s 推荐：' % vocab)
        for vocab_score in model.most_similar(positive=[vocab]):
            vocab, score = vocab_score
            print('\t%s %.2f' % (vocab, score))

def print_vocabs(model):
    vocabs = list(model.vocab.keys())[:MAX_CHOICE]
    print('choose:')
    print('\tindex\titem')
    for i in range(len(vocabs)):
        print('\t%d\t%s' % (i, vocabs[i]))

def vocab_index(vocab, model):
    vocabs = list(model.vocab.keys())
    return vocabs.index(vocab) if vocab in vocabs else -1

def recommend(model, pos, neg):
    print('recommend:')
    print('\tindex\tscore\titem')
    most_similar = model.most_similar(positive=list(pos), negative=list(neg))
    for i in range(len(most_similar)):
        if i > MAX_RECOMM:
            break
        vocab, score = most_similar[i]
        index = vocab_index(vocab, model)
        print('\t%d\t%.2f\t%s' % (index, score, vocab))
    print('')

def choose(model):
    vocabs = list(model.vocab.keys())
    pos = set()
    neg = set()
    while True:
        print_vocabs(model)
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

        recommend(model, pos, neg)


def main():
    prefs = read_prefs()
    model = train_model(prefs)
    save_model(model)
    model = load_model()
    # calc_item_cf(model)
    choose(model)
    
if __name__ == '__main__':
    main()