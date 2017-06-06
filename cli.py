#-*- encoding:utf-8 -*-
#/usr/bin/ipython3
import sys,os,time,json,urllib,shelve
from gensim.models.word2vec import *
import random

MODEL = 'output/word2vec.model'
PREFS = ['input/shelfadd_201704.csv', 'input/user_prefs.txt'][0]
SEP   = ' '
CACHE = {}#shelve.open('./shelve.cache')

MAX_CHOICE_CLI = 12
MAX_RECOMM_CLI = 12
MAX_RECOMM_WEB = 30
MAX_CHOICE_WEB = 30

def jsonObj(url, use_cache=True):
    if use_cache and url in CACHE.keys():
        jObj = CACHE[url] 
        return jObj
    time.sleep(0.01 * random.random())
    response = urllib.request.urlopen(url)
    data = response.read()      # a `bytes` object
    text = data.decode('utf-8') # a `str`; this step can't be used if data is binary
    jObj = json.loads(text)
    CACHE[url] = jObj
    return jObj

# [(bookId, readUpdateTime), ...]
def get_recent_books(vid):
    ret = []
    if not vid: return ret
    vid = str(vid)
    url = 'http://wr.qq.com:8080/shelf/sync?synckey=0&vid=%s' % vid
    jObj = jsonObj(url)
    if 'books' in jObj:
        for bookinfo in jObj['books']:
            if 'bookId' in bookinfo \
                and 'readUpdateTime' in bookinfo \
                and 'cover' in bookinfo \
                and 'http://res.weread.qq.com/publicfetch?' not in bookinfo['cover']:
                ret.append(bookinfo)
    ret.sort(key=lambda x:x['readUpdateTime'], reverse=True)
    return ret

def get_book_info(bookId):
    bookId = str(bookId)
    if not bookId.isdigit(): return None
    url = 'http://wr.qq.com:8080/book/info?bookId=%s' % bookId
    return jsonObj(url)

def get_book_title(bookId):
    bookId = str(bookId)
    if not bookId.isdigit(): return bookId
    url = 'http://wr.qq.com:8080/book/info?bookId=%s' % bookId
    jObj = jsonObj(url)
    return (jObj['title'] if 'title' in jObj else '') + ' '+ (jObj['author'] if 'author' in jObj else '') 

# {user:{dt1:item1, dt2:item2, ...}}
def read_prefs(path=PREFS):
    prefs = {}
    with open(path) as f:
        for line in f.readlines():
            parts = line.rstrip().split(',')
            if len(parts) == 3:
                time = parts[0]
                user = parts[1]
                item = parts[2]
                if time.isalpha(): continue
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

def print_vocabs(model, shuffle=False):
    vocabs = list(model.vocab.keys())
    print('choose:')
    print('\tindex\titem')
    li = random.sample(list(range(len(vocabs))), MAX_CHOICE_CLI) if shuffle else list(range(MAX_CHOICE_CLI))
    for i in li:
        print('\t%d\t%s' % (i, get_book_title(vocabs[i])))

def vocab_index(vocab, model):
    vocabs = list(model.vocab.keys())
    return vocabs.index(vocab) if vocab in vocabs else -1

def print_recommend(model, pos, neg):
    print('recommend:')
    print('\tindex\tscore\titem')
    most_similar = model.most_similar(positive=list(pos), negative=list(neg))
    for i in range(len(most_similar)):
        if i > MAX_RECOMM_CLI:
            break
        vocab, score = most_similar[i]
        index = vocab_index(vocab, model)
        print('\t%d\t%.2f\t%s' % (index, score, get_book_title(vocab)))
    print('')

def calc_recommend_books(model, pos, neg):
    ret = []
    vocabs = list(model.vocab.keys())
    pos = set(filter(lambda x:x in vocabs, pos))
    neg = set(filter(lambda x:x in vocabs, neg))
    most_similar = model.most_similar(positive=list(pos), negative=list(neg))
    for i in range(len(most_similar)):
        if i > MAX_RECOMM_WEB:
            break
        vocab, score = most_similar[i]
        index = vocab_index(vocab, model)
        bookinfo = get_book_info(vocab)
        if bookinfo:
            bookinfo['score'] = ret
            ret.append(bookinfo)
    return ret

def get_random_books(model):
    ret = []
    vocabs = list(model.vocab.keys())
    for bookid in random.sample(vocabs, MAX_CHOICE_WEB):
        ret.append(get_book_info(bookid))
    return ret
    
def cli_choose(model):
    vocabs = list(model.vocab.keys())
    pos = set()
    neg = set()
    shuffle = True
    print_vocabs(model, shuffle)
    while True:
        line = input('shuffle:(y/[n])')
        shuffle = line == 'y'
        if shuffle:
            print_vocabs(model, shuffle)
            continue

        line = input('like:')
        parts = line.split()
        for part in parts:
            if part.isdigit():
                vocab = vocabs[int(part)]
                pos.add(vocab)
        print('\t%s' % (list(map(lambda x:get_book_title(x), pos))))

        line = input('hate:')
        parts = line.split()
        for part in parts:
            if part.isdigit():
                vocab = vocabs[int(part)]
                neg.add(vocab)
        print('\t%s' % (list(map(lambda x:get_book_title(x), neg))))
        
        if pos or neg:
            print_recommend(model, pos, neg)


def main():
    # print(get_recent_books(2000007))
    random.seed(99)
    # prefs = read_prefs()
    # model = train_model(prefs)
    # save_model(model)
    model = load_model()
    # calc_item_cf(model)
    cli_choose(model)
    
if __name__ == '__main__':
    main()