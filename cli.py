#-*- encoding:utf-8 -*-
#/usr/bin/ipython3
import sys,os,time,json,urllib,shelve,random
import gensim

CACHE   = shelve.open('./cache/shelve.cache') if sys.platform == 'darwin' else {}
MODEL   = 'output/word2vec.model'
PREFS   = ['raw/shelfadd_201701_201704_mod100.csv', 'raw/shelfadd_201703_201704_mod100.csv', 'raw/shelfadd_201704_mod100.csv', 'raw/user_prefs.txt'][0]
HOTLIST = 'raw/wrbid_hotlist.txt'
SEP     = ' '

W2V_SIZE = 500
W2V_WINDOW = 5 # avg user read 10 book in 2 month
W2V_MIN_COUNT = 1 

MAX_CHOICE_CLI = 12
MAX_RECOMM_CLI = 12
MAX_RECOMM_WEB = 30
MAX_CHOICE_WEB_HOT_BOOK = 3
MAX_CHOICE_WEB_COLD_BOOK = 2
MAX_RECENT_BOOKS = 20

def get_col_vals(csvpath, colname, sep, outpath=None):
    ret = []
    with open(csvpath) as f:
        cols = None
        for line in f.readlines():
            if cols is None:
                cols = line.rstrip().split(sep)
                continue
            val = line.rstrip().split(sep)[cols.index(colname)]
            ret.append(val)
    if outpath:
        with open(outpath, 'w') as f:
            f.write('\n'.join(ret))
    return set(ret)

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
    return ret[:MAX_RECENT_BOOKS]

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

def cache_bookinfos():
    vid = get_col_vals('raw/shelfadd_201701_201704_mod100.csv', 'vid', ',', 'raw/vids_201701_201704_mod100.txt')
    b1 = get_col_vals('raw/wrbid_hotlist.txt', 'bookid', ',')
    b2 = get_col_vals('raw/wrbid_tag.txt', 'bookid', ',')
    b3 = get_col_vals('raw/shelfadd_201701_201704_mod100.csv', 'bookid', ',')
    for bookids in [b3.intersection(b2), b3 - b2, b1]:
        cnt = 0
        for bookid in bookids:
            cnt += 1
            print(cnt, bookid)
            get_book_info(bookid)

def read_hotlist_bookids():
    ret = []
    with open(HOTLIST) as f:
        for line in f.readlines():
            parts = line.rstrip().split()
            if len(parts) == 2:
                bookid = parts[0]
                rank = parts[1]
                ret.append(bookid)
    return ret

# {user:{dt1:item1, dt2:item2, ...}}
def read_prefs(path, t1=None, t2=None):
    prefs = {}
    with open(path) as f:
        for line in f.readlines():
            parts = line.rstrip().split(',')
            if len(parts) == 3:
                time = parts[0]
                user = parts[1]
                item = parts[2]
                if time.isalpha(): continue
                if t1 and t2 and not (t1 <= time <= t2): continue
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
    model = gensim.models.word2vec.Word2Vec(vocab, size=W2V_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=4)
    return model

def save_model(model, path=MODEL):
    model.wv.save_word2vec_format(path, binary=False)

def load_model(path=MODEL):
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)

def calc_item_cf(model):
    print('基于书籍的 word2vec 协同过滤推荐')
    for vocab in flatmap(model.wv.vocab):
        print('\n根据 %s 推荐：' % vocab)
        for vocab_score in model.most_similar(positive=[vocab]):
            vocab, score = vocab_score
            print('\t%s %.2f' % (vocab, score))

def print_vocabs(model, shuffle=False):
    vocabs = list(model.wv.vocab.keys())
    print('choose:')
    print('\tindex\titem')
    li = random.sample(list(range(len(vocabs))), MAX_CHOICE_CLI) if shuffle else list(range(MAX_CHOICE_CLI))
    for i in li:
        print('\t%d\t%s' % (i, get_book_title(vocabs[i])))

def vocab_index(vocab, model):
    vocabs = list(model.wv.vocab.keys())
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

def calc_recommend_items(model, pos, neg=[]):
    ret = []
    vocabs = list(model.wv.vocab.keys())
    pos = set(filter(lambda x:x in vocabs, pos))
    neg = set(filter(lambda x:x in vocabs, neg))
    if len(pos) + len(neg) == 0:
        return []
    most_similar = model.most_similar(positive=list(pos), negative=list(neg))
    for i in range(len(most_similar)):
        if i > MAX_RECOMM_WEB:
            break
        vocab, score = most_similar[i]
        index = vocab_index(vocab, model)
        ret.append(vocab)
    return ret

def calc_recommend_books(model, pos, neg=[]):
    ret = []
    vocabs = list(model.wv.vocab.keys())
    pos = set(filter(lambda x:x in vocabs, pos))
    neg = set(filter(lambda x:x in vocabs, neg))
    if len(pos) + len(neg) == 0:
        return []
    most_similar = model.most_similar(positive=list(pos), negative=list(neg))
    for i in range(len(most_similar)):
        if i > MAX_RECOMM_WEB:
            break
        vocab, score = most_similar[i]
        index = vocab_index(vocab, model)
        bookinfo = get_book_info(vocab)
        if bookinfo:
            bookinfo['score'] = score
            ret.append(bookinfo)
    return ret

def get_random_books(model):
    ret = []
    hotlist = read_hotlist_bookids()
    vocabs = list(model.wv.vocab.keys())
    for bookid in random.sample(hotlist, MAX_CHOICE_WEB_HOT_BOOK):
        ret.append(get_book_info(bookid))
    for bookid in random.sample(vocabs, MAX_CHOICE_WEB_COLD_BOOK):
        ret.append(get_book_info(bookid))
    return ret
    
def cli_choose(model):
    vocabs = list(model.wv.vocab.keys())
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
    # prefs = read_prefs(PREFS)
    # model = train_model(prefs)
    # save_model(model)
    model = load_model()
    # calc_item_cf(model)
    cli_choose(model)
    
if __name__ == '__main__':
    main()