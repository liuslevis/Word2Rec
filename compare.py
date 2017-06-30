#-*- encoding:utf-8 -*-
#/usr/bin/ipython3

# TODO 1 gen_train () fake data: rand -> w2v

import sys,os,time,json,urllib,shelve,random,datetime,collections,math
import gensim
import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
from cli import *

t1 = datetime.datetime(2017, 1, 1).strftime('%s')
t2 = datetime.datetime(2017, 4, 1).strftime('%s')
t3 = datetime.datetime(2017, 4, 30).strftime('%s')

TRAIN_FEATS = [
    'user_sex_0',
    'user_sex_1',
    'user_sex_2',
    'user_sex_3',
    'user_age_0',
    'user_age_1',
    'user_age_2',
    'user_age_3',
    'user_age_4',
    'user_city_0',
    'user_city_1',
    'user_city_2',
    'user_city_3',
    'user_city_4',

    'past_tag_num',

    'item_sex_0_ctr', 
    'item_sex_1_ctr', 
    'item_sex_2_ctr', 
    'item_sex_3_ctr', 
    'item_age_0_ctr', #-,20
    'item_age_1_ctr', #20,30
    'item_age_2_ctr', #30,40
    'item_age_3_ctr', #40,50
    'item_age_4_ctr', #40,50
    'item_city_0_ctr',
    'item_city_1_ctr',
    'item_city_2_ctr',
    'item_city_3_ctr',
    'item_city_4_ctr',
    ]

TRAIN_HEADER = ','.join([
    'label',
    'user',
    'item',
    'user_sex_0',
    'user_sex_1',
    'user_sex_2',
    'user_sex_3',
    'user_age_0',
    'user_age_1',
    'user_age_2',
    'user_age_3',
    'user_age_4',
    'user_city_0',
    'user_city_1',
    'user_city_2',
    'user_city_3',
    'user_city_4',
    'past_tag_num',
    'past_2day_tag_num',
    'item_sex_0_ctr',
    'item_sex_1_ctr',
    'item_sex_2_ctr',
    'item_sex_3_ctr',
    'item_age_0_ctr',
    'item_age_1_ctr',
    'item_age_2_ctr',
    'item_age_3_ctr',
    'item_age_4_ctr',
    'item_city_0_ctr',
    'item_city_1_ctr',
    'item_city_2_ctr',
    'item_city_3_ctr',
    'item_city_4_ctr',
    ])

shelfadd_path  = 'raw/shelfadd_201701_201704_mod100.csv'
user_prop_path = 'raw/user_prop_201701_201704_mod100.csv'
item_tag_path  = 'raw/wrbid_tag.csv'

train_path     = 'input/train_201701_201704_mod100.csv'
valid_path     = 'input/valid_201701_201704_mod100.csv'
test_path      = 'input/test/test_user_%s.csv'

USER_KEY = 'vid'
ITEM_KEY = 'bookid'
ACT_KEY = 'timestamp'

NEG_POS_SAMPLE_RATIO = 5
LR_THRESHOLD = .2 # .59 # LR 输出低于阈值值时，用候选集补齐

def read_csv_to_dict(path, key='vid', sep=',', valmap=None):
    ret = {}
    cols = None
    with open(path) as f:
        for line in f.readlines():
            if not cols:
                cols = line.rstrip().split(sep)
            else:
                parts = line.rstrip().split(sep)
                k = None
                v = {}
                for i in range(len(cols)):
                    val = parts[i]
                    if i == cols.index(key):
                        k = val
                    else:
                        val = val if not valmap else valmap(val)
                        v.update({cols[i]:val})
                ret.update({k:v})
    return ret

def common_tags(items1, items2, item_prop):
    tags1 = set()
    tags2 = set()
    for i in items1:
        tags1 |= set(item_prop[i]['tag'])

    for i in items2:
        tags2 |= set(item_prop[i]['tag'])
    return tags1 & tags2

def gen_lines(user, label_items, feats, item_prop, user_prop, fake_neg):
    li = []

    for label_item in label_items:
        li.append(gen_line(1, user, label_item, feats, item_prop, user_prop))
    if fake_neg:
        user_items = set(label_items).union(set(feats['user'][user]['past_items']))
        k = len(label_items) * NEG_POS_SAMPLE_RATIO
        for fake_item in random.sample(set(item_prop.keys())-user_items, k):
            li.append(gen_line(0, user, fake_item, feats, item_prop, user_prop))
    return li


def gen_line(label, user, item, feats, item_prop, user_prop):
    def get(item, d, default=-1):
        return d[item] if item in d else default
    
    sex_no  = get_sex_no(user_prop, user)
    age_no  = get_age_no(user_prop, user)
    city_no = get_city_no(user_prop, user)

    li = []
    li += [str(label)]
    li += [str(user)]
    li += [str(item)]
    
    li += [str(1 if sex_no == 0 else 0)]# 'user_sex_0',
    li += [str(1 if sex_no == 1 else 0)]# 'user_sex_1',
    li += [str(1 if sex_no == 2 else 0)]# 'user_sex_2',
    li += [str(1 if sex_no == 3 else 0)]# 'user_sex_3',
    
    li += [str(1 if age_no == 0 else 0)]# 'user_age_0',
    li += [str(1 if age_no == 1 else 0)]# 'user_age_1',
    li += [str(1 if age_no == 2 else 0)]# 'user_age_2',
    li += [str(1 if age_no == 3 else 0)]# 'user_age_3',
    li += [str(1 if age_no == 4 else 0)]# 'user_age_4',

    li += [str(1 if city_no == 0 else 0)]# 'user_city_0',
    li += [str(1 if city_no == 1 else 0)]# 'user_city_1',
    li += [str(1 if city_no == 2 else 0)]# 'user_city_2',
    li += [str(1 if city_no == 3 else 0)]# 'user_city_3',
    li += [str(1 if city_no == 4 else 0)]# 'user_city_4',

    li += [str(len(common_tags([item], feats['user'][user]['past_items'], item_prop)))]
    li += [str(len(common_tags([item], feats['user'][user]['past_2day_items'], item_prop)))]

    li += [str(feats['item'][item]['item_sex_0_ctr'])]
    li += [str(feats['item'][item]['item_sex_1_ctr'])]
    li += [str(feats['item'][item]['item_sex_2_ctr'])]
    li += [str(feats['item'][item]['item_sex_3_ctr'])]
    li += [str(feats['item'][item]['item_age_0_ctr'])]
    li += [str(feats['item'][item]['item_age_1_ctr'])]
    li += [str(feats['item'][item]['item_age_2_ctr'])]
    li += [str(feats['item'][item]['item_age_3_ctr'])]
    li += [str(feats['item'][item]['item_age_4_ctr'])]
    li += [str(feats['item'][item]['item_city_0_ctr'])]
    li += [str(feats['item'][item]['item_city_1_ctr'])]
    li += [str(feats['item'][item]['item_city_2_ctr'])]
    li += [str(feats['item'][item]['item_city_3_ctr'])]
    li += [str(feats['item'][item]['item_city_4_ctr'])]

    assert len(li) == len(TRAIN_HEADER.split(','))
    return ','.join(li)

# [t1,t2) feat
# [t2,t3] label
def gen_train(t1, t2, t3, actions, user_prop, item_prop, feats, train_path, valid_path):
    all_users = set()
    for ts in actions:
        user = actions[ts][USER_KEY]
        all_users.add(user)
    train_users = set(random.sample(list(all_users), int(len(all_users) * 0.7)))
    valid_users = all_users - train_users
    
    gen_data(t1, t2, t3, actions, train_users, user_prop, item_prop, feats, train_path, fake_neg=True)
    gen_data(t1, t2, t3, actions, valid_users, user_prop, item_prop, feats, valid_path, fake_neg=False)

def gen_feats(t1, t2, t3, actions, user_prop, item_prop):
    feats = {} 
    feats.setdefault('user',{})
    feats.setdefault('item',{})

    items = set(item_prop.keys())
    users = set(user_prop.keys())
    for ts in actions:
        user = actions[ts][USER_KEY]
        item = actions[ts][ITEM_KEY]
        items.add(item)
        users.add(user)
    
    items = list(items)
    users = list(users)

    item_for_index = {items[i]:i for i in range(len(items))}
    index_for_item = {i:items[i] for i in range(len(items))}
    user_for_index = {users[i]:i for i in range(len(users))}
    index_for_user = {i:users[i] for i in range(len(users))}

    for user in users:
        feats['user'].setdefault(user, {
            'past_items':[], 
            'label_items':[], 
            'past_2day_items':[],
        })
    for item in items:
        feats['item'].setdefault(item, {
            'item_sex_0_ctr':0, 
            'item_sex_1_ctr':0, 
            'item_sex_2_ctr':0, 
            'item_sex_3_ctr':0, 
            'item_age_0_ctr':0, 
            'item_age_1_ctr':0, 
            'item_age_2_ctr':0, 
            'item_age_3_ctr':0, 
            'item_age_4_ctr':0, 
            'item_city_0_ctr':0,
            'item_city_1_ctr':0,
            'item_city_2_ctr':0,
            'item_city_3_ctr':0,
            'item_city_4_ctr':0,
        })

    c_item_sex   = np.ones(shape=(4,len(items)))
    t_item_sex   = np.zeros(shape=(4,1))
    ctr_item_sex = np.zeros(shape=(4,len(items)))

    c_item_age   = np.ones(shape=(5,len(items)))
    t_item_age   = np.zeros(shape=(5,1))
    ctr_item_age = np.zeros(shape=(5,len(items)))

    c_item_city   = np.ones(shape=(5,len(items)))
    t_item_city   = np.zeros(shape=(5,1))
    ctr_item_city = np.zeros(shape=(5,len(items)))

    # cnt = 0
    for ts in actions:
        # cnt += 1
        # if cnt > 100: break
        user = actions[ts][USER_KEY]
        item = actions[ts][ITEM_KEY]

        if user not in user_prop or item not in item_prop: continue

        sex_no = get_sex_no(user_prop, user)
        age_no = get_age_no(user_prop, user)
        city_no = get_city_no(user_prop, user)
        item_index = item_for_index[item]
        user_index = user_for_index[user]

        if t1 <= ts <= t3:
            if t1 <= ts < t2:
                feats['user'][user]['past_items'].append(item)

                c_item_sex[sex_no][user_index] += 1
                c_item_age[age_no][user_index] += 1
                c_item_city[city_no][user_index] += 1

                t_item_sex[sex_no] += 1
                t_item_age[age_no] += 1
                t_item_city[city_no] += 1

            if t2 <= ts <= t3: 
                feats['user'][user]['label_items'].append(item)

            if int(t2) - int(ts) > 2 * 24 * 3600:
                feats['user'][user]['past_2day_items'].append(item)
        
    ctr_item_sex  = c_item_sex * 1.0 / t_item_sex
    ctr_item_age  = c_item_age * 1.0 / t_item_age
    ctr_item_city = c_item_city * 1.0 / t_item_city

    def ctr_to_feats(ctr, feats, string):
        for i in range(ctr.shape[0]):
            for j in range(ctr.shape[1]):
                if 0 < ctr[i][j] < 1:
                    # print('rich', index_for_item[j])
                    feats['item'][index_for_item[j]][string % i] = ctr[i][j]

    ctr_to_feats(ctr_item_sex, feats, 'item_sex_%d_ctr')
    ctr_to_feats(ctr_item_age, feats, 'item_age_%d_ctr')
    ctr_to_feats(ctr_item_city, feats, 'item_city_%d_ctr')
    return feats

# feats = gen_feats(t1, t2, t3, actions, user_prop, item_prop)

def get_sex_no(user_prop, user):
    ret = int(user_prop[user]['sex'])
    if ret > 3: return 3
    if ret < 0: return 0
    return ret

def get_age_no(user_prop, user):
    ret = int(user_prop[user]['age'])
    if ret < 20: return 0
    if ret < 30: return 1
    if ret < 40: return 2
    if ret < 50: return 3
    return 4

def get_city_no(user_prop, user):
    ret = int(user_prop[user]['city'])
    if ret < 0: return 0
    if ret > 4: return 4
    return ret

# ret: {user:{'past_items':[items...], 'label_items':[items,...]}}
def gen_data(t1, t2, t3, actions, users, user_prop, item_prop, feats, path, fake_neg):
    def skip(user, users, item):
        if user not in users:
            return True
        elif user not in user_prop:
            return True
        elif item and item not in item_prop:
            return True
        else:
            return False

    li = [TRAIN_HEADER]
    for user in users:
        if skip(user, users, None): continue
        if user not in feats['user'] or not feats['user'][user]['label_items']: continue

        items = feats['user'][user]['label_items']

        li += gen_lines(user, items, feats, item_prop, user_prop, fake_neg)
    with open(path, 'w') as f:
        f.write('\n'.join(li))
    print('gen_data() write:', path)
    return feats

def get_past_items(t1, t2, user, actions, item_prop):
    ret = set()
    for ts in actions:
        if t1 <= ts <= t2:
            user_ = actions[ts][USER_KEY]
            item = actions[ts][ITEM_KEY]
            if user_ == user and item in item_prop:
                ret.add(item)
    return ret

def get_past_2day_items(t1, t2, user, actions, item_prop):
    ret = set()
    for ts in actions:
        if t1 <= ts <= t2 and int(t2) - int(ts) > 2 * 24 * 3600:
            user_ = actions[ts][USER_KEY]
            item = actions[ts][ITEM_KEY]
            if user_ == user and item in item_prop:
                ret.add(item)
    return ret

# ret cand_item_scores
def gen_test(t1, t2, test_path, user, item_prop, user_prop, feats, w2v, topN, recent):
    cand_item_scores = cf_tag_recommend(t1, t2, w2v, user, actions, item_prop, topN, recent, items_only=False)

    test_path = test_path % str(user)
    if os.path.exists(test_path):
        return cand_item_scores

    li = [TRAIN_HEADER]
    past_items = get_past_items(t1, t2, user, actions, item_prop)
    cand_items = list(map(lambda x:x[0], cand_item_scores))
    if user in user_prop:
        for item in item_prop:
            if item not in past_items and item in cand_items:
                li.append(gen_line(0, user, item, feats, item_prop, user_prop))
    with open(test_path, 'w') as f:
        f.write('\n'.join(li))
    print('gen_test write:', test_path, 'user:', user, 'items:', len(li) - 1)
    return cand_item_scores

def lr_recommend(lr, t1, t2, cand_item_scores, test_path, user, actions, user_prop, item_prop, topN, recent):
    df = pd.read_csv(test_path % str(user))
    if len(df) == 0:
        return list()
    X,y = df[TRAIN_FEATS], df.label.values
    items = list(map(lambda x:str(x), df.item.values))
    scores = lr.predict_proba(X)[:,1]
    lr_item_scores = list(filter(lambda x:x[1] > LR_THRESHOLD, sorted(zip(items, scores), key=lambda x:x[1], reverse=True)))

    lr_items = set(map(lambda x:x[0], lr_item_scores))
    cand_items = set(map(lambda x:x[0], cand_item_scores))

    item_scores = sorted_nondup_items_scores(lr_item_scores)
    cand_item_scores = sorted_nondup_items_scores(cand_item_scores)

    for cand_item, score in cand_item_scores:
        if cand_item not in lr_items:
            item_scores.append((cand_item, score))
    return lr_item_scores, item_scores[:topN]

def print_book_titles(bookids, indents=1, print_console=False):
    li = []
    for bookid in bookids:
        s = ''.join(['\t' for i in range(indents)])
        s += bookid + get_book_title(bookid)
        li += [s]
    if print_console: print('\n' + li)
    return li

def print_items(items, item_idx=0, get_item_desc=get_book_title, ignore_ts=False, print_console=True):
    li = []
    for item in items:
        s = '\t'
        for i in range(len(item)):
            if i == item_idx:
                s += str(item[i]) + ' ' + get_item_desc(item[i]) + ' '
            elif type(item[i]) in [float, np.float64]:
                s +=  '%.3f' % item[i] + ' '
            elif type(item[i]) in [int]:
                s += '%d' % item[i]
            elif item[i].isdigit() and int(item[i]) > 10000000:
                if not ignore_ts:
                    s += time.strftime('%Y-%m-%d %H:%M', time.localtime(int(item[i]))) + ' '
            else:
                s += str(item[i]) + ' '
        li += [s]
    if print_console: print('\n'.join(li))
    return li

def lr_train(train_path, valid_path):
    # train
    lr = linear_model.LogisticRegression(C=1e5)
    df = pd.read_csv(train_path)
    X,y = df[TRAIN_FEATS], df.label.values
    lr.fit(X, y)
    # test
    df = pd.read_csv(valid_path)
    X,y = df[TRAIN_FEATS], df.label.values
    # pred = lr.predict_proba(X)[:,0] > 0.9
    # print(sum(pred == (y == 1)), len(pred == (y == 1)))
    return lr

def get_user_action(t1, t2, target_user, actions, items_only, item_prop=None):
    ret = set()
    items = set()
    for ts in actions:
        if t1 <= ts <= t2:
            user = actions[ts][USER_KEY]
            item = actions[ts][ITEM_KEY]
            if item_prop and item not in item_prop:
                continue
            if user == target_user and item not in items:
                ret.add((item, ts))
                items.add(item)
    item_ts = sorted(list(ret), key=lambda x:x[1])
    if items_only:
        return list(map(lambda x:x[0], item_ts))
    else:
        return item_ts

# @param T ground true list of user-item pairs
# @param R recommend list of user-item pairs
# @return recall, precision
# @usage print('recall/precision:',score(T=[(1,1),(2,2),(3,3)], R=[(1,2),(1,3),(1,4),(2,2),(3,3)]))
def recall_precision(T, R):
    recall    = len(set(T) & set(R)) / len(set(T))
    precision = len(set(T) & set(R)) / len(set(R))
    return recall, precision

# usage: hot_items = get_hot_items(actions)
def get_hot_items(actions, topN=500):
    li = []
    for ts in actions:
        item = actions[ts][ITEM_KEY]
        li.append(item)
    return set([elem[0] for elem in collections.Counter(li).most_common(topN)])

def sorted_nondup_items_scores(item_scores):
    d = {}
    for item, score in item_scores:
        if item not in d:
            d[item] = score
        else:
            s = d[item]
            if score > s:
                d[item] = score
    return sorted(list(d.items()), key=lambda x:x[1], reverse=True)

def cf_recommend(t1, t2, w2v, user, actions, topN, recent, items_only):
    item_scores = []
    items = get_user_action(t1, t2, user, actions, items_only=True)
    if not items:
        return []
    for item in items[-recent:]:
        topn = int(math.ceil(topN/recent))
        item_scores += calc_recommend_item_scores(w2v, pos=[item], neg=[], topn=topn)

    item_scores = sorted_nondup_items_scores(item_scores)[:topN]
    if items_only:
        return list(map(lambda x:x[0], item_scores))
    else:
        return item_scores


def cf_tag_recommend(t1, t2, w2v, user, actions, item_prop, topN, recent, items_only):
    w2v_item_scores = cf_recommend(t1, t2, w2v, user, actions, topN * 10, recent, items_only=False)
    past_items = get_past_items(t1, t2, user, actions, item_prop)

    item_scores = []
    for item, _ in w2v_item_scores:
        score = len(common_tags([item], past_items, item_prop)) if item in item_prop else 0
        item_scores.append((item, score))
    item_scores = sorted_nondup_items_scores(item_scores)[:topN]
    if items_only:
        return list(map(lambda x:x[0], item_scores))
    else:
        return item_scores

def cf_tag_mul_recommend(t1, t2, w2v, user, actions, item_prop, topN, recent, items_only):
    w2v_item_scores = cf_recommend(t1, t2, w2v, user, actions, topN * 10, recent, items_only=False)
    past_items = get_past_items(t1, t2, user, actions, item_prop)

    item_scores = []
    for item,score in w2v_item_scores:
        score = (score * len(common_tags([item], past_items, item_prop))) if item in item_prop else 0
        item_scores.append((item, score))
    item_scores = sorted_nondup_items_scores(item_scores)[:topN]
    if items_only:
        return list(map(lambda x:x[0], item_scores))
    else:
        return item_scores

user_prop = read_csv_to_dict(user_prop_path, key=USER_KEY, sep=',')
item_prop = read_csv_to_dict(item_tag_path,  key=ITEM_KEY, sep=',', valmap=lambda x:x.split('|'))
actions   = read_csv_to_dict(shelfadd_path,  key=ACT_KEY,  sep=',')
hot_items = get_hot_items(actions)
w2v       = train_model(read_prefs(PREFS, t1, t2))
# test_users = list(map(lambda x:str(x), np.unique(pd.read_csv(valid_path).user.values)))
rec_types = ['LR', 'word2vec','word2vec+tag','word2vecxtag']

# test_users = ['10707']
# test_users = ['10707','24329407','31304907','11252907','10959707','10959707']
test_users = ['24329407','31304907','11252907','10959707','13654807','13856907','26414907','30309507','3211707','12505207','11659107','7920207','12224807','3427207','8250707','4806807','21616807','15554207','31714007','24627507','39314307','2258207']
rec_types = ['LR']

feats = gen_feats(t1, t2, t3, actions, user_prop, item_prop)


for rec_type in rec_types:
    if rec_type == 'LR':
        if not os.path.exists(train_path) or not os.path.exists(valid_path):
            gen_train(t1, t2, t3, actions, user_prop, item_prop, feats, train_path, valid_path)
        lr = lr_train(train_path, valid_path)
    for n_recent in [15]:
        for topN in [200]:
            for n_test_user in [50]:
                n_test_user = min(n_test_user, len(test_users))

                out_path = 'output/report_user%d_topn%d_recent%d_%s.txt' % (n_test_user, topN, n_recent, rec_type)
                if rec_type == 'LR':
                    out_path = 'output/report_user%d_topn%d_recent%d_%s_feat%d.txt' % (n_test_user, topN, n_recent, rec_type, len(TRAIN_FEATS))
                print('calc %s' % out_path)

                hit = 0
                n_recall = 0
                n_precision = 0
                li = []
                for user in test_users[:n_test_user]:
                    user = str(user)
                    read_item_ts = get_user_action(t1, t2, user, actions, items_only=False, item_prop=item_prop)
                    add_item_ts = get_user_action(t2, t3, user, actions, items_only=False, item_prop=item_prop)

                    li += ['用户 %s 看过 %d 本：' % (user, len(read_item_ts))]
                    li += print_items(read_item_ts, print_console=False)

                    li += ['用户 %s 加书架 %d 本：' % (user, len(add_item_ts))]
                    Tu = set(map(lambda x:x[0], add_item_ts))
                    li += print_items(add_item_ts, print_console=False)

                    item_scores = None
                    if rec_type == 'LR':
                        cand_item_scores = gen_test(t1, t2, test_path, user, item_prop, user_prop, feats, w2v, topN, n_recent)
                        lr_item_scores, item_scores = lr_recommend(lr, t1, t2, cand_item_scores, test_path, user, actions, user_prop, item_prop, topN, n_recent)
                        li += ['根据属性 %s 推荐 %d 本，候选集补齐 %d 本：' % (str(TRAIN_FEATS), len(lr_item_scores), len(item_scores))]

                    elif rec_type == 'word2vec':
                        item_scores = cf_recommend(t1, t2, w2v, user, actions, topN, n_recent, items_only=False)
                        li += ['根据协同过滤（word2vec）推荐 %d 本：' % (len(item_scores))]
                    elif rec_type == 'word2vec+tag':
                        item_scores = cf_tag_recommend(t1, t2, w2v, user, actions, item_prop, topN, n_recent, items_only=False)
                        li += ['根据协同过滤（word2vec+tag）推荐 %d 本：' % (len(item_scores))]
                    elif rec_type == 'word2vecxtag':
                        item_scores = cf_tag_mul_recommend(t1, t2, w2v, user, actions, item_prop, topN, n_recent, items_only=False)
                        li += ['根据协同过滤（word2vec x tag）推荐 %d 本：' % (len(item_scores))]
                    
                    Ru = set(map(lambda x:str(x[0]), item_scores))
                    li += print_items(item_scores, print_console=False)

                    # print(recall_precision(Tu, Ru))

                    li += ['推荐命中: %d' % len(Tu & Ru)]
                    li += print_book_titles(Tu & Ru)

                    hit += len(Tu & Ru)
                    n_recall += len(Ru)
                    n_precision += topN

                li = ['推荐算法 %s' % rec_type, 
                    '算法阈值 %.4f' % LR_THRESHOLD,
                    '测试用户数量 %d' % n_test_user, 
                    'topN %d' % topN, 
                    '兴趣窗口 %d' % n_recent, 
                    '召回率 %.6f' % ((hit / n_recall) if n_recall > 0 else 0), 
                    '准确率 %.6f' % ((hit / n_precision) if n_precision > 0 else 0)] + li

                with open(out_path, 'w') as f:
                    f.write('\n'.join(li))
