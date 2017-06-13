#-*- encoding:utf-8 -*-
#/usr/bin/ipython3
import sys,os,time,json,urllib,shelve,random,datetime,collections
import gensim
import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
# import matplotlib.pyplot as plt
from cli import *

t1 = datetime.datetime(2017, 1, 1).strftime('%s')
t2 = datetime.datetime(2017, 4, 1).strftime('%s')
t3 = datetime.datetime(2017, 4, 30).strftime('%s')

TRAIN_FEATS = ['user_sex','user_age','user_city','read_tag_num']
# TRAIN_FEATS = ['read_tag_num']

TRAIN_HEADER = 'label,user,item,user_sex,user_age,user_city,read_tag_num'

shelfadd_path  = 'raw/shelfadd_201701_201704_mod100.csv'
user_prop_path = 'raw/user_prop_201701_201704_mod100.csv'
item_tag_path  = 'raw/wrbid_tag.csv'

train_path     = 'input/train_201701_201704_mod100.csv'
valid_path     = 'input/valid_201701_201704_mod100.csv'
test_path      = 'input/test_user.csv'

USER_KEY = 'vid'
ITEM_KEY = 'bookid'
ACT_KEY = 'timestamp'

POS_NEG_SAMPLE_RATIO = 20

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
    tags1 = []
    tags2 = []
    for i in items1:
        tags1 += item_prop[i]['tag']
    for i in items2:
        tags2 += item_prop[i]['tag']
    return set(tags1).intersection(set(tags2))


def gen_lines(user, future_items, past_items, item_prop, user_prop, fake_neg):
    li = []
    for future_item in future_items:
        li.append(gen_line(1, user, future_item, past_items, item_prop, user_prop))

    if fake_neg:
        user_items = set(future_items).union(set(past_items))
        k = len(future_items) * POS_NEG_SAMPLE_RATIO
        for fake_item in random.sample(item_prop.keys(), k):
            if fake_item not in user_items:
                li.append(gen_line(0, user, fake_item, past_items, item_prop, user_prop))
    return li


def gen_line(label, user, label_item, past_items, item_prop, user_prop):
    li = []
    li += [str(label)]
    li += [str(user)]
    li += [str(label_item)]
    li += [str(user_prop[user][prop] if prop in user_prop[user] else -1) for prop in ['sex', 'age', 'city']]
    li += [str(len(common_tags([label_item], past_items, item_prop)))]
    assert len(li) == len(TRAIN_HEADER.split(','))
    return ','.join(li)

# [t1,t2) feat
# [t2,t3] label
def gen_train(t1, t2, t3, actions, user_prop, item_prop, train_path, valid_path):
    all_users = set()
    for ts in actions:
        user = actions[ts][USER_KEY]
        all_users.add(user)
    train_users = set(random.sample(list(all_users), int(len(all_users) * 0.7)))
    valid_users = all_users - train_users
    gen_feat(t1, t2, t3, actions, train_users, user_prop, item_prop, train_path, fake_neg=True)
    gen_feat(t1, t2, t3, actions, valid_users, user_prop, item_prop, valid_path, fake_neg=False)

# ret = gen_train(t1, t2, t3, actions, user_prop, item_prop, train_path)
def gen_feat(t1, t2, t3, actions, users, user_prop, item_prop, path, fake_neg):
    def skip(user, users, item):
        if user not in users:
            return True
        elif user not in user_prop:
            return True
        elif item and item not in item_prop:
            return True
        else:
            return False
    ret = {} #{user:{'past':[items...], 'future':[items,...]}}
    for ts in actions:
        user = actions[ts][USER_KEY]
        item = actions[ts][ITEM_KEY]

        if skip(user, users, item): continue

        if t1 <= ts <= t3:
            d = {'past':[], 'future':[]}
            ret.setdefault(user,d)
            if t1 <= ts < t2:
                ret[user]['past'].append(item)
            elif t2 <= ts <= t3:
                ret[user]['future'].append(item)

    li = [TRAIN_HEADER]
    for user in ret:
        if skip(user, users, None): continue

        future_items = ret[user]['future']
        past_items   = ret[user]['past']
        li += gen_lines(user, future_items, past_items, item_prop, user_prop, fake_neg)
    with open(path, 'w') as f:
        f.write('\n'.join(li))
    print('gen_feat() write:', path)
    return ret

def get_past_items(t1, t2, user, actions, item_prop):
    ret = set()
    for ts in actions:
        if t1 <= ts <= t2:
            user_ = actions[ts][USER_KEY]
            item = actions[ts][ITEM_KEY]
            if user_ == user and item in item_prop:
                ret.add(item)
    return ret

def get_cf_items(w2v, items, topN=5):
    ret = []
    vocabs = list(w2v.wv.vocab.keys())
    for item in items:
        if item in vocabs:
            ret += calc_recommend_items(w2v, pos=[item])[:topN]
    return ret

def gen_test(t1, t2, t3, test_path, user, item_prop, user_prop, hot_items, w2v):
    li = [TRAIN_HEADER]
    past_items = get_past_items(t1, t2, user, actions, item_prop)
    cand_items = get_cf_items(w2v, past_items) # hot_items
    if user in user_prop:
        for label_item in item_prop:
            if label_item not in past_items and label_item in cand_items:
                li.append(gen_line(0, user, label_item, past_items, item_prop, user_prop))
    with open(test_path, 'w') as f:
        f.write('\n'.join(li))
    # print('gen_test write:', test_path)

def lr_recommend(lr, test_path, actions, user_prop, item_prop, topN):
    df = pd.read_csv(test_path)
    if len(df) == 0:
        return set()
    X,y = df[TRAIN_FEATS], df.label.values
    items = df.item.values
    scores = lr.predict_proba(X)[:,0]
    item_scores = sorted(zip(items, scores), key=lambda x:x[1], reverse=True)[:topN]
    Ru = set(map(lambda x:str(x[0]), item_scores))
    return Ru

def print_items(items):
    for item in items:
        print('\t', item, get_book_title(item))

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

def get_user_action(t1, t2, target_user, actions):
    Tu = set()
    for ts in actions:
        if t1 <= ts <= t2:
            user = actions[ts][USER_KEY]
            item = actions[ts][ITEM_KEY]
            if user == target_user:
                Tu.add(item)
    return Tu

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

user_prop = read_csv_to_dict(user_prop_path, key=USER_KEY, sep=',')
item_prop = read_csv_to_dict(item_tag_path,  key=ITEM_KEY, sep=',', valmap=lambda x:x.split('|'))
actions   = read_csv_to_dict(shelfadd_path,  key=ACT_KEY,  sep=',')
hot_items = get_hot_items(actions)
w2v       = train_model(read_prefs(PREFS, t1, t2))

# gen_train(t1, t2, t3, actions, user_prop, item_prop, train_path, valid_path)
lr = lr_train(train_path, valid_path)

topN = 50
hit = 0
n_recall = 0
n_precision = 0
test_users = list(map(lambda x:str(x), pd.read_csv(valid_path).user.values[:10]))
for test_user in test_users:#['30135407', '14438807', '2000007']:
    test_user = str(test_user)
    print('用户', test_user, '看过：')
    print_items(get_user_action(t1, t2, test_user, actions))

    print('用户', test_user, '加书架：')
    Tu = get_user_action(t2, t3, test_user, actions)
    print_items(Tu)

    print('根据属性', TRAIN_FEATS, '推荐：')
    gen_test(t1, t2, t3, test_path, test_user, item_prop, user_prop, hot_items, w2v)
    Ru = lr_recommend(lr, test_path, actions, user_prop, item_prop, topN)
    print_items(Ru)
    # print(recall_precision(Tu, Ru))

    print('推荐命中:', len(Tu & Ru))
    print()
    hit += len(Tu & Ru)
    n_recall += len(Ru)
    n_precision += topN

print('recall/precision:', hit / n_recall, hit / n_precision)
