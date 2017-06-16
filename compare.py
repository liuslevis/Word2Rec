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

NEG_POS_SAMPLE_RATIO = 1

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
        k = len(future_items) * NEG_POS_SAMPLE_RATIO
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

def gen_test(t1, t2, test_path, user, item_prop, user_prop, hot_items, w2v, topN, recent):
    li = [TRAIN_HEADER]
    past_items = get_past_items(t1, t2, user, actions, item_prop)
    cand_items = cf_recommend(t1, t2, w2v, user, actions, topN * 10, recent, items_only=True)
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
        return list()
    X,y = df[TRAIN_FEATS], df.label.values
    items = df.item.values
    scores = lr.predict_proba(X)[:,0]
    item_scores = sorted(zip(items, scores), key=lambda x:x[1], reverse=True)[:topN]
    return item_scores

def print_items(items, item_idx=0, get_item_desc=get_book_title, ignore_ts=False, print_console=True):
    li = []
    for item in items:
        s = '\t'
        for i in range(len(item)):
            if i == item_idx:
                s += str(item[i]) + ' ' + get_item_desc(item[i]) + ' '
            elif type(item[i]) in [float, np.float64]:
                s +=  '%.3f' % item[i] + ' '
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

def get_user_action(t1, t2, target_user, actions, items_only):
    ret = set()
    items = set()
    for ts in actions:
        if t1 <= ts <= t2:
            user = actions[ts][USER_KEY]
            item = actions[ts][ITEM_KEY]
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

def cf_recommend(t1, t2, w2v, user, actions, topN=200, recent=10, items_only=True):
    item_scores = []
    items = get_user_action(t1, t2, user, actions, items_only=True)
    if not items:
        return []
    for item in items[-recent:]:
        topn = int(math.ceil(topN/recent))
        item_scores += calc_recommend_item_scores(w2v, pos=[item], neg=[], topn=topn)
    item_scores = list(set(item_scores))
    item_scores = sorted(item_scores, key=lambda x:x[1], reverse=True)[:topN]
    if items_only:
        return list(map(lambda x:x[0], item_scores))
    else:
        return item_scores


user_prop = read_csv_to_dict(user_prop_path, key=USER_KEY, sep=',')
item_prop = read_csv_to_dict(item_tag_path,  key=ITEM_KEY, sep=',', valmap=lambda x:x.split('|'))
actions   = read_csv_to_dict(shelfadd_path,  key=ACT_KEY,  sep=',')
hot_items = get_hot_items(actions)
w2v       = train_model(read_prefs(PREFS, t1, t2))


for rec_type in ['LR']:#['LR', 'word2vec']:
    if rec_type == 'LR':
        # gen_train(t1, t2, t3, actions, user_prop, item_prop, train_path, valid_path)
        lr = lr_train(train_path, valid_path)
    for n_recent in [10]:
        for topN in [200]:
            for n_test_user in [50]:
                out_path = 'output/report_user%d_topn%d_recent%d_%s.txt' % (n_test_user, topN, n_recent, rec_type)
                print('calc %s' % out_path)
                test_users = list(map(lambda x:str(x), np.unique(pd.read_csv(valid_path).user.values)[:n_test_user]))

                hit = 0
                n_recall = 0
                n_precision = 0
                li = []
                for user in test_users:
                    user = str(user)
                    li += ['用户 %s 看过：' % user]
                    li += print_items(get_user_action(t1, t2, user, actions, items_only=False), print_console=False)

                    li += ['用户 %s 加书架：' % user]
                    item_ts = get_user_action(t2, t3, user, actions, items_only=False)
                    Tu = set(map(lambda x:x[0], item_ts))
                    li += print_items(item_ts, print_console=False)

                    item_scores = None
                    if rec_type == 'LR':
                        li += ['根据属性推荐：%s' % str(TRAIN_FEATS)]
                        gen_test(t1, t2, test_path, user, item_prop, user_prop, hot_items, w2v, topN, n_recent)
                        item_scores = lr_recommend(lr, test_path, actions, user_prop, item_prop, topN)
                    elif rec_type == 'word2vec':
                        li += ['根据协同过滤（word2vec）推荐：']
                        item_scores = cf_recommend(t1, t2, w2v, user, actions, topN, n_recent, items_only=False)
                    Ru = set(map(lambda x:str(x[0]), item_scores))
                    li += print_items(item_scores, print_console=False)

                    # print(recall_precision(Tu, Ru))

                    li += ['该用户推荐命中: %d\n' % len(Tu & Ru)]
                    hit += len(Tu & Ru)
                    n_recall += len(Ru)
                    n_precision += topN

                li = ['推荐算法 %s' % rec_type, 
                    '测试用户数量 %d' % n_test_user, 
                    'topN %d' % topN, 
                    '兴趣窗口 %d' % n_recent, 
                    '召回率 %.6f' % (hit / n_recall) if n_recall > 0 else 0, 
                    '准确率 %.6f' % (hit / n_precision) if n_precision > 0 else 0] + li

                with open(out_path, 'w') as f:
                    f.write('\n'.join(li))
                # print('\n'.join(li))