#-*- encoding:utf-8 -*-
#/usr/bin/ipython3
import sys,os,time,json,urllib,shelve,random,datetime
import gensim

t1 = datetime.datetime(2017, 1, 1).strftime('%s')
t2 = datetime.datetime(2017, 4, 1).strftime('%s')
t3 = datetime.datetime(2017, 4, 30).strftime('%s')

shelfadd_path  = 'input/shelfadd_201701_201704_mod100.csv'
user_prop_path = 'input/user_prop_201701_201704_mod100.csv'
item_tag_path  = 'input/wrbid_tag.csv'
train_path     = 'input/train_201701_201704_mod100.csv'

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


def gen_lines(user, future_items, past_items, item_prop, user_prop):
    li = []
    for future_item in future_items:
        li.append(gen_line(1, user, future_item, past_items, item_prop, user_prop))

    user_items = set(future_items).union(set(past_items))
    k = len(future_items) * POS_NEG_SAMPLE_RATIO
    for fake_item in random.sample(item_prop.keys(), k):
        if fake_item not in user_items:
            li.append(gen_line(0, user, fake_item, past_items, item_prop, user_prop))

    return li

TRAIN_HEADER = 'label,user,item,user_sex,user_age,user_city,read_tag_num'

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
def gen_train(t1, t2, t3, actions, user_prop, item_prop, train_path):
    ret = {} #{user:{'past':[items...], 'future':[items,...]}}
    for ts in actions:
        user = actions[ts][USER_KEY]
        item = actions[ts][ITEM_KEY]
        # skip out of dict user / item
        if user not in user_prop or item not in item_prop:
            continue
        if t1 <= ts <= t3:
            d = {'past':[], 'future':[]}
            ret.setdefault(user,d)
            if t1 <= ts < t2:
                ret[user]['past'].append(item)
            elif t2 <= ts <= t3:
                ret[user]['future'].append(item)

    li = [TRAIN_HEADER]
    for user in ret:
        future_items = ret[user]['future']
        past_items   = ret[user]['past']
        li += gen_lines(user, future_items, past_items, item_prop, user_prop)
    with open(train_path, 'w') as f:
        f.write('\n'.join(li))

    return ret



user_prop = read_csv_to_dict(user_prop_path, key=USER_KEY, sep=',')
item_prop = read_csv_to_dict(item_tag_path,  key=ITEM_KEY, sep=',', valmap=lambda x:x.split('|'))
actions   = read_csv_to_dict(shelfadd_path,  key=ACT_KEY,  sep=',')

ret = gen_train(t1, t2, t3, actions, user_prop, item_prop, train_path)



