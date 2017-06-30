#-*- encoding:utf-8 -*-
#/usr/bin/ipython3

import sys,os,time,json,urllib,shelve,random,datetime,collections,math
import gensim
import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix,classification_report

train_path     = 'input/train_201701_201704_mod100.csv'
valid_path     = 'input/valid_201701_201704_mod100.csv'

TRAIN_FEATS = [
    'user_city',
    'past_tag_num',
    'item_sex_0_ctr', 
    'item_sex_1_ctr', 
    'item_age_0_ctr', #-,20
    'item_age_1_ctr', #20,30
    'item_age_2_ctr', #30,40
    'item_age_3_ctr', #40,50
    'item_city_0_ctr',
    'item_city_1_ctr',
    ]

# df = pd.read_csv(train_path)
# X_train, y_train = df[TRAIN_FEATS].values, df.label.values

# df = pd.read_csv(valid_path)
# X_valid, y_valid = df[TRAIN_FEATS].values, df.label.values

# lr = linear_model.LogisticRegression(C=1e4)
# lr.fit(X_train, y_train)

# print("train summary:")
# print(classification_report(y_true=y_train, y_pred=lr.predict(X_train)))

# print("test summary:")
# print(classification_report(y_true=y_valid, y_pred=lr.predict(X_valid)))

df = pd.read_csv(train_path)
X,y = df[TRAIN_FEATS].values, df.label.values
kf = KFold(n_splits=2)
lr = None
threshold = 0.2
for train, test in kf.split(X):
    print()
    print('kfold train:%d test:%d' % (len(train), len(test)))
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(X_train, y_train)

    y_true = y_train
    y_pred = np.where(lr.predict_proba(X_train)[:,1] > threshold, 1, 0)
    print("train summary:")
    print(classification_report(y_true, y_pred))

    y_true = y_test
    y_pred = np.where(lr.predict_proba(X_test)[:,1] > threshold, 1, 0)
    print("valid summary:")
    print(classification_report(y_true, y_pred))

