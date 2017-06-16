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

TRAIN_FEATS = ['user_sex','user_age','user_city','read_tag_num']

TRAIN_HEADER = 'label,user,item,user_sex,user_age,user_city,read_tag_num'


df = pd.read_csv(train_path)

X,y = df[TRAIN_FEATS].values, df.label.values

kf = KFold(n_splits=5)
for train, test in kf.split(X):
    print()
    print('kfold train:%d test:%d' % (len(train), len(test)))
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(X_train, y_train)
    y_true = y_test
    y_pred = lr.predict(X_test)
    print("summary:")
    print(classification_report(y_true, y_pred))


