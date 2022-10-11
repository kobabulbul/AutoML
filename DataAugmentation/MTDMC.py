from joblib import Parallel, delayed
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import sklearn.model_selection
import pandas as pd
import matplotlib.pyplot as plt
from data_augmentation import VirtualSampleGeneration
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import h5py
from scipy.stats.qmc import LatinHypercube
import scipy.stats.qmc as qmc

def train_cv(train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    vsg = VirtualSampleGeneration(X_train, y_train)

    model = RandomForestRegressor(n_estimators=400, max_depth=9)
    vsg.add_virtual_data(model, 1000)

    snew_X_train = vsg.train_input
    snew_y_train = vsg.train_output

    model.fit(snew_X_train, snew_y_train)
    y_pred = model.predict(snew_X_train)
    y_pred_test = model.predict(X_test)

    trainRMSE = np.sqrt(mean_squared_error(snew_y_train, y_pred))
    trainR2 = model.score(snew_X_train, snew_y_train)

    testRMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
    testR2 = model.score(X_test, y_test)
    return testRMSE, testR2, trainRMSE, trainR2

def train_multirun(K):
    skf = sklearn.model_selection.KFold(n_splits=K)
    testRMSEcv, testR2cv, trainRMSEcv, trainR2cv = zip(*Parallel(n_jobs=-1)(delayed(train_cv)(train_index, test_index) for train_index, test_index in skf.split(X, y)))
    return testRMSEcv, testR2cv, trainRMSEcv, trainR2cv

def train_va_cv(train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    model = RandomForestRegressor(n_estimators=400, max_depth=9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    trainRMSE = np.sqrt(mean_squared_error(y_train, y_pred))
    trainR2 = model.score(X_train, y_train)

    testRMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
    testR2 = model.score(X_test, y_test)
    return testRMSE, testR2, trainRMSE, trainR2

def train_va_multirun(K):
    skf = sklearn.model_selection.KFold(n_splits=K)
    testRMSEcv, testR2cv, trainRMSEcv, trainR2cv = zip(*Parallel(n_jobs=-1)(delayed(train_va_cv)(train_index, test_index) for train_index, test_index in skf.split(X, y)))
    return testRMSEcv, testR2cv, trainRMSEcv, trainR2cv
data = pd.read_csv("slump_test.csv")

X = data.iloc[:, 1:10].values
y = data.iloc[:, 10].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1)

K = 10
multiruns = 50


hf = h5py.File('vsg_augmentation.h5', 'w')
g1 = hf.create_group('Vanilla')
g2 = hf.create_group('VSG')

testRMSEcv, testR2cv, trainRMSEcv, trainR2cv = zip(*Parallel(n_jobs=-1)(delayed(train_va_multirun)(K) for i in range(multiruns)))

g1.create_dataset('testRMSE', data=testRMSEcv)
g1.create_dataset('testR2', data=testR2cv)
g1.create_dataset('trainRMSE', data=trainRMSEcv)
g1.create_dataset('trainR2', data=trainR2cv)

testRMSEcv1, testR2cv1, trainRMSEcv1, trainR2cv1 = zip(*Parallel(n_jobs=-1)(delayed(train_multirun)(K) for i in range(multiruns)))

g2.create_dataset('testRMSE', data=testRMSEcv1)
g2.create_dataset('testR2', data=testR2cv1)
g2.create_dataset('trainRMSE', data=trainRMSEcv1)
g2.create_dataset('trainR2', data=trainR2cv1)

hf.close()