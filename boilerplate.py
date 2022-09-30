# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:00:30 2018

@author: Programming
"""

import myLib, pathlib, copy, datetime, math, os, sys, shutil, zipfile, re, time, datetime, pytz
import urllib as url
import csv, json, sys
import schedule
import threading
from Robinhood import Robinhood

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
import talib
import tensorflow

import inspect
import projectv2
import myLibv2 as lib

###################################FUNCTIONS###################################

def create_model(inputLength = 0, outputLength = 0, neurons = 0, layers = 3, 
                 dropout_rate = 0.0, batch_size = 25, epochs = 100,
                 activation='relu', optimizer = 'rmsprop'):
    
    regressor = Sequential()
    regressor.add(Dense(units = neurons,activation = activation, kernel_initializer = "uniform",input_dim = inputLength))
    regressor.add(Dropout(dropout_rate))
    for i in range(0,layers):
        regressor.add(Dense(units = neurons,activation = activation, kernel_initializer = "uniform"))
        regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(units = outputLength,activation = 'linear', kernel_initializer = "uniform"))
    regressor.compile(optimizer = optimizer,loss = 'mean_squared_error')#,metrics = ['mean_absolute_error', 'mean_absolute_percentage_error']
    return regressor

def create_model_lstm(inputLength = 0, outputLength = 0, neurons = 0, layers = 3, 
                 dropout_rate = 0.0, batch_size = 25, epochs = 100,
                 activation='relu', optimizer = 'rmsprop'):
    
    regressor = Sequential()
    regressor.add(LSTM(units = neurons,activation = activation, kernel_initializer = "uniform",input_dim = inputLength))
    regressor.add(Dropout(dropout_rate))
    for i in range(0,layers):
        regressor.add(LSTM(units = neurons,activation = activation, kernel_initializer = "uniform"))
        regressor.add(Dropout(dropout_rate))
    regressor.add(LSTM(units = outputLength,activation = 'linear', kernel_initializer = "uniform"))
    regressor.compile(optimizer = optimizer,loss = 'mean_squared_error',metrics = ['mean_absolute_error', 'mean_absolute_percentage_error'])
    return regressor

def get_ta_indicators(open_values, high_values, low_values, close_values, volume, low_version = False):
    
    indicators = []
    ############################ STATISTIC FUNCTIONS ##########################
    indicators.append(talib.STDDEV(close_values, timeperiod=5, nbdev=1))
    indicators.append(talib.WMA(close_values, timeperiod=30))
    indicators.append(talib.TSF(close_values, timeperiod=14))
    indicators.append(talib.VAR(close_values, timeperiod=5, nbdev=1))
    indicators.append(talib.LINEARREG_SLOPE(close_values, timeperiod=14))
    indicators.append(talib.LINEARREG_INTERCEPT(close_values, timeperiod=14))
    indicators.append(talib.LINEARREG_ANGLE(close_values, timeperiod=14))
    indicators.append(talib.LINEARREG(close_values, timeperiod=14))
    indicators.append(talib.CORREL(high_values, low_values, timeperiod=30))
    indicators.append(talib.BETA(high_values, low_values, timeperiod=5))
    ####TODO: low value versions being added... remove later if doesnt help####
    if(low_version):
        indicators.append(talib.STDDEV(low_values, timeperiod=5, nbdev=1))
        indicators.append(talib.WMA(low_values, timeperiod=30))
        indicators.append(talib.TSF(low_values, timeperiod=14))
        indicators.append(talib.VAR(low_values, timeperiod=5, nbdev=1))
        indicators.append(talib.LINEARREG_SLOPE(low_values, timeperiod=14))
        indicators.append(talib.LINEARREG_INTERCEPT(low_values, timeperiod=14))
        indicators.append(talib.LINEARREG_ANGLE(low_values, timeperiod=14))
        indicators.append(talib.LINEARREG(low_values, timeperiod=14))
    
    ########################## MATH TRANSFORM FUNCTIONS #######################
    #test = talib.ACOS(close_values.transpose())
    #indicators.append(talib.ACOS(close_values))
    #indicators.append(talib.ASIN(close_values))
    #indicators.append(talib.ATAN(close_values))
    #indicators.append(talib.CEIL(close_values))
    #indicators.append(talib.COS(close_values))
    #indicators.append(talib.COSH(close_values))
    #indicators.append(talib.EXP(close_values))
    #indicators.append(talib.FLOOR(close_values))
    #indicators.append(talib.LN(close_values))
    #indicators.append(talib.LOG10(close_values))
    #indicators.append(talib.SIN(close_values))
    #indicators.append(talib.SINH(close_values))
    #indicators.append(talib.SQRT(close_values))
    #indicators.append(talib.TAN(close_values))
    #####TODO: low value versions being added... remove later if doesnt help###
    #indicators.append(talib.ACOS(low_values))
    #indicators.append(talib.ASIN(low_values))
    #indicators.append(talib.ATAN(low_values))
    #indicators.append(talib.CEIL(low_values))
    #indicators.append(talib.COS(low_values))
    #indicators.append(talib.COSH(low_values))
    #indicators.append(talib.EXP(low_values))
    #indicators.append(talib.FLOOR(low_values))
    #indicators.append(talib.LN(low_values))
    #indicators.append(talib.LOG10(low_values))
    #indicators.append(talib.SIN(low_values))
    #indicators.append(talib.SINH(low_values))
    #indicators.append(talib.SQRT(low_values))
    #indicators.append(talib.TAN(low_values))
    
    ########################## PRICE TRANSFORM FUNCTIONS ######################
    
    indicators.append(talib.AVGPRICE(open_values, high_values, low_values, close_values))
    indicators.append(talib.MEDPRICE(high_values, low_values))
    indicators.append(talib.TYPPRICE(high_values, low_values, close_values))
    indicators.append(talib.WCLPRICE(high_values, low_values, close_values))
    
    ########################## OVERLAP STUDIES FUNCTIONS ######################
    
    upperband, middleband, lowerband = talib.BBANDS(close_values, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    indicators.append(upperband)
    indicators.append(middleband)
    indicators.append(lowerband)
    indicators.append(talib.DEMA(close_values, timeperiod=30))
    indicators.append(talib.EMA(close_values, timeperiod=30))
    indicators.append(talib.HT_TRENDLINE(close_values))
    indicators.append(talib.KAMA(close_values, timeperiod=30))
    indicators.append(talib.MA(close_values, timeperiod=30, matype=0))
    mama, fama = talib.MAMA(close_values, fastlimit=0.5, slowlimit=0.05)
    indicators.append(mama)
    indicators.append(fama)
    #indicators.append(talib.MAVP(close_values, periods, minperiod=2, maxperiod=30, matype=0))
    indicators.append(talib.MIDPOINT(close_values, timeperiod=14))
    indicators.append(talib.MIDPRICE(high_values, low_values, timeperiod=14))
    indicators.append(talib.SAR(high_values, low_values, acceleration=0.00, maximum=0.2))
    indicators.append(talib.SAR(high_values, low_values, acceleration=0.01, maximum=0.2))
    indicators.append(talib.SAR(high_values, low_values, acceleration=0.02, maximum=0.2))
    indicators.append(talib.SAR(high_values, low_values, acceleration=0.03, maximum=0.2))
    indicators.append(talib.SAREXT(high_values, low_values, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0))
    indicators.append(talib.SMA(close_values, timeperiod=10))
    indicators.append(talib.SMA(close_values, timeperiod=20))
    indicators.append(talib.SMA(close_values, timeperiod=50))  # [5, 15, 50, 200]
    indicators.append(talib.SMA(close_values, timeperiod=200))
    indicators.append(talib.T3(close_values, timeperiod=5, vfactor=0))
    indicators.append(talib.TEMA(close_values, timeperiod=30))
    indicators.append(talib.TRIMA(close_values, timeperiod=30))
    indicators.append(talib.WMA(close_values, timeperiod=30))
    ####TODO: low value versions being added... remove later if doesnt help####
    if(low_version):
        upperband, middleband, lowerband = talib.BBANDS(low_values, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        indicators.append(upperband)
        indicators.append(middleband)
        indicators.append(lowerband)
        indicators.append(talib.DEMA(low_values, timeperiod=30))
        indicators.append(talib.EMA(low_values, timeperiod=30))
        indicators.append(talib.HT_TRENDLINE(low_values))
        indicators.append(talib.KAMA(low_values, timeperiod=30))
        indicators.append(talib.MA(low_values, timeperiod=30, matype=0))
        mama, fama = talib.MAMA(low_values, fastlimit=0.5, slowlimit=0.05)
        indicators.append(mama)
        indicators.append(fama)
        indicators.append(talib.MIDPOINT(low_values, timeperiod=14))
        indicators.append(talib.SMA(low_values, timeperiod=5))
        indicators.append(talib.SMA(low_values, timeperiod=15))
        indicators.append(talib.SMA(low_values, timeperiod=50))
        indicators.append(talib.SMA(low_values, timeperiod=200))
        indicators.append(talib.T3(low_values, timeperiod=5, vfactor=0))
        indicators.append(talib.TEMA(low_values, timeperiod=30))
        indicators.append(talib.TRIMA(low_values, timeperiod=30))
        indicators.append(talib.WMA(low_values, timeperiod=30))
    ###########################################################################
    
    ######################## MOMENTUM INDICATOR FUNCTIONS #####################
    
    indicators.append(talib.ADX(high_values, low_values, close_values, timeperiod=14))
    indicators.append(talib.ADXR(high_values, low_values, close_values, timeperiod=14))
    indicators.append(talib.APO(close_values, fastperiod=12, slowperiod=26, matype=0))
    aroondown, aroonup = talib.AROON(high_values, low_values, timeperiod=14)
    indicators.append(aroondown)
    indicators.append(aroonup)
    indicators.append(talib.AROONOSC(high_values, low_values, timeperiod=14))
    indicators.append(talib.BOP(open_values, high_values, low_values, close_values,))
    indicators.append(talib.CCI(high_values, low_values, close_values, timeperiod=14))
    indicators.append(talib.CMO(close_values, timeperiod=14))
    indicators.append(talib.DX(high_values, low_values, close_values, timeperiod=14))
    macd, macdsignal, macdhist = talib.MACD(close_values, fastperiod=12, slowperiod=26, signalperiod=9)
    indicators.append(macd)
    indicators.append(macdsignal)
    indicators.append(macdhist)
    macd1, macdsignal1, macdhist1 = talib.MACDEXT(close_values, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    indicators.append(macd1)
    indicators.append(macdsignal1)
    indicators.append(macdhist1)
    macd2, macdsignal2, macdhist2 = talib.MACDFIX(close_values, signalperiod=9)
    indicators.append(macd2)
    indicators.append(macdsignal2)
    indicators.append(macdhist2)
    indicators.append(talib.MFI(high_values, low_values, close_values, volume, timeperiod=14))
    indicators.append(talib.MINUS_DI(high_values, low_values, close_values, timeperiod=14))
    indicators.append(talib.MINUS_DM(high_values, low_values, timeperiod=14))
    indicators.append(talib.MOM(close_values, timeperiod=10))
    indicators.append(talib.PLUS_DI(high_values, low_values, close_values, timeperiod=14))
    indicators.append(talib.PLUS_DM(high_values, low_values, timeperiod=14))
    indicators.append(talib.PPO(close_values, fastperiod=12, slowperiod=26, matype=0))
    indicators.append(talib.ROC(close_values, timeperiod=10))
    indicators.append(talib.ROCP(close_values, timeperiod=10))
    indicators.append(talib.ROCR(close_values, timeperiod=10))
    indicators.append(talib.ROCR100(close_values, timeperiod=10))
    indicators.append(talib.RSI(close_values, timeperiod=2))
    indicators.append(talib.RSI(close_values, timeperiod=10))
    slowk, slowd = talib.STOCH(high_values, low_values, close_values, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    fastk, fastd = talib.STOCHF(high_values, low_values, close_values, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastk1, fastd1 = talib.STOCHRSI(close_values, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    indicators.append(slowk)
    indicators.append(slowd)
    indicators.append(fastk)
    indicators.append(fastd)
    indicators.append(fastk1)
    indicators.append(fastd1)
    indicators.append(talib.TRIX(close_values, timeperiod=30))
    indicators.append(talib.ULTOSC(high_values, low_values, close_values, timeperiod1=7, timeperiod2=14, timeperiod3=28))
    indicators.append(talib.WILLR(high_values, low_values, close_values, timeperiod=14))
    
    ######################### VOLUME INDICATOR FUNCTIONS ######################
    
    indicators.append(talib.AD(high_values, low_values, close_values, volume))
    indicators.append(talib.ADOSC(high_values, low_values, close_values, volume, fastperiod=3, slowperiod=10))
    indicators.append(talib.OBV(close_values, volume))
    
    ########################## CYCLE INDICATOR FUNCTIONS ######################
    #worse with cycle indicators possibly
    indicators.append(talib.HT_DCPERIOD(close_values))
    indicators.append(talib.HT_DCPHASE(close_values))
    inphase, quadrature = talib.HT_PHASOR(close_values)
    indicators.append(inphase)
    indicators.append(quadrature)
    sine, leadsine = talib.HT_SINE(close_values)
    indicators.append(sine)
    indicators.append(leadsine)
    indicators.append(talib.HT_TRENDMODE(close_values))
    ####TODO: low value versions being added... remove later if doesnt help
    if(low_version):
        indicators.append(talib.HT_DCPERIOD(low_values))
        indicators.append(talib.HT_DCPHASE(low_values))
        inphase, quadrature = talib.HT_PHASOR(low_values)
        indicators.append(inphase)
        indicators.append(quadrature)
        sine, leadsine = talib.HT_SINE(low_values)
        indicators.append(sine)
        indicators.append(leadsine)
        indicators.append(talib.HT_TRENDMODE(low_values))
        
    
    ####################### VOLATILITY INDICATOR FUNCTIONS ####################
    
    indicators.append(talib.ATR(high_values, low_values, close_values, timeperiod=14))
    indicators.append(talib.NATR(high_values, low_values, close_values, timeperiod=14))
    indicators.append(talib.TRANGE(high_values, low_values, close_values))
    
    ############################ PATTERN RECOGNITION ##########################
    
    indicators.append(talib.CDL2CROWS(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDL3BLACKCROWS(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDL3INSIDE(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDL3LINESTRIKE(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDL3OUTSIDE(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDL3STARSINSOUTH(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDL3WHITESOLDIERS(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLABANDONEDBABY(open_values, high_values, low_values, close_values, penetration=0))
    indicators.append(talib.CDLADVANCEBLOCK(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLBELTHOLD(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLBREAKAWAY(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLCLOSINGMARUBOZU(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLCONCEALBABYSWALL(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLCOUNTERATTACK(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLDARKCLOUDCOVER(open_values, high_values, low_values, close_values, penetration=0))
    indicators.append(talib.CDLDOJI(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLDOJISTAR(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLDRAGONFLYDOJI(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLENGULFING(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLEVENINGDOJISTAR(open_values, high_values, low_values, close_values, penetration=0))
    indicators.append(talib.CDLEVENINGSTAR(open_values, high_values, low_values, close_values, penetration=0))
    indicators.append(talib.CDLGAPSIDESIDEWHITE(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLGRAVESTONEDOJI(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLHAMMER(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLHANGINGMAN(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLHARAMI(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLHARAMICROSS(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLHIGHWAVE(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLHIKKAKE(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLHIKKAKEMOD(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLHOMINGPIGEON(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLIDENTICAL3CROWS(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLINNECK(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLINVERTEDHAMMER(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLKICKING(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLKICKINGBYLENGTH(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLLADDERBOTTOM(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLLONGLEGGEDDOJI(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLLONGLINE(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLMARUBOZU(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLMATCHINGLOW(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLMATHOLD(open_values, high_values, low_values, close_values, penetration=0))
    indicators.append(talib.CDLMORNINGDOJISTAR(open_values, high_values, low_values, close_values, penetration=0))
    indicators.append(talib.CDLMORNINGSTAR(open_values, high_values, low_values, close_values, penetration=0))
    indicators.append(talib.CDLONNECK(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLPIERCING(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLRICKSHAWMAN(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLRISEFALL3METHODS(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLSEPARATINGLINES(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLSHOOTINGSTAR(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLSHORTLINE(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLSPINNINGTOP(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLSTALLEDPATTERN(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLSTICKSANDWICH(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLTAKURI(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLTASUKIGAP(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLTHRUSTING(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLTRISTAR(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLUNIQUE3RIVER(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLUPSIDEGAP2CROWS(open_values, high_values, low_values, close_values))
    indicators.append(talib.CDLXSIDEGAP3METHODS(open_values, high_values, low_values, close_values))
    
    return indicators


#def get_pattern_recognition(open_values, high_values, low_values, close_values, volume):
#    indicators = []
#    indicators.append(["CDL2CROWS",talib.CDL2CROWS(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDL3BLACKCROWS",talib.CDL3BLACKCROWS(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDL3INSIDE",talib.CDL3INSIDE(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDL3LINESTRIKE",talib.CDL3LINESTRIKE(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDL3OUTSIDE",talib.CDL3OUTSIDE(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDL3STARSINSOUTH",talib.CDL3STARSINSOUTH(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDL3WHITESOLDIERS",talib.CDL3WHITESOLDIERS(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLABANDONEDBABY",talib.CDLABANDONEDBABY(open_values, high_values, low_values, close_values, penetration=0)])
#    indicators.append(["CDLADVANCEBLOCK",talib.CDLADVANCEBLOCK(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLBELTHOLD",talib.CDLBELTHOLD(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLBREAKAWAY",talib.CDLBREAKAWAY(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLCLOSINGMARUBOZU",talib.CDLCLOSINGMARUBOZU(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLCONCEALBABYSWALL",talib.CDLCONCEALBABYSWALL(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLCOUNTERATTACK",talib.CDLCOUNTERATTACK(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLDARKCLOUDCOVER",talib.CDLDARKCLOUDCOVER(open_values, high_values, low_values, close_values, penetration=0)])
#    indicators.append(["CDLDOJI",talib.CDLDOJI(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLDOJISTAR",talib.CDLDOJISTAR(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLDRAGONFLYDOJI",talib.CDLDRAGONFLYDOJI(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLENGULFING",talib.CDLENGULFING(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLEVENINGDOJISTAR",talib.CDLEVENINGDOJISTAR(open_values, high_values, low_values, close_values, penetration=0)])
#    indicators.append(["CDLEVENINGSTAR",talib.CDLEVENINGSTAR(open_values, high_values, low_values, close_values, penetration=0)])
#    indicators.append(["CDLGAPSIDESIDEWHITE",talib.CDLGAPSIDESIDEWHITE(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLGRAVESTONEDOJI",talib.CDLGRAVESTONEDOJI(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLHAMMER",talib.CDLHAMMER(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLHANGINGMAN",talib.CDLHANGINGMAN(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLHARAMI",talib.CDLHARAMI(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLHARAMICROSS",talib.CDLHARAMICROSS(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLHIGHWAVE",talib.CDLHIGHWAVE(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLHIKKAKE",talib.CDLHIKKAKE(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLHIKKAKEMOD",talib.CDLHIKKAKEMOD(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLHOMINGPIGEON",talib.CDLHOMINGPIGEON(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLIDENTICAL3CROWS",talib.CDLIDENTICAL3CROWS(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLINNECK",talib.CDLINNECK(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLINVERTEDHAMMER",talib.CDLINVERTEDHAMMER(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLKICKING",talib.CDLKICKING(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLKICKINGBYLENGTH",talib.CDLKICKINGBYLENGTH(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLLADDERBOTTOM",talib.CDLLADDERBOTTOM(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLLONGLEGGEDDOJI",talib.CDLLONGLEGGEDDOJI(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLLONGLINE",talib.CDLLONGLINE(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLMARUBOZU",talib.CDLMARUBOZU(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLMATCHINGLOW",talib.CDLMATCHINGLOW(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLMATHOLD",talib.CDLMATHOLD(open_values, high_values, low_values, close_values, penetration=0)])
#    indicators.append(["CDLMORNINGDOJISTAR",talib.CDLMORNINGDOJISTAR(open_values, high_values, low_values, close_values, penetration=0)])
#    indicators.append(["CDLMORNINGSTAR",talib.CDLMORNINGSTAR(open_values, high_values, low_values, close_values, penetration=0)])
#    indicators.append(["CDLONNECK",talib.CDLONNECK(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLPIERCING",talib.CDLPIERCING(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLRICKSHAWMAN",talib.CDLRICKSHAWMAN(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLRISEFALL3METHODS",talib.CDLRISEFALL3METHODS(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLSEPARATINGLINES",talib.CDLSEPARATINGLINES(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLSHOOTINGSTAR",talib.CDLSHOOTINGSTAR(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLSHORTLINE",talib.CDLSHORTLINE(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLSPINNINGTOP",talib.CDLSPINNINGTOP(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLSTALLEDPATTERN",talib.CDLSTALLEDPATTERN(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLSTICKSANDWICH",talib.CDLSTICKSANDWICH(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLTAKURI",talib.CDLTAKURI(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLTASUKIGAP",talib.CDLTASUKIGAP(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLTHRUSTING",talib.CDLTHRUSTING(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLTRISTAR",talib.CDLTRISTAR(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLUNIQUE3RIVER",talib.CDLUNIQUE3RIVER(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLUPSIDEGAP2CROWS",talib.CDLUPSIDEGAP2CROWS(open_values, high_values, low_values, close_values)])
#    indicators.append(["CDLXSIDEGAP3METHODS",talib.CDLXSIDEGAP3METHODS(open_values, high_values, low_values, close_values)])
#    return indicators
#def vwap(close_values, volume):
#    for i in range(len(close_values))
#        talib.SMA(x * volume, y) / talib.SMA(volume, y)

def combine_indicators(indicators, index, col):
    ind_df = pd.DataFrame()
    
    for i in range(len(indicators)):
        temp_df = pd.DataFrame(indicators[i])
        ind_df = ind_df.join(temp_df, how='outer', rsuffix='_ind')
    ind_df = ind_df.set_index(index)
    ind_df.columns = col
    return ind_df

def time_series_queries(symbols):
    print("Grabbing [Time Series] data...")
    start = time.time()
    time_series_df = pd.DataFrame()
    for symbol in symbols:
        time.sleep(1)
        timeseries = lib.getTimeSeries("TIME_SERIES_DAILY",symbol,"daily","full","csv","R2M366UPHZGI1XC7")
        tempDataFrame = pd.DataFrame(timeseries[1:]).set_index(0)
        tempDataFrame.columns = timeseries[0:1][0][1:]
        time_series_df = time_series_df.join(tempDataFrame, how='outer', rsuffix='_'+str(symbol))
        print("TIME_SERIES_DAILY symbol:"+symbol)
        
    end = time.time()
    print("Complete.")
    print("Process took:"+str(end - start)+" seconds or "+str((end - start)/60)+" minutes")
    return time_series_df




def indicator_queries(symbol = "", function_type = [], time_period = [], series_type = []):
    print("Grabbing [Indicator] data and adding to dataset...")
    start = time.time()
    indicator_df = pd.DataFrame()
#    tslist[1:]
#    indicator_df.columns = tslist[0:1][0][1:]
    i = 0
    for function in function_type:
        for t in time_period:
            for series in series_type:
                time.sleep(1)
                indicator = [[]]
                while(indicator == [[]]):
                    indicator = lib.json_to_list(lib.getIndicator(function,symbol,"daily",t,series))
                    if(indicator == [[]]):
                        print("failed. waiting 1 second...")
                        time.sleep(1)
                    
                print("function_type:"+str(function)+"time_period:"+str(t)+"series_type:"+str(series))
                #print(indicator)
                tempDataFrame = pd.DataFrame(indicator[1:]).set_index(0)
                #print(tempDataFrame)
                #print(len(tempDataFrame.columns))
                temp_list = []
                for col in range(len(tempDataFrame.columns)):
                    temp_list.append(function+"_"+str(t)+"_"+series)
                tempDataFrame.columns = temp_list
                indicator_df = indicator_df.join(tempDataFrame, how='outer', rsuffix='_'+str(i))
                i+=1
    end = time.time()
    print("Complete.")
    print("Process took:"+str(end - start)+" seconds or "+str((end - start)/60)+" minutes")
    return indicator_df 

def normalize_test_1yr(test_data):
    ratio = 260/test_data[0]
    temp_list = []
    for i in range(len(test_data)-1):
        temp_list.append(test_data[i]*ratio)
    if(test_data[-1] > 1):
        temp_list.append(test_data[-1]*ratio)
    else:
        temp_list.append(test_data[-1])
    return temp_list                

def diff_report(pred, actual):
    print("Difference between Predicted:")
    print(pred)
    print("and Actual:")
    print(actual)
    print("is:")
    print(pred-actual)

def adj_diff_report(pred, actual, adj):
    print("Difference between Predicted:")
    print(pred)
    print("and Actual:")
    print(actual)
    print("is:")
    diff =pred - actual
    print(diff)
    print("Partially Adjusted Difference is:")
    open_adj = diff[0][0]
    part_adj = diff - open_adj
    print(part_adj)
    print("Fully Adjusted Difference is:")
    print(part_adj+adj)
    print("Partially Reversed Difference is:")
    open_adj = diff[0][0]
    part_adj = diff + open_adj
    print(part_adj)
    print("Fully Reversed Difference is:")
    print(part_adj-adj)

def adj_report(predictions, actual_open, opt_adj):
    open_adj = actual_open-predictions[:,0:1][0][0]
    print("Open Adjustment:")
    print(open_adj)
    print("Optimization Adjustment:")
    print(opt_adj)
    print("Base Prediction:")
    print(predictions)
    print("Base With Open Adjustment:")
    print(predictions[:,:]+open_adj)
    print("Optimized No Open Adjustment:")
    print(predictions[:,:]+opt_adj)
    print("Optimized with Open Adjustment:")
    print(predictions[:,:]+open_adj+opt_adj)    
    
    ######just for viewing, probably not useful#####
    print("REVERSALS:")
    print("Reverse Open Adjustment:")
    print(predictions[:,:]-open_adj+opt_adj)
    print("Reverse Optimize Adjustment:")
    print(predictions[:,:]+open_adj-opt_adj)
    print("Full Reverse Adjustment:")
    print(predictions[:,:]+open_adj-opt_adj)

def test_optimize_strategy(y_test,y_pred,open_adj = False):
    tens = optimize_strategy(y_test,y_pred,buy_adj = 0,adj_amt = .1,open_adj = open_adj)
    fives = optimize_strategy(y_test,y_pred,buy_adj = tens[-1],adj_amt = .05,open_adj = open_adj)
    return optimize_strategy(y_test,y_pred,buy_adj = fives[-1],adj_amt = .01,open_adj = open_adj)

def optimize_strategy(y_test,y_pred,buy_adj = 0,adj_amt = .1,open_adj = False):
    curr_val = test_strategy(y_test,y_pred,buy_adj,open_adj = open_adj)[-1]
    neg =  test_strategy(y_test,y_pred,buy_adj-adj_amt,open_adj = open_adj)[-1]
    pos = test_strategy(y_test,y_pred,buy_adj+adj_amt,open_adj = open_adj)[-1]
    
    if(neg > pos and neg > curr_val):
        adj_amt = -adj_amt
        curr_val = neg
        buy_adj += adj_amt
        
    if(neg < pos and pos > curr_val):
        curr_val = pos
        buy_adj += adj_amt
        
    while(True):
        buy_adj += adj_amt
        next_val = test_strategy(y_test,y_pred,buy_adj,open_adj = open_adj)[-1]
        if(next_val > curr_val):
            #print(test_strategy(y_test,y_pred,buy_adj,open_adj = open_adj))
            curr_val = next_val
        else:
            break
      
    result = test_strategy(y_test,y_pred,buy_adj-adj_amt,open_adj = open_adj)
    result.append(buy_adj-adj_amt)
    return result

def test_strategy(y_test,y_pred,buy_adj, open_adj = False):    
    #buy_adj = 0
    #open_adj = True
    open_pred = y_pred[:,0:1]
    open_actual = y_test[:,0:1]
    open_diff = 0
    if(open_adj):
        open_diff = (open_actual - open_pred)/2
        
    
    low_pred = y_pred[:,2:3]
    low_actual = y_test[:,2:3]
    high_actual = y_test[:,1:2]
    close_actual = y_test[:,3:4]
#    print(open_pred)
#    print(open_actual)
#    print(open_diff)
#    print(low_pred)
    adj_pred = low_pred+open_diff+buy_adj
    base_gain = close_actual-adj_pred
    above_low = adj_pred > low_actual
    below_high = adj_pred < high_actual
#    print(close_actual)
#    print(low_pred)
#    print(buy_adj)
    
    i=0
    num_gains = 0
    gross_gains = 0
    num_loss = 0
    gross_loss = 0
    standard_1yr = (100/len(y_test))
    for i in range(0, len(y_test)):
        #print("executed: "+str((above_low[i][0] and below_high[i][0]))+" gain/loss: "+str(base_gain[i][0]))
        if((above_low[i][0] and below_high[i][0]) and base_gain[i][0] > 0):
            num_gains+=1
            gross_gains += base_gain[i][0]
        if((above_low[i][0] and below_high[i][0]) and base_gain[i][0] <= 0):
            num_loss+=1
            gross_loss += base_gain[i][0]
            
    return [(i+1)*standard_1yr, 
            num_gains*standard_1yr,gross_gains*standard_1yr,
            num_loss*standard_1yr,gross_loss*standard_1yr,
            (gross_gains+gross_loss)*standard_1yr]  

def test_strategy_old(y_test,y_pred,buy_adj):    
    #buy_adj = -.42
    low_diff =   (y_pred[:,2:3]+buy_adj) - y_test[:,2:3] 
    open_diff = y_pred[:,0:1] - y_test[:,0:1] 
    adj_diff =  low_diff - open_diff
    adj_low = (y_pred[:,2:3]+buy_adj) - open_diff  
    pot_gain =  y_test[:,3:4] - (y_pred[:,2:3]+buy_adj)
    adj_gain = y_test[:,3:4] - adj_low
    
    i = -1
    num_gains = 0
    gross_gains = 0
    num_loss = 0
    gross_loss = 0
    for row in low_diff:
        i+=1
        if(row>0):
            if(adj_gain[i][0] > 0):
                num_gains+=1
                gross_gains += adj_gain[i][0]
            if(adj_gain[i][0] < 0):
                num_loss+=1
                gross_loss += adj_gain[i][0]
    return [i+1, num_gains,gross_gains,num_loss,gross_loss,(gross_gains+gross_loss)]    
    
def test_single(y_actual,y_predicted,y_ohlc,adjustment, verbose = 0):
    y_actual = y_actual
    y_ohlc = y_ohlc
    y_predicted = y_predicted
    ###########################
    
    
    did_execute = 0
    didnt_execute = 0
    
    executed_gain = 0
    executed_loss = 0
    
    gross_exec_gain = 0
    gross_exec_loss = 0
    
    unexec_gain = 0
    unexec_loss = 0
    
    gross_unexec_gain = 0
    gross_unexec_loss = 0
    
    num_underestimated = 0
    num_overestimated = 0
    
    below_low = 0
    above_high = 0
    
    for i in range(0,len(y_actual)):
        y_pred_adj = y_predicted[i][0]+adjustment
        
        high_actual = y_ohlc[i][1]
        low_actual = y_ohlc[i][2]
        close_actual = y_ohlc[i][3] 
        
        if(y_pred_adj<low_actual):
            below_low += 1
        if(y_pred_adj>high_actual):    
            above_high += 1
        
        if(y_pred_adj>low_actual and y_pred_adj<high_actual):
            did_execute+=1
            if(y_pred_adj<close_actual):
                executed_gain+=1
                gross_exec_gain += close_actual-y_pred_adj
                if(verbose == 1):
                    print("Base Pred: "+str(y_predicted[i][0])+" Adj Pred: "+str(y_pred_adj)+" EXEC GAIN")
                    print(str(y_ohlc[i])+"with net of "+str(close_actual-y_pred_adj))
            else:
                executed_loss+=1
                gross_exec_loss += close_actual-y_pred_adj
                if(verbose == 1):
                    print("Base Pred: "+str(y_predicted[i][0])+" Adj Pred: "+str(y_pred_adj)+" EXEC LOSS")
                    print(str(y_ohlc[i])+"with net of "+str(close_actual-y_pred_adj))
                
            
        else:
            didnt_execute+=1
            if(y_pred_adj<close_actual):
                unexec_gain+=1
                gross_unexec_gain += close_actual-y_pred_adj
                if(verbose == 1):
                    print("Base Pred: "+str(y_predicted[i][0])+" Adj Pred: "+str(y_pred_adj)+" MISSED GAIN")
                    print(str(y_ohlc[i])+"with net of "+str(close_actual-y_pred_adj))                
            else:
                unexec_loss+=1
                gross_unexec_loss += close_actual-y_pred_adj
                if(verbose == 1):
                    print("Base Pred: "+str(y_predicted[i][0])+" Adj Pred: "+str(y_pred_adj)+" MISSED LOSS")
                    print(str(y_ohlc[i])+"with net of "+str(close_actual-y_pred_adj))
                
            
            
        
        if(y_pred_adj<y_actual[i][0]):
            num_underestimated+=1
        if(y_pred_adj>y_actual[i][0]):
            num_overestimated+=1
            
    
    result = [ ["Total Datapoints:", len(y_actual)], 
            ["Exec Gain:",executed_gain,gross_exec_gain],
            ["Exec Loss:",executed_loss,gross_exec_loss],
            ["Exec Net:",executed_gain-executed_loss,gross_exec_gain+gross_exec_loss],
            ["Unexec Gain:",unexec_gain,gross_unexec_gain],
            ["Unexec Loss:",unexec_loss,gross_unexec_loss],
            ["Unexec Net:",unexec_gain-unexec_loss,gross_unexec_gain+gross_unexec_loss],
            ["Over/Under Low:",num_overestimated,num_underestimated],
            ["OverHigh/UnderLow:",above_high,(executed_gain+executed_loss),below_low]]
    
    return result

def scan_range(y_actual,y_predicted,y_ohlc,search_range = 1):
    i = -search_range
    best = test_single(y_actual,y_predicted,y_ohlc,i)
    best_i = i
    while(i<search_range):
        current = test_single(y_actual,y_predicted,y_ohlc,i)
        if(current[3][2] > best[3][2]):#[3][2]
            best = current
            best_i = i
        i+=.01
    print("Best Adj:"+str(best_i))
    result = test_single(y_actual,y_predicted,y_ohlc,best_i)
    result.append(["Best Adjustment:",best_i])
    return result

def accuracy_test(close_actual,close_pred):
    
    pred_below = 0
    pred_above = 0
    within_1 = 0
    within_05 = 0
    within_04 = 0
    within_03 = 0
    within_02 = 0
    within_01 = 0
    within_005 = 0
    standard_1yr = (100/len(close_actual))
    for i in range(len(close_actual)):
        if(close_pred[i] > close_actual[i]):
            pred_above+=1
        if(close_pred[i] < close_actual[i]):
            pred_below+=1
        if(close_pred[i]>close_actual[i]-1 and close_pred[i]<close_actual[i]+1):
            within_1+=1
        if(close_pred[i]>close_actual[i]-.5 and close_pred[i]<close_actual[i]+.5):
            within_05+=1
        if(close_pred[i]>close_actual[i]-.4 and close_pred[i]<close_actual[i]+.5):
            within_04+=1
        if(close_pred[i]>close_actual[i]-.3 and close_pred[i]<close_actual[i]+.5):
            within_03+=1            
        if(close_pred[i]>close_actual[i]-.2 and close_pred[i]<close_actual[i]+.5):
            within_02+=1           
        if(close_pred[i]>close_actual[i]-.1 and close_pred[i]<close_actual[i]+.1):
            within_01+=1
        if(close_pred[i]>close_actual[i]-.05 and close_pred[i]<close_actual[i]+.05):
            within_005+=1
    
    
    
    
    return [["Prediction Above ", pred_above*standard_1yr],
            ["Prediction Below ", pred_below*standard_1yr],
            ["Predictions within 1: ", within_1*standard_1yr],
            ["Predictions within .5: ", within_05*standard_1yr],
            ["Predictions within .4: ", within_04*standard_1yr],
            ["Predictions within .3: ", within_03*standard_1yr],
            ["Predictions within .2: ", within_02*standard_1yr],
            ["Predictions within .1: ", within_01*standard_1yr],
            ["Predictions within .05: ", within_005*standard_1yr]]
    
def buypredlow_sellclose(actual, pred):
    pred_low = pred[:,1:2]
    close_actual = actual[:,2:3]
    low_actual = actual[:,1:2]
    high_actual = actual[:,0:1]
    
    low_gain_execution = 0
    gross_low_gain = 0
    
    low_loss_execution = 0
    gross_low_loss = 0
    
    low_failed_execution = 0
    
    net_gain = 0
    
    percent_ratio = 100/len(pred_low)
    
    for i in range(len(pred_low)):
        if(low_actual[i]<pred_low[i] and high_actual[i]>pred_low[i]):
            if(pred_low[i]<close_actual[i]):
                low_gain_execution += 1
                gross_low_gain += close_actual[i]-pred_low[i]
                net_gain += close_actual[i]-pred_low[i]
            else:
                low_loss_execution += 1
                gross_low_loss += close_actual[i]-pred_low[i]
                net_gain += close_actual[i]-pred_low[i]

        else:
            low_failed_execution += 1
            
    return [["low_gain_execution%: ",low_gain_execution*percent_ratio],
            ["gross_low_gain: ",gross_low_gain],
            ["low_loss_execution%: ",low_loss_execution*percent_ratio],
            ["gross_low_loss: ",gross_low_loss],
            ["low_failed_execution%: ",low_failed_execution*percent_ratio],
            ["net_gain: ",net_gain]]




def buyopen_sellpredhigh_stopclose(actual, pred):
    
    pred_high = pred[:,0:1]
    high_actual = actual[:,0:1]
    close_actual = actual[:,2:3]
    
    high_execution = 0
    gross_high_gain = 0
    
    close_gain_execution = 0
    close_loss_execution = 0
    close_total_execution = 0
    close_gain = 0 
    close_loss = 0
    
    net_gain = 0
    
    percent_ratio = 100/len(pred_high)
    
    for i in range(len(pred_high)):
        if(high_actual[i]>pred_high[i]):
            high_execution +=1
            gross_high_gain += pred_high[i]
            net_gain += pred_high[i]
        else:
            if(close_actual[i]>0):
                close_gain_execution += 1
                close_total_execution += 1
                close_gain += close_actual[i]
                net_gain += close_actual[i]
            else:
                close_loss_execution += 1
                close_total_execution += 1
                close_loss += close_actual[i]
                net_gain += close_actual[i]    
    return [["high_execution %: ",high_execution*percent_ratio],
            ["gross_high_gain: ",gross_high_gain],
            ["close_gain_execution %: ",close_gain_execution*percent_ratio],
            ["close_gain: ",close_gain],
            ["close_loss_execution %: ",close_loss_execution*percent_ratio],
            ["close_loss: ",close_loss],
            ["close_total_execution %: ",close_total_execution*percent_ratio],
            ["net_gain: ",net_gain],]
                

def buyopen_sellpredhigh_stoppredlow(actual, pred, high_adj = 0):
    
    pred_high = pred[:,0:1]+high_adj
    pred_low = pred[:,1:2]
    high_actual = actual[:,0:1]
    low_actual = actual[:,1:2]
    close_actual = actual[:,2:3]
    
    high_execution = 0
    gross_high_gain = 0
    
    low_loss_execution = 0
    low_loss = 0
    
    close_gain_execution = 0
    close_loss_execution = 0
    close_total_execution = 0
    close_gain = 0 
    close_loss = 0
    
    
    net_gain = 0
    
    percent_ratio = 100/len(pred_low)
    
    for i in range(len(pred_high)):
        if(high_actual[i]>pred_high[i]):
            high_execution +=1
            gross_high_gain += pred_high[i]
            net_gain += pred_high[i]
        else:
            
            if(pred_low[i]>low_actual[i]):
                low_loss_execution += 1
                low_loss += pred_low[i]
                net_gain += pred_low[i]  
            else:
                if(close_actual[i]>0):
                    close_gain_execution += 1
                    close_total_execution += 1
                    close_gain += close_actual[i]
                    net_gain += close_actual[i]
                else:
                    close_loss_execution += 1
                    close_total_execution += 1
                    close_loss += close_actual[i]
                    net_gain += close_actual[i]     
                
                
    return [["high_execution %: ",high_execution*percent_ratio],
            ["gross_high_gain: ",gross_high_gain],
            ["low_loss_execution %: ",low_loss_execution*percent_ratio],
            ["low_loss: ",low_loss],
            ["close_gain_execution %: ",close_gain_execution*percent_ratio],
            ["close_gain: ",close_gain],
            ["close_loss_execution %: ",close_loss_execution*percent_ratio],
            ["close_loss: ",close_loss],
            ["close_total_execution %: ",close_total_execution*percent_ratio],
            ["net_gain: ",net_gain]]

def check_bb(actual,predicted):
    
    bull = 0
    bull_correct = 0
    bull_incorrect = 0
    bear = 0
    bear_correct = 0
    bear_incorrect = 0
    
    for i in range(len(actual)):
        
        if(actual[i][0] < 0):
            bear +=1
            if(predicted[i][0] < 0):
                bear_correct +=1
            else:
                bear_incorrect +=1
                
        if(actual[i][0] > 0):
            bull +=1
            if(predicted[i][0] > 0):
                bull_correct +=1
            else:
                bull_incorrect +=1
            
    return [["Correct Bull: ",bull_correct/bull],
            ["Incorrect Bull: ",bull_incorrect/bull],
            ["Correct Bear: ",bear_correct/bear],
            ["Incorrect Bear: ",bear_incorrect/bear]]
    
def check_bb2(actual,predicted):
    
    bull = 0
    bull_correct = 0
    bull_incorrect = 0
    bear = 0
    bear_correct = 0
    bear_incorrect = 0
    
    for i in range(len(actual)):
        
        if(predicted[i][0] < 0):
            bear +=1
            if(actual[i][0] < 0):
                bear_correct +=1
            else:
                bear_incorrect +=1
                
        if(predicted[i][0] > 0):
            bull +=1
            if(actual[i][0] > 0):
                bull_correct +=1
            else:
                bull_incorrect +=1
            
    return [["Correct Bull2: ",bull_correct/bull],
            ["Incorrect Bull2: ",bull_incorrect/bull],
            ["Correct Bear2: ",bear_correct/bear],
            ["Incorrect Bear2: ",bear_incorrect/bear]]    
    
def main():
    print("test")
    
if __name__ == "__main__":
    main()