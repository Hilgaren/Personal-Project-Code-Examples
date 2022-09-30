# -*- coding: utf-8 -*-
"""
Created on Fri May 14 20:47:57 2021

@author: Alex Andrzejek
"""

#import vaex
#import glob2

import tensorflow as tf
#with tf.Session() as sess:
#  devices = sess.list_devices()
#  
#assert tf.test.is_gpu_available()
#assert tf.test.is_built_with_cuda()  
#
#tf.test.gpu_device_name()
#tf.keras.backend.clear_session()
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#with tf.device('/gpu:0'):
#    from tensorflow.python.client import device_lib
#    print(device_lib.list_local_devices())
#print(tf.__version__)
#print(tf.test.is_gpu_available() )
#print(device_lib.list_local_devices())
#tf.test.gpu_device_name()

#tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
#
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from getpass import getpass
from mysql.connector import connect, Error

import datetime 
import calendar
import sys

import dask
import vaex

import time



csv_file = "Anew_universe.csv"


#print("connecting to mysql")
#try:
#    connection = connect(
#            host="localhost",
#            user="XXXX", #input("Enter username: "),
#            password="XXXX", #getpass("Enter password: "),
#            database="market_universe",
#        )
#except Error as e:
#    print(e)
#print("connection established")
#
#cursor = connection.cursor(buffered=True)
#cursor.reset()
#connection.is_connected()
##checking available databases
#print("showing databases")
#show_db_query = "SHOW DATABASES"
#with connection.cursor() as cursor:
#    cursor.execute(show_db_query)
#    for db in cursor:
#        print(db)
#
##creating market universe db
#create_db_query = "CREATE DATABASE market_universe"
#with connection.cursor() as cursor:
#    cursor.execute(create_db_query)
#
##creating market universe db
#create_db_query = "select * from information_schema.columns where table_name='market_universe'"
#with connection.cursor() as cursor:
#    cursor.execute(create_db_query)
#mysqlalchemy bullshit code that shouldnt be needeed
# Create SQLAlchemy engine to connect to MySQL Database


#  
#engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
#				.format(host="localhost", db="market_universe", user="XXXX", pw="XXXX"))
#df.to_sql('market_universe', con = engine, if_exists = 'append', chunksize = 1000,index=False)



#------------------------------------------------------------------------------

############################# BUILD BASIC #####################################

df = pd.read_csv('Anew_universe_3.csv')

#removing negatives from PRC
df["PRC"] = df["PRC"].abs()
#creating adjusted price
df["ADJPRC"] = df["PRC"] / df["CFACPR"]
#creating calculated market cap
df["MKTCAP"] = df["PRC"] * df["SHROUT"]

############################### 1 YR AGO ######################################

#creating boolean 1YRAGO - whether has 1 yr history at current date
permno_list = df["PERMNO"].unique()
tf.device('/device:GPU:0')
with tf.device('/device:GPU:0'):
    yr_ago = pd.DataFrame()
    count = 0
    for permno in permno_list:
        sys.stdout.write('\r'+str(count/len(permno_list)))
        count = count + 1
        
        company_yr_ago = pd.DataFrame()
        company_df = df[df["PERMNO"] == permno]
        company_yr_ago["1YRAGO"] = company_df["PRC"].shift(253).notnull()
        yr_ago = pd.concat([yr_ago, company_yr_ago.copy()])


df["1YRAGO"] = yr_ago["1YRAGO"]


df
df.columns
df.to_csv('Anew_universe_3.csv')
#df.drop(df.columns[0], axis=1).to_csv('Anew_universe_2.csv')

########################### MOMENTUM FEATURES #################################

k_mom_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,42,63,84,105,126,147,168,189,210,231,252]


##LOGIC
#FOR EACH PERMNO *(Company)
#SHIFT INDEX TO CALCULATE RETURN
#FOR EACH K (TIME INTERVAL)
permno_list = df["PERMNO"].unique()
#full momentum df "query result"
momentum_df = pd.DataFrame()
count = 0
for permno in permno_list:
    sys.stdout.write('\r'+str(count/len(permno_list)))
    count = count+1
    company_df = df[df["PERMNO"] == permno]
    #momentum features for specific company
    company_momentum_df = pd.DataFrame()
#   FOR EACH K (TIME INTERVAL)     
    for k in k_mom_list:
        #get t-1 day back
        if 'mom_'+str(k) in company_momentum_df.columns:
            company_momentum_df['mom_'+str(k)] = pd.concat([company_momentum_df['mom_'+str(k)],(company_df["ADJPRC"].shift(1) - company_df["ADJPRC"].shift(k+1)) /company_df["ADJPRC"].shift(k+1)])
        else:
            company_momentum_df['mom_'+str(k)] = (company_df["ADJPRC"].shift(1) - company_df["ADJPRC"].shift(k+1)) /company_df["ADJPRC"].shift(k+1) 
    
    #concat copy of company momentum df to combined momentum df
    momentum_df = pd.concat([momentum_df, company_momentum_df.copy()])

momentum_df.to_csv('momentum.csv')
#del momentum_df
#del df
#momentum_df = pd.read_csv('momentum.csv')

############################## FUNCTIONS ######################################
def isPenultimate(dates_list):
    #date = datetime.datetime.strptime(date_str, '%d/%m/%Y')
#    penultimate_day_of_month = calendar.monthrange(date.year, date.month)[1]-1
#    if date.day == penultimate_day_of_month:
#        return True
#    return False
    output = []
    length = len(dates_list)
    for i in range(length):
        date = dates_list[i]
        if(i+2 < length):
            output.append(dates_list[i].month == dates_list[i+1].month and dates_list[i].month != dates_list[i+2].month)
        elif( i == (length - 1) - 1):
            output.append(1)
        elif(i == (length - 1)):
            output.append(False)
    return output            

#date_str = "31/01/2021"
#def islast(date_str):
#    date = datetime.datetime.strptime(date_str, '%d/%m/%Y')
#    penultimate_day_of_month = calendar.monthrange(date.year, date.month)[1]
#    if date.day == penultimate_day_of_month:
#        return True
#    return False
#islast(date_str)


def year_history(permno, date_str_list):
    for i in date_str_list:
        date = datetime.datetime.strptime(date_str, '%d/%m/%Y')


############################ DATE MANIPULATION ################################

k_mkt_list = [10,11,12,13,14,15,16,17,18,19,20,21,42,63,84,105,126,147,168,189,210,231,252]

###date index sorting
#getting unique dates
dates = df["date"].unique()
#converting to date objects for sorting
dates_list = [datetime.datetime.strptime(date, '%d/%m/%Y').date() for date in dates]
#sorting dates
dates_list.sort()


#managing holidays 
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='1964-01-02', end='2020-12-31').to_pydatetime()
holidays = [holiday.date() for holiday in holidays]
is_holiday = [date in holidays for date in dates_list]

#filter holidays from date list
no_holidays = []
for date in dates_list:
    if(date not in holidays):
        no_holidays.append(date)

#run filtered list for accurate penultimate days
holiday_adjusted_penultimates = isPenultimate(no_holidays)
#build penultimate list
penultimate_dates_list = []
for i in range(len(no_holidays)):
    if(holiday_adjusted_penultimates[i] == True):
        penultimate_dates_list.append(no_holidays[i])
#build penultimate to standard index
is_penultimate = [date in penultimate_dates_list for date in dates_list]


#converting back to strings
dates = [datetime.datetime.strftime(date, '%d/%m/%Y') for date in dates_list]
day_of_week = [date.weekday() for date in dates_list]
#weekend = [day_of_week > 4]

#is_penultimate = isPenultimate(dates_list)
top_500_length = [0] * len(dates)

########################## MARKET FEATURES ####################################
#joining with 1d returns to build index

momentum_df = pd.read_csv('momentum.csv')

df["return"] = momentum_df["mom_1"]
#getting "irregular" dates to diagnose
#index df
market_index_days = pd.DataFrame(index = dates)
#irregular df
irregular = pd.DataFrame(index = dates)
day_df = df[df["date"] == "04/01/1965"]
top_500 = day_df.sort_values(by=['MKTCAP'], ascending = False).head(n=500)
top_500 = top_500[top_500["1YRAGO"] == True]
top_500_permno = top_500["PERMNO"].unique()


count = 252
length = len(dates)
index_value = 1
for i in range(253,length):
    #ETA stuff
    start = time.time()
    count = count + 1
    remaining = length - count
    #
    print("index: "+str(i))
    day = dates[i]
    #if day is a holiday, skip
#    if(datetime.datetime.strptime(day, '%d/%m/%Y').date() in holidays):
#        continue
    #get stock-days
    day_df = df[df["date"] == day]
    #ireggularity statistic day length
    irregular.at[day,'day_length'] = len(day_df)
    irregular.at[day,'holiday'] = day in holidays
    #get top 500 subset for day
    top_500 = day_df[day_df["PERMNO"].isin(top_500_permno)]


    #handling irregulars
    top_500_len = len(top_500)
    if(top_500_len < 400):
        back_day = dates[i-1]
        day_df = df[df["date"] == back_day]
        irregular.at[day,'back_day'] = 1
        #top_500 = day_df[day_df["PERMNO"].isin(top_500_permno)]

    #setting top 500 subset market cap total
    total_mkt_cap = top_500["MKTCAP"].sum()
    #calculating weighted values and summing to create index value
    #market_index_days.at[day,'MKTINDEX'] = (top_500["PRC"]*(top_500["MKTCAP"]/total_mkt_cap)).sum() #SHOULDNT THIS BE CALCULATED WITH TOP 500?
    
    #market return
    vweighted_avg_mkt_return = (top_500["return"]*(top_500["MKTCAP"]/total_mkt_cap)).sum()
    if(top_500_len < 400):
        market_index_days.at[day,'MKTINDEX'] = index_value
        market_index_days.at[day,'MKT_RET_1'] = 0
        market_index_days.at[day,'MKT_STD_1'] = 0
        
    else:
        index_value = index_value + index_value * vweighted_avg_mkt_return
        market_index_days.at[day,'MKTINDEX'] = index_value
        market_index_days.at[day,'MKT_RET_1'] = vweighted_avg_mkt_return
        market_index_days.at[day,'MKT_STD_1'] = top_500["return"].std()

        
    print(" day len: "+str(len(day_df)))
    top_500_len = len(top_500)
    if(is_penultimate[i]):
        print("is penultimate")   
        top_500 = day_df.sort_values(by=['MKTCAP'], ascending = False).head(n=500)
        top_500 = top_500[top_500["1YRAGO"] == True]
        print("top 500 from "+str(top_500_permno)+" to ")
        top_500_permno = top_500["PERMNO"].unique() 
        print(str(top_500_permno))
        top_500_len = len(top_500)
        
    irregular.at[day,'top_500_length'] = len(top_500)
    print(" top 500 len: "+str(top_500_len))
    #ETA stuff
    end = time.time()
    taken = end - start
    print(str(day)+' Took '+str(taken)+' seconds. ETA: '+str(taken*remaining/60)+' minutes - current index '+str(market_index_days.loc[day].values[0]))




market_index_days.to_csv('market_index_days_2.csv')
#market_index_days = pd.read_csv('market_index_days_2.csv')






#---------------------------- DERIVATIVE FEATURES -----------------------------

#k = 10
#market_index_days.columns = ['MKTINDEX','MKT_RET_1']

total_market_days = len(market_index_days)
k_length = len(k_mkt_list)
count = 0    
for k in k_mkt_list:
    
    market_index_days['MKT_RET_'+str(k)] = (market_index_days['MKTINDEX'] - market_index_days["MKTINDEX"].shift(k))/market_index_days["MKTINDEX"].shift(k)
    
    for i in range(252,total_market_days):
        index_name = market_index_days.index[i]
        market_index_days.loc[index_name, 'MKT_STD_'+str(k)] =  market_index_days.iloc[i-k:i, 1].std() 

    count = count + 1
    sys.stdout.write('\r'+str(count/k_length))
        
        

market_index_days.to_csv('market_index_days_2.csv')
#market_index_days = pd.read_csv('market_index_days_2.csv')




####################### MARKET MODEL FEATURES #################################


#with tf.device('/device:GPU:0'):

import cupy as cp

# cp.cuda.Device(0).use()
# x = cp.array([1, 2, 3])
# print(x.device)
# print(cp.cuda.runtime.getDeviceCount())

#only running on not pre-existing permnos
permno_remaining = permno_list
permno_saved = pd.read_csv('permnos.csv').values.tolist()
permno_remaining = [p for p in permno_remaining if p not in permno_saved]#filter(lambda i: i not in permno_saved, permno_remaining)
    
#full market model df "query result" 
market_model_df = pd.read_csv('market_model.csv')  # pd.read_csv('market_model.csv') # pd.DataFrame() 
permnos = permno_saved
    
    
    
count = 0
length = len(permno_list)
    
#permno = permno_remaining[-1]

#for each stock
for permno in permno_remaining:
    
    start = time.time()
    count = count + 1
    remaining = length - count
    
    permnos.append(permno)
    #get stock days
    company_df = df[df["PERMNO"] == permno]
    company_dates = company_df["date"].unique()
    #get stock momentum values
    company_momentum = momentum_df.loc[company_df.index]
    #get market days
    market_subset = market_index_days.loc[company_dates]
        
    #for each k
    for k in k_mkt_list:
        #for each day
        warning_count = 0
        for i in range(len(company_dates)):
            k_company_col = company_momentum.columns.get_loc("mom_1")
            k_market_col  = market_subset.columns.get_loc("MKT_RET_1") #market_index_days.columns.get_loc("MKT_RET_1")
            #get previous k days (including current)
            k_company_momentum = company_momentum.iloc[i-k:i, k_company_col].dropna().to_numpy()
            k_market_momentum  = market_subset.iloc[i-k:i, k_market_col].dropna().to_numpy()
            #calculate coefficients
            if(k_market_momentum.size != 0 and k_market_momentum.size == k and k_company_momentum.size == k):
                X = np.concatenate([np.ones((k,1)),  k_market_momentum.reshape((-1, 1))], axis=1)
                coefs, ssr, _, _ = np.linalg.lstsq(X, k_company_momentum, rcond=None) 
                alpha_hat = coefs[0]
                beta_hat = coefs[1]
                sigma_hat = (ssr / (k-1)) ** 0.5
                #create columns 
                cur_index = company_df.index[i]
                market_model_df.at[cur_index,'alpha_hat_'+str(k)]  = alpha_hat
                market_model_df.at[cur_index,'beta_hat_'+str(k)]   = beta_hat
                market_model_df.at[cur_index,'sigma_hat_'+str(k)]  = sigma_hat
            else:
                cur_index = company_df.index[i]
                market_model_df.at[cur_index,'alpha_hat_'+str(k)]  = np.nan
                market_model_df.at[cur_index,'beta_hat_'+str(k)]   = np.nan
                market_model_df.at[cur_index,'sigma_hat_'+str(k)]  = np.nan
                
            #warnings
#                if(not (k_market_momentum.size == k and k_company_momentum.size == k)):
#                    warning_count = warning_count + 1
#                    sys.stdout.write('\r'+"market - stock list size mismatch .."+str(warning_count))
#                    print()
    #ETA stuff
    end = time.time()
    taken = end - start
    print(str(permno)+' Took '+str(taken)+' seconds. ETA: '+str(taken*remaining/3600)+' hours - current index '+str(count) + " of " + str(length))
    #saving over interval
    if(count%338 == 0):
        print("saving every ~1%")
        market_model_df.to_csv('market_model.csv')
        pd.DataFrame(permnos, columns = ["PERMNO"]).to_csv('permnos.csv')
        print("save complete! resuming...")
    #break
    
market_model_df.to_csv('market_model.csv')
pd.DataFrame(permnos, columns = ["PERMNO"]).to_csv('permnos.csv')
            

################### FROG IN THE PAN / INFO DISCRETENESS #######################


#what we need to do here is 
# 1. for each stock day and 
#    for each k period: determine whether the k-return for that stock is 
#    positive or negative and record 
# 2. for each stock day and
#    for each k period: determine %positive and % negative return for that 
#    k-return period
# 3. use these values for each stock-day to generate the FIP t,i indicator


k_fip_list = [10,21,42,63,84,105,126,147,168,189,210,231,252]
join_table = df.loc[:,['Unnamed: 0','PERMNO','date']].to_numpy()
ret_table  = momentum_df.to_numpy()[:,[0,1]]
mkt_table  = market_index_days.to_numpy()[:,[0,1,2]]

#for each stock
for permno in permno_list:
    #timer stuff
    start = time.time()
    count = count + 1
    remaining = length - count
    
    
    #get stock days
    company_df = df[df["PERMNO"] == permno]
    company_dates = company_df["date"].unique()
    #get stock momentum values
    company_momentum = momentum_df.loc[company_df.index]
    #get market days
    market_subset = market_index_days.loc[company_dates]


                
###############################################################################
#np.warnings.filterwarnings('ignore')
test_cm = company_momentum.iloc[i-k:i, k_company_col]
test_mm = market_subset.iloc[i-k:i, k_market_col]
###############################################################################
#        market_index_days['MKT_STD_'+str(k)] = 
#        
#        if 'MKT_RET_'+str(k) in company_momentum_df.columns:
#            market_index_days['MKT_RET_'+str(k)] = pd.concat([company_momentum_df['MKT_RET_'+str(k)],(company_df["ADJPRC"].shift(1) - company_df["ADJPRC"].shift(k+1)) /company_df["ADJPRC"].shift(k+1)])
#        else:
#            market_index_days['MKT_RET_'+str(k)] = (company_df["ADJPRC"].shift(1) - company_df["ADJPRC"].shift(k+1)) /company_df["ADJPRC"].shift(k+1) 




# market_index_days = pd.read_csv('market_index_days.csv') 

    #sys.stdout.write('\r'+str(day)+' Took '+str(taken)+' seconds. ETA: '+str(taken*remaining/60)+' minutes - current index '+str(market_index_days.loc[day].values[0]))
        
#        if(top_500_length[i] == 0):
#            print("top 500 gen error. length 0. exiting...")
#            condition = 1


test = irregular[irregular["top_500_length"] < 400]
test


day_df  = df[df["date"] == "28/12/1972"] 
top_500 = day_df.sort_values(by=['MKTCAP'], ascending = False).head(n=500)

back_day = dates[i-1]
day_df = df[df["date"] == back_day]
irregular.at[day,'back_day'] = 1
top_500 = day_df[day_df["PERMNO"].isin(top_500_permno)]



top_500 = top_500[top_500["1YRAGO"] == True]

###############################################################################
#condition = 0
#with tf.device('/gpu:0'):
#    #initializing market_days df
#    market_index_days = pd.DataFrame(index = dates)
#    #initializing first day top 500
#    #day = "04/01/1965"
#    day_df = df[df["date"] == "04/01/1965"]
#    #day_df = df[df["date"] == "24/05/1968"]
#    #is_penultimate[2241]
#    #dates[2241]
#    top_500 = day_df.sort_values(by=['MKTCAP'], ascending = False).head(n=500)
#    top_500 = top_500[top_500["1YRAGO"] == True]
#    top_500_permno = top_500["PERMNO"].unique()
#    # total_mkt_cap = day_df["MKTCAP"].sum()
#    # (day_df["PRC"]*(day_df["MKTCAP"]/total_mkt_cap)).sum()
#    
#    count = 252
#    length = len(dates)
#    for i in range(253,length):
#        day = dates[i]
#        start = time.time()
#        count = count + 1
#        remaining = length - count
#        
#        #getting all stock-days for date        
#        day_df = df[df["date"] == day]
#        
#        #if penultimate and holiday
#        if(len(day_df) <= 1000 and is_holiday[i]):
#            print()
#            print("is penultimate + holiday")
#            #go back one day
#            j = 1
#            back_day = dates[i-j]
#            back_day_df = df[df["date"] == day]
#            
#            #in case multiple in a row
#            while(is_holiday[i-j] == 1):
#                print()
#                print(day +" is also holiday, moving back one")
#                
#                back_day = dates[i-j]
#                back_day_df = df[df["date"] == day]
#                j = j + 1                   
#            
#            #handle penultimate
#            #print("is penultimate")
#            top_500 = back_day_df.sort_values(by=['MKTCAP'], ascending = False).head(n=500)
#            top_500 = top_500[top_500["1YRAGO"] == True]
#            print("top 500 from "+str(top_500_permno)+" to ")
#            top_500_permno = top_500["PERMNO"].unique() 
#            print(str(top_500_permno))
#            top_500_length[i] = len(top_500_permno)
#            if(top_500_length[i] == 0):
#                print("top 500 gen error. length 0. exiting...")
#                condition = 1            
#            
#            
#        #if abnormal
#        if(len(day_df) <= 1000):
#            print()
#            print(day +" is irregular, moving back one")
#
#            j = 1
#            back_day = dates[i-j]
#            back_day_df = df[df["date"] == day]
#            
#            #in case multiple in a row
#            while(len(back_day_df) <= 1000):
#                print()
#                print(day +" is also irregular, moving back one")
#                
#                back_day = dates[i-j]
#                back_day_df = df[df["date"] == day]
#                j = j + 1           
#            
#            #set day to previous value
#            market_index_days.at[day,'MKTINDEX'] = market_index_days.at[back_day,'MKTINDEX']
#            #set abnormal day
#            market_index_days.at[day,'ABNORMAL'] = 1 
#            
#        #if penultimate 
#        
#        # normal
#        if()
#        #if abnormal day (partial holiday), go back one
#
#        #
#        day_df = day_df[day_df["PERMNO"].isin(top_500_permno)]
#        
#        #setting 
#        total_mkt_cap = day_df["MKTCAP"].sum()
#        #calculating weighted values and summing to create index value
#        market_index_days.at[day,'MKTINDEX'] = (day_df["PRC"]*(day_df["MKTCAP"]/total_mkt_cap)).sum() #SHOULDNT THIS BE CALCULATED WITH TOP 500?
#        #if penultimate update top 500 index
#        if(is_penultimate[i]):
#            print("is penultimate")
#            top_500 = day_df.sort_values(by=['MKTCAP'], ascending = False).head(n=500)
#            top_500 = top_500[top_500["1YRAGO"] == True]
#            print("top 500 from "+str(top_500_permno)+" to ")
#            top_500_permno = top_500["PERMNO"].unique() 
#            print(str(top_500_permno))
#            top_500_length[i] = len(top_500_permno)
#            if(top_500_length[i] == 0):
#                print("top 500 gen error. length 0. exiting...")
#                condition = 1
#            
#        #TODO: CONFIRM MKT RETURN LOGIC IS CONSISTENT WITH PAPER + DO STANDARD DEVIATIONS AS WELL
#    
#        end = time.time()
#        taken = end - start
#        sys.stdout.write('\r'+str(day)+' Took '+str(taken)+' seconds. ETA: '+str(taken*remaining/60)+' minutes - current index '+str(market_index_days.loc[day].values[0]))
#        if (condition == 1):
#            break
    
    for k in k_mkt_list:
        if 'mkt_ret_'+str(k) in market_index_days.columns:
            market_index_days['mkt_ret_'+str(k)] = pd.concat([market_index_days['mkt_ret_'+str(k)],(market_index_days["MKTINDEX"].shift(1) - market_index_days["MKTINDEX"].shift(k+1)) /market_index_days["MKTINDEX"].shift(k+1)])
        else:
            market_index_days['mkt_ret_'+str(k)] = (market_index_days["MKTINDEX"].shift(1) - market_index_days["MKTINDEX"].shift(k+1)) /market_index_days["MKTINDEX"].shift(k+1) 
       #market_index_days[market_index_days.index.duplicated()]     
        #day_df = day_df.sort_values(by=['MKTCAP'], ascending = False)
        
# index_list = market_index_days.reset_index()["MKTINDEX"]
# test =pd.DataFrame({"MKTINDEX" :index_list,"date":dates,"PENULT": is_penultimate*1, "WEEKDAY": day_of_week})
# test["PENULT"] = test["PENULT"].astype(int)
# test["weekend"] = test["WEEKDAY"] > 4
#index by date

# col 1 is df, col 2+ is features


with tf.device('/gpu:0'):
    
date_str = "30/01/1965"
ispenultimate(date_str)
day_df = df[df["date"] == date_str]
top_500 = day_df.sort_values(by=['MKTCAP'], ascending = False).head(n=500)
top_500



from dateutil.relativedelta import relativedelta

#permno_list = df["PERMNO"].unique()
#momentum_df = pd.DataFrame()
#count = 0
#for i in range(2):
#    permno = permno_list[i]
#    sys.stdout.write('\r'+str(count/len(permno_list)))
#    count = count+1
#    company_df = df[df["PERMNO"] == permno]
#    company_momentum_df = pd.DataFrame()
##   FOR EACH K (TIME INTERVAL)     
#    for k in k_list:
#        #get t-1 day back
#        if 'mom_'+str(k) in company_momentum_df.columns:
#            company_momentum_df['mom_'+str(k)] = pd.concat([company_momentum_df['mom_'+str(k)],(company_df["ADJPRC"].shift(1) - company_df["ADJPRC"].shift(k+1)) /company_df["ADJPRC"].shift(k+1)])
#        else:
#            company_momentum_df['mom_'+str(k)] = (company_df["ADJPRC"].shift(1) - company_df["ADJPRC"].shift(k+1)) /company_df["ADJPRC"].shift(k+1) 
#    
#    #concat copy of company momentum df to combined momentum df
#    momentum_df = pd.concat([momentum_df, company_momentum_df.copy()])







#PLAN OF ATTACK:
#CREATE A NEW TABLE FOR EACH SET OF FEATURES
#JOIN TABLES TOGETHER FOR SUPER TABLE QUERY
#TRY AND KNOCK OUT MOMENTUM TOMORROW
#test_df = pd.DataFrame()
#stock_A = df[df['PERMNO'] == 87432]
#
#for k in k_list:
#        #get t-1 day back
#        if 'mom_'+str(k) in test_df.columns:
#            test_df['mom_'+str(k)] = pd.concat([test_df['mom_'+str(k)],(stock_A["ADJPRC"].shift(1) - stock_A["ADJPRC"].shift(k)) /stock_A["ADJPRC"].shift(k)])
#        else:
#            test_df['mom_'+str(k)] = (stock_A["ADJPRC"].shift(1) - stock_A["ADJPRC"].shift(k)) /stock_A["ADJPRC"].shift(k) 
#



#k = 21
#test_df['mom_'+str(k)] = (stock_A["ADJPRC"].shift(1) - stock_A["ADJPRC"].shift(k)) /stock_A["ADJPRC"].shift(k) 

#test_df = pd.DataFrame()
#stock_A = df[df['PERMNO'] == 87431]
#
#k = 21
#test_df['mom_'+str(k)] = (stock_A["ADJPRC"].shift(1) - stock_A["ADJPRC"].shift(k)) /stock_A["ADJPRC"].shift(k) 
#






#cursor.reset()
#show_db_query = "SHOW DATABASES"
#with connection.cursor() as cursor:
#    cursor.execute(show_db_query)
#    for db in cursor:
#        print(db)


#closing connection
#connection.close()


#------------------------------------------------------------------------------
#dv = vaex.from_csv(csv_file)
#
#    
##dv = vaex.open('hdf5_files/*.hdf5')
#
#dv.head()
#
#
#dv['MKTCAP'] = dv['SHROUT'] * dv['PRC']
#
#dv['MKTCAP']
#
#
#dv.export_hdf5(f'hdf5_files/market_universe_new.hdf5')


k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,42,63,84,105,126,147,168,189,210,231,252]
#
#
#
##get unique values in a column 
#tickers = dv['TICKER'].unique()
#
##for stock
#stock_A = dv[dv['TICKER'] == 'A']
##date
#stock_A[stock_A['date'] == '04/01/1964']
#
#import datetime 
##convert date string to date object
#date_time_str = '04/01/1965'
#date_time_obj = datetime.datetime.strptime(date_time_str, '%d/%m/%Y')
#
##get n day back
#d = datetime.timedelta(days = 1)
#a = date_time_obj - d
#a
#a.toString()
##TIMING OPERATIONS
#import time
#
#start = time.time()
#print("hello")
#end = time.time()
#print("operation took: "+str(end - start)+" seconds")
#
##
#(close_today_minus_1 - close_k_days_ago)/close_k_days_ago)

k_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,42,63,84,105,126,147,168,189,210,231,252]
##LOGIC
##FOR EACH TICKER
#for ticker in tickers:
#    dv_ticker = dv[dv['TICKER'] == ticker]
#    #FOR EACH DATE
#    dates = dv_ticker['date'].unique()
#    for date in dates:
#        #convert date string to date object
#        date_time_obj = datetime.datetime.strptime(date, '%d/%m/%Y')
#        date_str = date_time_obj.strftime("%d/%m/%Y") 
#        
##        ticker_date = dv_ticker[dv_ticker['date'] == date]
##        FOR EACH K (TIME INTERVAL)
#           
#        for k in k_list:
#            #get t-1 day back
#            delta_t = datetime.timedelta(days = 1)
#            t_1_date_obj = date_time_obj - delta_t
#            t_1_date_str = t_1_date_obj.strftime("%d/%m/%Y")
#            #get k day back
#            delta_k = datetime.timedelta(days = (k+1))
#            k_date_obj = date_time_obj - delta_k
#            k_date_str = t_1_date_obj.strftime("%d/%m/%Y")        
#            #column name and values
#            column_name = 'mom_'+str(k)
#            dv_ticker[column_name] = dv_ticker[dv_ticker['date']]     
#           
#        
##COMPUTE AND SAVE RETURN 
#
#stock_A_date =    stock_A[stock_A['date'] == '04/01/1965'] 
#stock_A_minus_k = stock_A[stock_A['date'] == '04/01/1964'] 
#
#stock_A['mom_1yr'] = stock_A[stock_A['date'] == '04/01/1965' ]
