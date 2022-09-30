
#adding VIX data to predictive model, max strategy optimization seems to have jumped
#seems to be more varied, low values are lower 


from boilerplate import *
import tensorflow

################################### PROJECT ###################################
project_name = "SPKE_RNN_DAILY_V1_M1"
project = "C:/Projects/"+project_name
mdatadir = project+"/mdata"
zipmdatadir = project+"/zip" 
jsondatadir =project+ "/jsondata"
processedcsvdir = project+"/cleanCSV"
h5dir = project+"/h5"
h5_name = '/'+project_name+'_RECENT_BEST.h5'

p = projectv2.Project(project)
p.addDir(mdatadir)
p.addDir(zipmdatadir)
p.addDir(jsondatadir)
p.addDir(processedcsvdir)
p.addDir(h5dir)

now = datetime.datetime.now()

################################# TIME SERIES #################################

symbols = ["SPKE", "VIX"]
ts_df = test = time_series_queries(symbols)
ts_df = ts_df[pd.notnull(ts_df['open'])] #removing null value
#ts_df = ts_df.reindex(index=ts_df.index[::-1])#reversing
ts_index = ts_df.index

################################## INDICATOR ##################################

#reversed_arr = ts_nparray[::-1] #reversing np array,not needed if reversing df

process_start = time.time()

open_values = np.array(ts_df['open']).astype("float")
high_values =  np.array(ts_df['high']).astype("float")
low_values = np.array(ts_df['low']).astype("float")
close_values = np.array(ts_df['close']).astype("float")
volume = np.array(ts_df['volume']).astype("float")

indicators = get_ta_indicators(open_values, high_values, low_values, close_values, volume, low_version = True)
indicator_df = combine_indicators(indicators, ts_index)

process_end = time.time()
print("Indicator Processes took"+str(process_end - process_start)+" seconds or "+str((process_end - process_start)/60)+" minutes")

############################## DATA MANIPULATION ##############################
#########ALWAYS clean data before manipulating, or weirdness can happen########
    
combined_df = ts_df.join(indicator_df, how='outer', rsuffix='_ind')
combined_df = combined_df.dropna(how='any')
combined_df = combined_df.drop(['volume_VIX'], axis=1)
#day_in_progress = False
#previous = dataset[-2:-1]
#latest = dataset[-1:]
#TRUNCATING DATA BEFORE 2010 MOVE BEFORE NASDAQ
truncate = False
dataset = combined_df#.stock split in May 2000, can add more if want later
if(truncate):
    dataset = combined_df.truncate(before='2010-01-04')

dataset = dataset.iloc[:,:].values.astype('float32')

############################### Setting Up X, y ###############################

number_datapoints = 5
scx = MinMaxScaler()
scy = MinMaxScaler()
dataset_scaled = scx.fit_transform(dataset)



X = []
y = []
X_scaled = []
y_scaled = []
for i in range(number_datapoints, dataset_scaled.shape[0]):
    X_scaled.append(dataset_scaled[i-number_datapoints:i, :])
    #y.append(dataset[i, 0:4]) 
    y.append(dataset[i, 1:4]-dataset[i, 0:1])
    X.append(dataset[i-number_datapoints:i, :])
    
X = np.array(X)
y = np.array(y)

X_scaled, y_scaled = np.array(X_scaled), scy.fit_transform(np.array(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

X_train_scaled = []
for i in range(len(X_train)):
    X_train_scaled.append(scx.transform(X_train[i]))
X_train_scaled = np.array(X_train_scaled)

X_test_scaled = []
for i in range(len(X_test)):
    X_test_scaled.append(scx.transform(X_test[i]))
X_test_scaled = np.array(X_test_scaled)

y_train_scaled = scy.transform(y_train)
y_test_scaled = scy.transform(y_test)

Count_Row=combined_df.shape[0]
Count_Col=combined_df.shape[1]

############################### ANN PARAMATERS ################################
#parameters = {'inputLength':[inputLength],
#              'neurons':[numNodes],
#              'batch_size':[5,25,50],
#              'epochs':[100,250,500],
#              'dropout_rate':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#              'activation':['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
#              'optimizer':['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}



with tensorflow.device("/GPU:0"):
    process_start = time.time()
    
    dropout = 0.4
    neurons = 1000 
    activation = 'tanh'
    output_size = y.shape[1]
    #int(round((inputLength-output_size)/2))
    epochs = 300
    batch_size = 256
    regressor = Sequential()
    
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = neurons, activation=activation, return_sequences = True, input_shape = (X_train.shape[1], Count_Col)))
    regressor.add(Dropout(dropout))
    
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = neurons, activation=activation, return_sequences = True))
    regressor.add(Dropout(dropout))
    
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = neurons, activation=activation, return_sequences = True))
    regressor.add(Dropout(dropout))
    
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = neurons, activation=activation))
    regressor.add(Dropout(dropout))
    
    # Adding the output layer
    regressor.add(Dense(units = output_size, activation='linear'))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    regressor.fit(X_train_scaled, y_train_scaled, validation_split=0.2, epochs = epochs, batch_size = batch_size, verbose = 1) #128,256,512,1024
    #test = regressor.fit(X_train_scaled, y_train_scaled, validation_split=0.2, epochs = 10, batch_size = batch_size, verbose = 1)
    
    regressor.save(h5dir+"/TRUN"+str(truncate)+"_DIFF_"+str(dropout)+
                   "DROP_"+str(number_datapoints)+"DPRNN_"+str(neurons)+"UNIT_"
                   +str(epochs)+"EPOCH_"+str(batch_size)+"BATCH_"
                   +str(now.year)+"-"+str(now.month)+"-"+str(now.day)+".h5")
    process_end = time.time()
    print("Training Processes took "+str(process_end - process_start)+" seconds or "+str((process_end - process_start)/60)+" minutes")

################################## PREDICTION #################################
y_pred = scy.inverse_transform(regressor.predict(X_scaled))
y_test_pred = scy.inverse_transform(regressor.predict(X_test_scaled))

#test_data = test_optimize_strategy(y_test,y_test_pred,True)
predict_X = scy.inverse_transform(regressor.predict(X_scaled))
predict_latest = predict_X[-1:]
predict_previous = predict_X[-2:-1]
print("predict_previous Forecast:")
print(predict_previous)
print("predict_latest Forecast:")
print(predict_latest)
next_day = []
next_day.append(dataset_scaled[-number_datapoints:, :]) 
next_day = np.array(next_day)
predict_next = scy.inverse_transform(regressor.predict(next_day))
print("predict_next Forecast:")
print(predict_next)

############################## DIFFERENCE #####################################

high_accuracy = accuracy_test(y[:,0:1],predict_X[:,0:1])
low_accuracy = accuracy_test(y[:,1:2],predict_X[:,1:2])
close_accuracy = accuracy_test(y[:,2:3],predict_X[:,2:3])


#print('\n'.join('{}: {}'.format(*k) for k in enumerate(open_accuracy)))
print("high_accuracy =")
print('\n'.join('{}: {}'.format(*k) for k in enumerate(high_accuracy)))
print("low_accuracy =")
print('\n'.join('{}: {}'.format(*k) for k in enumerate(low_accuracy)))
print("close_accuracy =")
print('\n'.join('{}: {}'.format(*k) for k in enumerate(close_accuracy)))






print("TEST "+str(len(y_test)))
print("buyopen_sellpredhigh_stopclose")
bo_sph_sc = buyopen_sellpredhigh_stopclose(y_test, y_test_pred)
print('\n'.join('{}: {}'.format(*k) for k in enumerate(bo_sph_sc)))

print("buyopen_sellpredhigh_stoppredlow")
bo_sph_spl = buyopen_sellpredhigh_stoppredlow(y_test, y_test_pred, high_adj = 0)
print('\n'.join('{}: {}'.format(*k) for k in enumerate(bo_sph_spl)))

print("buypredlow_sellclose")
bpl_sc = buypredlow_sellclose(y_test, y_test_pred)
print('\n'.join('{}: {}'.format(*k) for k in enumerate(bpl_sc)))

days_back = 130
print("RECENT "+str(len(y[-days_back:,:])))
print("buyopen_sellpredhigh_stopclose")
bo_sph_sc = buyopen_sellpredhigh_stopclose(y[-days_back:,:], y_pred[-days_back:,:])
print('\n'.join('{}: {}'.format(*k) for k in enumerate(bo_sph_sc)))

print("buyopen_sellpredhigh_stoppredlow")
bo_sph_spl = buyopen_sellpredhigh_stoppredlow(y[-days_back:,:], y_pred[-days_back:,:], high_adj = 0)
print('\n'.join('{}: {}'.format(*k) for k in enumerate(bo_sph_spl)))

print("buypredlow_sellclose")
bpl_sc = buypredlow_sellclose(y[-days_back:,:], y_pred[-days_back:,:])
print('\n'.join('{}: {}'.format(*k) for k in enumerate(bpl_sc)))









################################### REPORTING ##################################
test_data_base = test_strategy(y_test,y_test_pred,0)
test_data_base_open_adj = test_strategy(y_test,y_test_pred,0,True)
test_data_opt_no_open_adjust = test_optimize_strategy(y_test,y_test_pred)
test_data_opt_open_adj = test_optimize_strategy(y_test,y_test_pred,True)

print("Test Data Base:")
print(test_data_base)
print("Test Data Base With Open Adjustment:")
print(test_data_base_open_adj)     
print("Test Data Optimized No Open Adjustment:")
print(test_data_opt_no_open_adjust) 
print("Test Data Optimized with Open Adjustment:")
print(test_data_opt_open_adj) 
################################### ABSOLUTE ##################################
open_accuracy = accuracy_test(y[:,0:1],predict_X[:,0:1])
high_accuracy = accuracy_test(y[:,1:2],predict_X[:,1:2])
low_accuracy = accuracy_test(y[:,2:3],predict_X[:,2:3])
close_accuracy = accuracy_test(y[:,3:4],predict_X[:,3:4])

print("open_accuracy =")
print('\n'.join('{}: {}'.format(*k) for k in enumerate(open_accuracy)))
print("high_accuracy =")
print('\n'.join('{}: {}'.format(*k) for k in enumerate(high_accuracy)))
print("low_accuracy =")
print('\n'.join('{}: {}'.format(*k) for k in enumerate(low_accuracy)))
print("close_accuracy =")
print('\n'.join('{}: {}'.format(*k) for k in enumerate(close_accuracy)))

################################  ################################



################################ END REPORTING  ###############################




#X_test_scaled = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], Count_Col))
predict_X_test_scaled = regressor.predict(X_test)
predict_X_test_unscaled = scy.inverse_transform(predict_X_test_scaled)
y_test_unscaled =  scy.inverse_transform(y_test)
y_unscaled = scy.inverse_transform(y)
y_test_unscaled = scy.inverse_transform(y_test)
X_unscaled = []
for i in range(len(X)):
    X_unscaled.append(scx.inverse_transform(X[i]))
X_test_unscaled = []
for i in range(len(X_test)):
    X_test_unscaled.append(scx.inverse_transform(X_test[i]))
predict_X_scaled = regressor.predict(X)
predict_X_unscaled = scy.inverse_transform(predict_X_scaled)
temp = []
temp.append(X[-1,:,:])
temp = np.array(temp)

predict_latest_scaled = regressor.predict(temp)
predict_latest_unscaled = scy.inverse_transform(predict_latest_scaled)

test_single(y_unscaled,predict_X_unscaled,y_ohlc,0.5)




regressor.save(h5dir+h5_name)




########################### OPTIMIZATION MAINTENANCE ###########################

################################### REPORTING ##################################

############################# TUNE FOR LATEST ##################################
########################### ANN OPTIMIZATION TESTS ############################
##################################### END #####################################
