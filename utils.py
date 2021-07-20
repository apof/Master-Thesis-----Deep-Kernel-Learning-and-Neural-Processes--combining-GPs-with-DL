import numpy as np
import matplotlib.pyplot as plt


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_directional_accuracy(actual,predicted):
    return np.mean((np.sign(actual[1:] - actual[:-1]) == (np.sign(predicted[1:] - actual[:-1])).astype(int)))

def create_synthetic_timeseries():
	N  = 2000
	b   = 0.3
	c   = 0.45
	tau = 17

	y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
     1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]

	for n in range(17,N+99):
		y.append(y[n] - b*y[n] + c*y[n-tau]/(1+y[n-tau]**10))

	return y


def split_univariate_timeseries(variable,timesteps,future_step=1):
    timeseries = []
    targets = []
    start = 0
    end = timesteps
    inp = []
    lbl = []
    while(end + (future_step-1) <= len(variable)):
        inp.append(variable[start:(end-1)])
        lbl.append(variable[(end-1)+(future_step-1)])
        start += 1
        end += 1
    inp = np.array(inp)
    lbl = np.array(lbl)
    
    return np.array(inp),np.array(lbl)

def split_timeseries(timesteps,target_step,inputs,labels,dates = None):
    start = 0
    end = timesteps
    target_index = target_step + timesteps
    
    inp = []
    lbl = []
    d = []
    while(target_index < len(inputs)):
        inp.append(inputs[start:end])
        lbl.append(labels[target_index])
        d.append(dates[target_index])
        start += 1
        end += 1
        target_index += 1
    inp = np.array(inp)
    lbl = np.array(lbl)
    d = np.array(d)
    
    return np.array(inp),np.array(lbl),np.array(d)

def train_test_split(inputs,labels,dates,window,train_percentage,valid_percentage,horizon_days):
    
    input_batches = []
    test_batches = []
    validation_batches = []
    
    start = 0
    end = window + horizon_days
    train_size = round(train_percentage*window)
    ## define the validation size equal to the test size
    test_size = round((1 - train_percentage)*window)
    validation_size = round((valid_percentage)*test_size)
    test_size = test_size - validation_size
    
    while(end <= inputs.shape[0]):
    	## select the data of the window
        batch_data = inputs[start:end]
        batch_labels = labels[start:end]
        batch_dates = dates[start:end]
        ## exclude some days in the future to avoid look - ahead error
        input_batches.append((batch_data[0:train_size],batch_labels[0:train_size],batch_dates[0:train_size]))
        validation_batches.append((batch_data[(train_size + horizon_days):-test_size],batch_labels[(train_size + horizon_days):-test_size],batch_dates[(train_size + horizon_days):-test_size]))
        test_batches.append((batch_data[(train_size + horizon_days + validation_size):],batch_labels[(train_size + horizon_days + validation_size):],batch_dates[(train_size + horizon_days + validation_size):]))
        
        start += test_size
        end += test_size
        
    return input_batches,validation_batches,test_batches


def plot_directional_results(predictions,labels,index):
    directional_result = (np.sign(labels[1:] - labels[:-1]) == (np.sign(predictions[1:] - labels[:-1])).astype(int))
    color_list = []
    for i,res in enumerate(directional_result):
      if(res == True):
        color_list.append('red')
      else:
        color_list.append('blue')

    plt.figure(figsize=(14,14))
    plt.scatter(index[1:],predictions[1:],color=color_list,s=20)