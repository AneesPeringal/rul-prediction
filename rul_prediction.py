import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import sleep
from os import listdir
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation as FA

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

folder_path = './CMAPSSData/'

listdir(folder_path)

file_name = 'FD001.txt'

df_train = pd.read_csv(folder_path + 'train_' + file_name, header = None, sep = ' ')
df_test = pd.read_csv(folder_path + 'test_'+file_name, header = None, sep = ' ')
rul_test = pd.read_csv(folder_path + 'RUL_'+file_name, header = None)



col_names = []

col_names.append('unit')
col_names.append('time')

for i in range(1,4):
    col_names.append('os'+str(i))
for i in range(1,22):
    col_names.append('s'+str(i))

df_train = df_train.iloc[:,:-2].copy()
df_train.columns = col_names

df_test = df_test.iloc[:,:-2].copy()
df_test.columns = col_names


rul_list = []
engine_numbers = max(df_train['unit'])
for n in np.arange(1,engine_numbers+1):
    
    time_list = np.array(df_train[df_train['unit'] == n]['time'])
    length = len(time_list)
    rul = list(length - time_list)
    rul_list += rul
    
df_train['rul'] = rul_list

rul_list = []

for n in np.arange(1,engine_numbers+1):
    
    time_list = np.array(df_test[df_test['unit'] == n]['time'])
    length = len(time_list)
    rul_val = rul_test.iloc[n-1].item()
    rul = list(length - time_list + rul_val)
    rul_list += rul

df_test['rul'] = rul_list




drop_cols1 = ['os3','s1','s5','s6','s10','s16','s18','s19']

df_train = df_train.drop(drop_cols1, axis = 1)
df_test = df_test.drop(drop_cols1, axis = 1)

minmax_dict = {}

for c in df_train.columns:
    if 's' in c:
        minmax_dict[c+'min'] = df_train[c].min()
        minmax_dict[c+'max']=  df_train[c].max()
        
for c in df_train.columns:
    if 's' in c:
        df_train[c] = (df_train[c] - minmax_dict[c+'min']) / (minmax_dict[c+'max'] - minmax_dict[c+'min'])
        
for c in df_test.columns:
    if 's' in c:
        df_test[c] = (df_test[c] - minmax_dict[c+'min']) / (minmax_dict[c+'max'] - minmax_dict[c+'min'])

def smooth(s, b = 0.98):

    v = np.zeros(len(s)+1) #v_0 is already 0.
    bc = np.zeros(len(s)+1)

    for i in range(1, len(v)): #v_t = 0.95
        v[i] = (b * v[i-1] + (1-b) * s[i-1]) 
        bc[i] = 1 - b**i

    sm = v[1:] / bc[1:]
    
    return sm

s = [1,2,3,4,5]

#Smoothing each time series for each engine in both training and test sets

for c in df_train.columns:
    
    if 's' in c:
        sm_list = []

        for n in np.arange(1,101):
            s = np.array(df_train[df_train['unit'] == n][c].copy())
            sm = list(smooth(s, 0.98))
            sm_list += sm
        
        df_train[c+'_smoothed'] = sm_list
        
for c in df_test.columns:
    
    if 's' in c:
        sm_list = []

        for n in np.arange(1,101):
            s = np.array(df_test[df_test['unit'] == n][c].copy())
            sm = list(smooth(s, 0.98))
            sm_list += sm
        
        df_test[c+'_smoothed'] = sm_list
#Remove the original series

for c in df_train.columns:
    if ('s' in c) and ('smoothed' not in c):
        df_train[c] = df_train[c+'_smoothed']
        df_train.drop(c+'_smoothed', axis = 1, inplace = True)
        
for c in df_test.columns:
    if ('s' in c) and ('smoothed' not in c):
        df_test[c] = df_test[c+'_smoothed']
        df_test.drop(c+'_smoothed', axis = 1, inplace = True)

n_features = len([c for c in df_train.columns if 's' in c]) #plus one for time
window = 20
print(f'number of features: {n_features}, window size: {window}')
np.random.seed(5)
units = np.arange(1,101)
train_units = list(np.random.choice(units, 80, replace = False))
val_units = list(set(units) - set(train_units))
print(val_units)

train_data = df_train[df_train['unit'].isin(train_units)].copy()
val_data = df_train[df_train['unit'].isin(val_units)].copy()

train_indices = list(train_data[(train_data['rul'] >= (window - 1)) & (train_data['time'] > 10)].index)
val_indices = list(val_data[(val_data['rul'] >= (window - 1)) & (val_data['time'] > 10)].index)

class data(Dataset):
    
    def __init__(self, list_indices, df_train):
        
        self.indices = list_indices
        self.df_train = df_train
        
    def __len__(self):
        
        return len(self.indices)
    
    def __getitem__(self, idx):
        
        ind = self.indices[idx]
        X_ = self.df_train.iloc[ind : ind + 20, :].drop(['time','unit','rul'], axis = 1).copy().to_numpy()
        y_ = self.df_train.iloc[ind + 19]['rul']
        
        return X_, y_
    
torch.manual_seed(5)
    
train = data(train_indices, df_train)
val = data(val_indices, df_train)

trainloader = DataLoader(train, batch_size = 64, shuffle = True)
valloader = DataLoader(val, batch_size = len(val_indices), shuffle = True)

units = np.arange(1,101)

class test(Dataset):
    
    def __init__(self, units, df_test):
        
        self.units = units
        self.df_test = df_test
        
    def __len__(self):
        
        return len(self.units)
    
    def __getitem__(self, idx):
        
        n = self.units[idx]
        U = self.df_test[self.df_test['unit'] == n].copy()
        X_ = U.reset_index().iloc[-20:,:].drop(['time','index','unit','rul'], axis = 1).copy().to_numpy()
        y_ = U['rul'].min()
        
        return X_, y_
    
test = test(units, df_test)
testloader = DataLoader(test, batch_size = 100)


## Custom loss function
class CustomLoss(nn.Module):
    def __init__(self, alpha):
        super(CustomLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2) + self.alpha*(torch.mean(torch.relu(predictions-targets)))
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cuda'
# device = 'cpu'


class LSTMRegressor(nn.Module):
    
    def __init__(self, n_features, hidden_units):
        super().__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.n_layers = 1
        self.lstm = nn.LSTM(input_size = n_features, hidden_size = self.hidden_units, batch_first = True, num_layers = self.n_layers)
        self.linear1 = nn.Linear(in_features=self.hidden_units, out_features=12)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=12, out_features=12)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=12, out_features=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_units,device=x.device).requires_grad_()
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_units,device=x.device).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear1(hn[0])
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out).flatten()
        
        return out
    

learning_rate = 0.001
n_hidden_units = 12

torch.manual_seed(15)
list_alpha = [0, 0.2, 0.4, 23, 131, 267, 540, 814]


for alpha in list_alpha:
    model = LSTMRegressor(n_features, n_hidden_units).to(device)
    # loss_fn = nn.MSELoss()
    # alpha = 814
    loss_fn = CustomLoss(alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                                
    ks = [key for key in model.state_dict().keys() if 'linear' in key and '.weight' in key]

    for k in ks:
        nn.init.kaiming_uniform_(model.state_dict()[k])
        
    bs = [key for key in model.state_dict().keys() if 'linear' in key and '.bias' in key]

    for b in bs:
        nn.init.constant_(model.state_dict()[b], 0)

    def validation():
        
        model.eval()
        X, y = next(iter(valloader))
        X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
        
        with torch.no_grad():
            y_pred = model(X)
            val_loss = loss_fn(y_pred, y).item()
            
        return val_loss

    loss_L1 = nn.L1Loss()
        
    def test():
        model.eval()
        X, y = next(iter(testloader))
        X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
        
        with torch.no_grad():
            y_pred = model(X)
            # y_pred = torch.round(y_pred)
            test_loss_MSE = torch.mean((y_pred - y) ** 2).item() #loss_fn(y_pred, y).item()
            test_loss_L1 = loss_L1(y_pred, y).item()
            test_ASUE = (torch.mean(torch.relu(y-y_pred))).item()
            
        return test_loss_MSE, test_loss_L1, test_ASUE, y_pred, y

    T = []
    V = []
    epochs = 35

    for i in tqdm(range(epochs)):
        
        L = 0
        model.train()
        
        for batch, (X,y) in enumerate(trainloader):
            
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            L += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # alpha = alpha*(1 +((y<y_pred).count_nonzero().item()/len(y)+0.5))
            # loss_fn = CustomLoss(alpha)
            # print(alpha)
        val_loss = validation()
        
        T.append(L/len(trainloader))
        V.append(val_loss)
        (y<y_pred).count_nonzero().item()/len(y)
        # if (i+1) % 10 == 0:
        #     sleep(0.5)
        print(f'epoch:{i+1}, avg_train_loss:{L/len(trainloader)}, val_loss:{val_loss}')
        
    torch.save(model.state_dict(), f'saved_models\\LSTM_{alpha}')

    file = open("Results.txt","a")
    file.write("alpha = "+str(alpha)+"\n")
    mse, l1,asue, y_pred, y = test()

    file.write(f'Test MSE:{round(mse,2)}, L1:{round(l1,2)}, ASUE:{round(asue,2)}')
    file.write(f'MSE of overesitmated: {round(torch.mean(1*(y_pred>y)*(y_pred - y) ** 2).item(),2)}')
    file.write(f'Overestimation rate: {(y<y_pred).count_nonzero().item()/len(y)}'+'\n')
    file.close()