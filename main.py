import math

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
import torch

from sklearn.model_selection import train_test_split

from tqdm import tqdm

# 读取数据
df = pd.read_csv('JSGF001_2019.csv')
df.set_index('时间',inplace=True)

time_steps = 96
input_dim = 22
data_size = int(df.shape[0] / 96)

# 构造数据集
X = []
y = []
start_index = 0
for i in range(data_size):
    X.append(df.iloc[start_index:start_index+time_steps, 0:22].to_numpy())
    y.append(df.iloc[start_index:start_index+time_steps, 22].to_numpy())

    start_index = start_index + time_steps
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)

# 数据标准化--对每一列
for j in range(input_dim):
    mean = np.mean(X[:, j])
    var = np.var(X[:, j])
    X[:, j] = (X[:, j] - mean) / math.sqrt(var)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 定义模型
class LSTM_Network(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * output_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output.reshape((output.shape[0], -1))
        output = self.fc(output)
        return output

# hyper parameters
hidden_size = 4
n_epoch = 30
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

# 构造模型
model = LSTM_Network(input_dim, hidden_size, time_steps, num_layers=2).double().to(device)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# train
for epoch in range(n_epoch):
    model = model.train()
    pbar = tqdm(train_data_loader)
    pbar.set_description("Epoch {}:".format(epoch))
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        predict = model(inputs)
        loss = loss_func(predict, labels)

        pbar.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
    
    model = model.eval()
    X_test = X_test.to(device)
    val = model(X_test)
    val = val.data.cpu().numpy()
    score = np.sqrt(metrics.mean_squared_error(y_test,val))
    print("Eval RMSE: {}".format(score))

df = pd.read_csv('JSGF001_2020.csv')
df.set_index('时间',inplace=True)

# 构造数据集
X = []
y = []
start_index = 0
for i in range(data_size):
    X.append(df.iloc[start_index:start_index+time_steps, 0:22].to_numpy())
    y.append(df.iloc[start_index:start_index+time_steps, 22].to_numpy())

    start_index = start_index + time_steps

X = np.array(X)
y = np.array(y)

print(X.shape, y.shape)

for j in range(input_dim):
    mean = np.mean(X[:, j])
    var = np.var(X[:, j])
    X[:, j] = (X[:, j] - mean) / math.sqrt(var)

test = torch.from_numpy(X).to(device)
model = model.eval()
test_predict = model(test)
test_predict = test_predict.data.cpu().numpy()
score = np.sqrt(metrics.mean_squared_error(y,test_predict))
print("Test RMSE: {}".format(score))




