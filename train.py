import math

import pandas as pd
import numpy as np

import torch

from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv('JSGF001_2019.csv')
df.set_index('时间',inplace=True)

time_steps = 96
input_dim = 22
data_size = int(df.shape[0] / 96)

# 构造数据集
X = np.empty((0, time_steps, input_dim))
y = np.empty((0, time_steps))
start_index = 0
for i in range(data_size):
    x_ =  np.expand_dims(df.iloc[start_index:start_index+time_steps, 0:22].to_numpy(), axis=0)
    X = np.concatenate((X, x_), axis=0)
    y_ = np.expand_dims(df.iloc[start_index:start_index+time_steps, 22].to_numpy(), axis=0)
    y = np.concatenate((y, y_), axis=0)

    start_index = start_index + time_steps

print(X.shape, y.shape)

# 数据标准化--对每一列
for j in range(input_dim):
    mean = np.mean(X[:, j])
    var = np.var(X[:, j])
    X[:, j] = (X[:, j] - mean) / math.sqrt(var)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class LSTM_Network(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x

hidden_size = 32
model = LSTM_Network(input_dim, hidden_size, , num_layers=2)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


