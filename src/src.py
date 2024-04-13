import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
import torch.optim as optim
from fastai.tabular.all import *

# Generate synthetic dataset
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# TensorFlow/Keras model setup
model_keras = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_keras = model_keras.fit(X, y, epochs=100, verbose=0)

# PyTorch model setup
class PyTorchNet(nn.Module):
    def __init__(self):
        super(PyTorchNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

net = PyTorchNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Convert Numpy arrays to Torch tensors
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y).unsqueeze(1)

# Train PyTorch model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(X_torch)
    loss = criterion(outputs, y_torch)
    loss.backward()
    optimizer.step()

# Fast.ai model setup
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Target'] = y

dls = TabularDataLoaders.from_df(df, procs=[Normalize], y_names="Target",
                                 y_block=CategoryBlock(), splits=RandomSplitter()(range_of(df)),
                                 cont_names=['Feature1', 'Feature2'], bs=64)

learn = tabular_learner(dls, layers=[10, 10], metrics=accuracy)
learn.fit_one_cycle(10)
