import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import torch
import torch.nn as nn
import torch.optim as optim
from fastai.tabular.all import *
from fastai.metrics import accuracy
import csv

# Generate synthetic dataset
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
y_categorical = tf.keras.utils.to_categorical(y)

# TensorFlow/Keras model setup
model_keras = Sequential([
    Input(shape=(2,)),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(2, activation='softmax')  # Use 2 neurons for categorical output
])
model_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_keras = model_keras.fit(X, y_categorical, epochs=100, verbose=0)

# PyTorch model setup
class PyTorchNet(nn.Module):
    def __init__(self):
        super(PyTorchNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)  # Use 2 outputs for categorical output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.log_softmax(self.fc3(x), dim=1)

net = PyTorchNet()
criterion = nn.NLLLoss()  # Use Negative Log Likelihood Loss for log_softmax
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Convert Numpy arrays to Torch tensors
X_torch = torch.FloatTensor(X)
y_torch = torch.LongTensor(y)  # Use LongTensor for categorical targets

# Train PyTorch model
torch_history = {'accuracy': [], 'loss': []}
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(X_torch)
    loss = criterion(outputs, y_torch)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        accuracy = (preds == y_torch).float().mean()
    torch_history['accuracy'].append(accuracy.item())
    torch_history['loss'].append(loss.item())

# # Fast.ai model setup
# df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
# df['Target'] = y
#
# dls = TabularDataLoaders.from_df(df, procs=[Normalize], y_names="Target",
#                                  y_block=CategoryBlock(), splits=RandomSplitter()(range_of(df)),
#                                  cont_names=['Feature1', 'Feature2'], bs=64)
#
# # Ensure that the accuracy metric is a function
# learn = tabular_learner(dls, layers=[10, 10], metrics=[accuracy])
# learn.fit_one_cycle(10)

# Save the performance plot
plt.figure(figsize=(10, 5))
plt.plot(history_keras.history['accuracy'], label='Keras Accuracy', color='red')
plt.plot(torch_history['accuracy'], label='PyTorch Accuracy', color='blue')
# plt.plot([x['accuracy'].item() for x in learn.recorder.values], label='FastAI Accuracy', color='green')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('IMG.png')
plt.close()

# Save the logs

with open('log.txt', 'w', newline='') as log_file:
    log_writer = csv.writer(log_file)

    # Write headers for the CSV file
    log_writer.writerow(['Model', 'Epoch', 'Accuracy', 'Loss'])

    # Write Keras model history
    # Properly access the .history attribute of the History object
    for i in range(len(history_keras.history['accuracy'])):
        log_writer.writerow(['Keras', i, history_keras.history['accuracy'][i], history_keras.history['loss'][i]])

    # Write PyTorch model history
    for i in range(len(torch_history['accuracy'])):
        log_writer.writerow(['PyTorch', i, torch_history['accuracy'][i], torch_history['loss'][i]])

    # log_file.write("FastAI History:\n")
    # log_file.write(f"Accuracy: {[x['accuracy'].item() for x in learn.recorder.values]}\n")
    # log_file.write(f"Loss: {[x['loss'] for x in learn.recorder.values]}\n")

print("Training complete. The performance plot and logs have been saved.")
