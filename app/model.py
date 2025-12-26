import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from logger_common import *

class FC(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.5):
        super(FC, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_size, output_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        return self.sequence(x)

class MLP(nn.Module):
    def __init__(self, sizes=[16, 64, 1], dropout_rate=0.5):
        super(MLP, self).__init__()
        n = len(sizes)
        assert n >= 2, 'Must have at least 1 input layer and 1 output layer'
        self.model = nn.Sequential(*[
            FC(input_size, output_size, dropout_rate) 
                for input_size, output_size in zip(sizes[:n-2], sizes[1:n-1])
        ])
        self.model.append(nn.Linear(sizes[n-2], sizes[n-1], dtype=torch.float32))

    def forward(self, x):
        # logger.info(x.shape)
        return self.model(x)

class Scaler:
    def __init__(self):
        self.mu = None
        self.std = None

    def __call__(self, x, inverse=False, fit=False, default_std=1):
        assert isinstance(x, torch.Tensor), 'x must be of torch.Tensor type'
        with torch.no_grad():
            if x.ndim == 1:
                x = x.view(-1, 1)
            if fit:
                self.mu = x.mean(dim=1, keepdim=True)
                self.std = x.std(dim=1, keepdim=True)
                self.std[self.std == 0] = default_std

            if inverse:
                return x * self.std + self.mu
            return (x - self.mu)/self.std

def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(torch.tensor(data[i:i + window_size], dtype=torch.float32))
        y.append(torch.tensor(data[i + window_size], dtype=torch.float32))
    return torch.stack(X, dim=0), torch.stack(y, dim=0)

def train_model(model, scaler, X_train, y_train, epochs=100, lr=0.01):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train = scaler(X_train, fit=True)
    y_train = scaler(y_train)
    assert X_train.dtype == torch.float32
    assert y_train.dtype == torch.float32
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"Model loaded from {path}")
    else:
        logger.info(f"Model path {path} does not exist.")

# def predict(model, scaler, x):
#     model.eval()
#     if not isinstance(x, torch.Tensor):
#         x = torch.tensor(x, dtype=torch.float32)
#     if x.ndim == 1:
#         x = x.reshape(1,-1)
#     with torch.no_grad():
#         x = scaler(x, fit=True)
#         y = model(x)
#         return scaler(y, inverse=True)

def predict(model, scaler, x, steps=1):
    model.eval()
    assert steps >= 1, 'Prediction horizon must be at least 1'
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dtype != torch.float32:
        x = x.to(dtype=torch.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    for step in range(steps):
        with torch.no_grad():
            x_in = scaler(x[:, step:], fit=True)
            y_pred = model(x_in)
            y_pred = torch.round(F.relu(scaler(y_pred, inverse=True)))
            x = torch.cat([x, y_pred], dim=1)
    return x[:, -steps:]

