"""
Module: model_configurations
Author: lpeach
Purpose: To store the model configurations as a module.

This module contains the configurations for the each deep learning model, including parameters and settings
that are used throughout the model's lifecycle. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


class CNN_LSTM_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(CNN_LSTM_BiGRU, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(dropout)
        
        # Calculate the size after CNN layers
        self.cnn_output_size = input_size // 4 * 32  # Adjust based on pooling layers
        
        # LSTM layer
        self.lstm = nn.LSTM(self.cnn_output_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # BiGRU layer
        self.bigru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # CNN forward pass
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Print shape after CNN layers
        #print(f'After CNN: {x.shape}')
        
        # Flatten for LSTM input
        x = x.view(x.size(0), x.size(2), -1)
        
        # Print shape before LSTM
        #print(f'Before LSTM: {x.shape}')
        
        # LSTM forward pass
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        
        # Print shape after LSTM
        #print(f'After LSTM: {x.shape}')
        
        # BiGRU forward pass
        h0 = torch.zeros(self.bigru.num_layers * 2, x.size(0), self.bigru.hidden_size).to(x.device)
        x, _ = self.bigru(x, h0)
        
        # Print shape after BiGRU
        #print(f'After BiGRU: {x.shape}')
        
        # Fully connected layer
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x


# Define the CNN-LSTM Model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, output_size, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Reshape input_shape to fit into Conv2d
        channels, time_steps, features = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Convolutional layers
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, features, time_steps)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Reshape for LSTM input (batch_size, time_steps, channels * height * width)
        batch_size, features, time_steps = x.size()
        x = x.permute(0, 2, 1)  # Reshape back to (batch_size, time_steps, features)
        
        # LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        # FC layer
        out = self.dropout(out[:, -1, :])  # Take only the last time step's output
        out = self.fc(out)
        
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Define the GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out