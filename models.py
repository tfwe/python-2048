import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256):
        super(ValueNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Reshape to (batch, input_size)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = self.fc4(x)
        return torch.tanh(x)

class PolicyNetwork(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_gru_layers=1):
        super(PolicyNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_gru_layers, batch_first=True)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Reshape to (batch, seq_len, input_size)
        # _, hidden = self.gru(x)
        # x = hidden[-1]  # Take the last layer's hidden state
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)
