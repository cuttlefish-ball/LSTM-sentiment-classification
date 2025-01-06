from torch import nn
from torch.nn import functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers ,batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output1 = self.fc1(lstm_out[:, -1, :])
        output1=  self.relu(output1)
        output2 = self.fc2(output1)
        output2=  self.relu(output2)
        output3 = self.fc3(output2)
        output=F.softmax(output3, dim= 1)
        return output