import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        # 2 * hidden_size because bidirectional
        self.linear = nn.Linear(2 * hidden_size, 1) # output size is 1 
        
    def forward(self, x):
        '''
        Input:
        x.shape = (batch_size, 370, n_components)
        Each batch has n_components features with a sequence length of 370.
        '''
        out, _ = self.lstm(x) # shape: (batch_size, 370, 2 * hidden_size)
        out = self.linear(out)
        return out