import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        '''
        x.shape = (batch_size, 370, n_components).
        Each input into the LSTM has n_components features
        with a sequence length of 370.
        '''
        pass
    