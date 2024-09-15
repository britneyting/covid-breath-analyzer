class Hyperparameters():
    def __init__(self, seed,
                 n_components, lstm_hidden_size,
                 split_ratio, num_epochs, batch_size, 
                 loss_fn, optimizer,
                 threshold):
        
        self.seed = seed
        
        # desired components in reduced space
        self.n_components = n_components
        
        # number of hidden units in LSTM
        self.lstm_hidden_size = lstm_hidden_size
        
        # train-validation data split ratio
        self.split_ratio = split_ratio
        
        # number of epochs to train model
        self.num_epochs = num_epochs
        
        # batch size of train inputs
        self.batch_size = batch_size
        
        # loss function
        self.loss_fn = loss_fn
        
        # optimizer
        self.optimizer = optimizer
        
        # threshold for classifying logits as negative (0) or positive (1)
        # logit < threshold signifies negative prediction
        # logit > threshold signifies positive prediction
        self.threshold = threshold