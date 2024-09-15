class Predictor():
    def __init__(self, model, dataloader, hyperparameters):
        self.model = model
        self.dataloader = dataloader
    
    def _step(self):
        raise NotImplementedError()
    
    def get_avg_loss_for_epoch(self):
        running_loss = 0
        for idx, (features, labels) in enumerate(self.dataloader):
            loss = self._step(features, labels)
            running_loss += loss
        
        return running_loss / len(self.dataloader.dataset)
    
    def run_epoch(self):
        return self.get_avg_loss_for_epoch()
    
    
class Trainer(Predictor):
    def __init__(self, model, trainloader, hyperparameters):
        super().__init__(model, trainloader, hyperparameters)
        self.model = model
        
        # hyperparameters
        self.loss_fn = hyperparameters.loss_fn
        self.optimizer = hyperparameters.optimizer


    def _step(self, x, y):
        '''
        Trains a batch input x.
        Returns a batch prediction and losses
        '''
        self.model.train()
        predicted = self.model.forward(x)
        loss = self.loss_fn(predicted, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    
class Validator(Predictor):
    def __init__(self, model, validateloader, hyperparameters):
        super().__init__(model, validateloader, hyperparameters)
        self.model = model
        self.loss_fn = hyperparameters.loss_fn
    
    def _step(self, x, y):
        '''
        Tests a batch input
        Returns an inference and losses
        '''
        self.model.eval()
        predicted = self.model.forward(x)
        loss = self.loss_fn(predicted, y)
        return loss.item()


class Runner():
    def __init__(self, model, trainloader, validateloader, hyperparameters):
        self.trainer = Trainer(model, trainloader, hyperparameters)
        self.validator = Validator(model, validateloader, hyperparameters)
        self.num_epochs = hyperparameters.num_epochs
        
    def run_all(self):
        train_losses, validation_losses = [], [] # can plot
        for epoch in range(self.num_epochs):
            train_losses.append(self.trainer.run_epoch())
            validation_losses.append(self.validator.run_epoch())
            
        return train_losses, validation_losses
    
    