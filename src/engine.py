class Predictor():
    def __init__(self):
        pass
    
    def get_avg_loss_for_epoch(self, propagator, dataloader):
        running_loss = 0
        for idx, (features, labels) in enumerate(dataloader):
            loss = propagator(features, labels)
            running_loss += loss
        
        return running_loss / len(dataloader.dataset)
    
    
class Trainer(Predictor):
    def __init__(self, model, trainloader, hyperparameters):
        super().__init__()
        self.model = model
        self.trainloader = trainloader
        
        # hyperparameters
        self.loss_fn = hyperparameters.loss_fn
        self.optimizer = hyperparameters.optimizer
        
    def train_epoch(self):
        
        def _train_step(self, x, y):
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
        
        return self.get_avg_loss_for_epoch(_train_step, self.trainloader)
    
    
class Validator(Predictor):
    def __init__(self, model, validateloader, hyperparameters):
        super().__init__()
        self.model = model
        self.validateloader = validateloader
        self.loss_fn = hyperparameters.loss_fn
        
    def validate_epoch(self):
        
        def _validate_step(self, x, y):
            '''
            Tests a batch input
            Returns an inference and losses
            '''
            self.model.eval()
            predicted = self.model.forward(x)
            loss = self.loss_fn(predicted, y)
            return loss.item()
        
        return self.get_avg_loss_for_epoch(_validate_step, self.validateloader)


class Runner(Trainer, Validator):
    def __init__(self, model, trainloader, validateloader, hyperparameters):
        super().__init__()
        self.model = model
        self.trainer = Trainer(model, trainloader, hyperparameters)
        self.validator = Validator(model, validateloader, hyperparameters)
        self.num_epochs = hyperparameters.num_epochs
        
    def run(self):
        train_losses, validation_losses = [], [] # can plot
        for epoch in range(self.num_epochs):
            train_losses.append(self.trainer.train_epoch())
            validation_losses.append(self.validator.validate_epoch())
            
        return train_losses, validation_losses
    
    