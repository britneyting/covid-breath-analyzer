import argparse

import numpy as np
import random
import torch
import pandas as pd
from preprocess import Preprocessor
from model import LSTMClassifier
from hyperparameters import Hyperparameters
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from engine import Runner


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str)
    parser.add_argument('--test_data_dir', type=str)
    parser.add_argument('--inferences_dir', type=str)
    args = parser.parse_args()
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    inferences_dir = args.inferences_dir
    

    # initiate model and hyperparameters and set seed
    hyperparameters = Hyperparameters(seed=47,
                                      n_components=16,
                                      lstm_hidden_size=8,
                                      split_ratio=0.1, 
                                      num_epochs=100, 
                                      batch_size=16, 
                                      loss_fn=BCEWithLogitsLoss(),
                                      optimizer=Adam,
                                      threshold=0.5)
    model = LSTMClassifier(hyperparameters.n_components, hyperparameters.lstm_hidden_size)
    hyperparameters.optimizer = hyperparameters.optimizer(model.parameters())

    np.random.seed(hyperparameters.seed)
    random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)
    torch.cuda.manual_seed(hyperparameters.seed)

    # preprocess data
    preprocessor = Preprocessor(train_data_dir)
    preprocessor.combine_data()
    preprocessor.normalize_data()
    preprocessor.reduce_dimensionality(hyperparameters.n_components)
    trainloader, validateloader = preprocessor.split_data(
        hyperparameters.split_ratio, 
        hyperparameters.seed, 
        hyperparameters.batch_size)
    
    # train and validate model
    runner = Runner(model, trainloader, validateloader, hyperparameters)
    train_losses, validate_losses = runner.run_all()
    
    # make inference on test data
    test_preprocessor = Preprocessor(test_data_dir)
    test_preprocessor.combine_data()
    test_preprocessor.normalize_data()
    test_preprocessor.reduce_dimensionality()
    testloader = preprocessor.processed_data
    inference_logits = model(testloader)
    
    # save inferences in csv tab delimited file
    inferences = pd.DataFrame(inference_logits > hyperparameters.threshold)
    inferences.to_csv(f"{inferences_dir}_inferences.csv")
    