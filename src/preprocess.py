import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CovidDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        num_patients, num_samples, num_sensors = self.X.shape
        return num_patients
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

class Preprocessor():
    '''
    Provides methods for preprocessing data for use in model
    training.
    '''
    
    def __init__(self, dir):
        self.dir = dir
        self.processed_data = []
        
        self.classification = {'POSITIVE': 1, 'NEGATIVE': 0}
        self.labels = []

    
    def combine_data(self):
        files = os.listdir(self.dir)
        for f in files:
            fname = os.path.join(self.dir, f)
            
            df = pd.read_csv(fname, sep='\t', header=2)
            self.processed_data.append(df.iloc[:372, 1:].to_numpy('float32'))
            
            label = self.classification[
                        pd.read_csv(fname, nrows=1)
                          .iloc[0,0]
                          .split(' ')[1]]
            self.labels.append(label)
        
        self.processed_data = np.array(self.processed_data)
        self.labels = np.array(self.labels).reshape(-1,1)

    def normalize_data(self):
        mean = self.processed_data.mean(axis=1)
        mean = np.expand_dims(mean, axis=1)
        
        std = self.processed_data.std(axis=1)
        std = np.expand_dims(std, axis=1)
        
        self.processed_data = (self.processed_data - mean) / std
    
    def reduce_dimensionality(self, n_components=16):
        pca = PCA(n_components)
        reduced_data = []
        for patient in range(len(self.processed_data)):
            reduced_data.append(pca.fit_transform(self.processed_data[patient]))
        self.processed_data = np.array(reduced_data)
    
    def split_data(self, ratio=0.1, seed=47, batch_size=16):
        X_train, X_validate, y_train, y_validate = train_test_split(
            self.processed_data, self.labels, 
            test_size=ratio,
            random_state=seed,
            shuffle=True,
            stratify=self.labels
            )
        
        train_data = CovidDataset(X_train, y_train)
        validate_data = CovidDataset(X_validate, y_validate)
        
        trainloader = DataLoader(train_data, batch_size, shuffle=True)
        validateloader = DataLoader(validate_data, batch_size, shuffle=True)
        
        return trainloader, validateloader
        
