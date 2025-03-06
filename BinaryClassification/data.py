import torch
import numpy as np
import matplotlib.pyplot as plt

# Create the random data
torch.manual_seed(42)
n_amostras = 1000

glicose = torch.normal(mean=120, std=30, size=(n_amostras,))  #
imc = torch.normal(mean=25, std=5, size=(n_amostras,))  
pressao = torch.normal(mean=80, std=10, size=(n_amostras,))  

probs = torch.sigmoid((glicose - 130) * 0.05 + (imc - 27) * 0.1 + (pressao - 85) * 0.02)
diabetes = (probs > 0.5).float()  

X = torch.stack([glicose, imc, pressao], dim=1)
y = diabetes.view(-1, 1)  

# Normalizing the data
mean = X.mean(dim=0)
std = X.std(dim=0)
X_norm = (X - mean) / std

# Split data in train and test
split = int(0.75 * len(X_norm))
Xtrain, Xtest = X_norm[:split], X_norm[split:]
ytrain, ytest = y[:split], y[split:]
