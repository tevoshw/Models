import torch

torch.manual_seed(12)
X = torch.rand(100, 1)
random_weight = torch.rand(1, )
random_bias = torch.rand(1, )
y = (torch.matmul(X, random_weight) + random_bias).unsqueeze(dim = 1)

split = int(0.8 * len(X))
Xtrain, Xtest = X[:split], X[split:]
ytrain, ytest =  y[:split], y[split:]

