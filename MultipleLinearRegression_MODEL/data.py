import torch

# Create the data
torch.manual_seed(100)
X = torch.rand(100, 3)
random_weights = torch.rand(3,)
random_bias = torch.rand(1,)
y = (torch.matmul(X, random_weights) + random_bias).unsqueeze(dim = 1)

# Split the data
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
