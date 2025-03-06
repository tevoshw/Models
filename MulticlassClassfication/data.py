from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
iris = load_iris()

X = iris.data
y = iris.target
# 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)

xtr, xte, ytr, yte = train_test_split(X, y, test_size = 0.2, random_state= 42)
xtr = torch.tensor(xtr, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.long)
