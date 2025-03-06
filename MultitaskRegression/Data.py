import torch

# Definindo a semente para reprodutibilidade
torch.manual_seed(42)

temperatura = torch.randint(0, 40, (100, 1))  
umidade = torch.rand(100, 1) * 100  
vento = torch.randint(0, 30, (100, 1)) 
X = torch.cat((temperatura.float(), umidade.float(), vento.float()), dim=1)

X_min = X.min(dim=0, keepdim=True)[0]  
X_max = X.max(dim=0, keepdim=True)[0]  
X_normalized = (X - X_min) / (X_max - X_min)  

W = torch.rand(3, 2)  
b = torch.rand(1, 2)  

y = torch.matmul(X_normalized, W) + b  

split = int(0.8 * len(X_normalized))
X_train, X_test = X_normalized[:split], X_normalized[split:]
y_train, y_test = y[:split], y[split:]

