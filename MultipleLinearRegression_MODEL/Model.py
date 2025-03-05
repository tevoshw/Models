from data import X_train, X_test, y_train, y_test
import torch
from torch import nn
from torch import optim
from pathlib import Path


# Create the model
class MultipleLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# Call the functions]
torch.manual_seed(100)
model = MultipleLinearRegression()
loss_fn = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

# Show started parameters
"""
#print(f"Start weights: {model.linear.weight}")
#print(f"Start bias: {model.linear.bias}")
#print()

# Predicitions before trainig
 
model.eval()
with torch.inference_mode():
    y_preds_before = model(X_train)
    loss = loss_fn(y_preds_before, y_train)
    print(f"The loss before training it's: {loss:.4f}") """


# Training
epochs = 100000
for epoch in range(epochs):
    #Prepare to train
    model.train()
    optimizer.zero_grad()

    y_preds_while = model(X_train)
    loss = loss_fn(y_preds_while, y_train)

    # Change the parameters
    loss.backward()
    optimizer.step()

# Show know parameters

""""
print()
print(f"Knwow weights: {model.linear.weight}")
print(f"Know bias: {model.linear.bias}") """


# Predictions before training
""""
model.eval()
with torch.inference_mode():
    y_preds_after = model(X_test)
    loss = loss_fn(y_preds_after, y_test)
    print(f"The loss after training it's: {loss:.4f}") """


# Save the model in a new dir
FOLDER_PATH = Path("Saved_Models")
FOLDER_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = FOLDER_PATH / "MultipleLinearRegression_Model.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)


