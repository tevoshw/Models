import torch
from torch import nn, optim
from pathlib import Path
from data import Xtrain, ytrain, Xtest, ytest


# Model
class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

model = BinaryClassificationModel()

# Optimizer and loss function
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr= 0.001)


# Training
epochs = 20000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred_while = model(Xtrain)
    loss = loss_fn(y_pred_while, ytrain)

    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        print(f"Epoch: {epoch}, loss: {loss:.3f}")

print("\n\n\n")

# Testing with xtest
model.eval()
with torch.inference_mode():
    y_pred_after = model(Xtest)
    loss = loss_fn(y_pred_after, ytest)
    print(f"A loss do treino foi de: {loss:.3f}")


# Saving path
FOLDER_PATH = Path("Saved_Models")
SAVE_PATH = FOLDER_PATH / "BinaryClassification_Model.pth"
torch.save(model.state_dict(), SAVE_PATH)

