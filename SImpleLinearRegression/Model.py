from data import Xtrain, Xtest, ytrain, ytest
import torch
from torch import nn, optim
from pathlib import Path


class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

torch.manual_seed(12)
model = SimpleLinearRegression()
loss_fn = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr= 0.01)


epochs = 10000
for epoch in range(epochs + 1):

    optimizer.zero_grad()
    model.train()

    y_preds_while = model(Xtrain)
    loss_while = loss_fn(y_preds_while, ytrain)

    loss_while.backward()
    optimizer.step()


FOLDER_PATH = Path("Saved_Models")
FOLDER_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = FOLDER_PATH / "SimpleLinearRegression_Model.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)

