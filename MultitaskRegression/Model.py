from Data import X_train, X_test, y_test, y_train
import torch
from torch import nn, optim
from pathlib import Path


# Create the model
class MultitaskRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    

model = MultitaskRegression()
loss_function = nn.MSELoss()
optim = optim.SGD(model.parameters(), lr = 0.01)

epochs = 100000
for epoch in range(epochs):
    model.train()
    optim.zero_grad()

    y_preds_while = model(X_train)
    loss = loss_function(y_preds_while, y_train)

    loss.backward()
    optim.step()

model.eval()
with torch.inference_mode():
    y_preds_before = model(X_test)
    loss = loss_function(y_preds_before, y_test)
    print(f"A loss function foi de: {loss:.4f}")


FOLDER_PATH = Path("Saved_Models")
SAVE_PATH = FOLDER_PATH / "MultitaskRegression_Model.pth"
torch.save(model.state_dict(), SAVE_PATH)
