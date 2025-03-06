from data import xtr, xte, ytr, yte
import torch
from torch import optim, nn
from pathlib import Path

class MulticlassClassification(nn.Module):
    def __init__(self):
        super(MulticlassClassification, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
model = MulticlassClassification()
loss_f = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr = 0.001)

epochs = 10000
for epoch in range(epochs):
    model.train()
    optim.zero_grad()

    y_preds_w = model(xtr)
    loss = loss_f(y_preds_w, ytr)

    loss.backward()
    optim.step()
    if epoch % 2000 == 0:
        print(f"Epoch: {epoch}, loss:{loss:.3f}")

FOLDER_PATH = Path("Saved_Models")
PATH = FOLDER_PATH / "MulticlassesClassification_Model.pth"
torch.save(model.state_dict(), PATH)
