import torch
from torch import nn
from pathlib import Path


class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
torch.manual_seed(12)
loaded_model = SimpleLinearRegression()

FOLDER_PATH = Path("Saved_Models")
MODEL_PATH = FOLDER_PATH / "SimpleLinearRegression_Model.pth"
loaded_model.load_state_dict(torch.load(MODEL_PATH))

quantity_of_sell = float(input("Enter the quantity of the object ou sell: "))
user_input = torch.tensor([quantity_of_sell], dtype= torch.float32).unsqueeze(dim = 1)


# Do predictions
with torch.inference_mode():
    loaded_model.eval()
    y_pred_new = loaded_model(user_input)
    print(f"The price in $ are: ${y_pred_new.item():.3f}")