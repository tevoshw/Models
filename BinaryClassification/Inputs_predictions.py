import torch
from torch import nn
from pathlib import Path

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

loaded_model = BinaryClassificationModel()

# Load the model
FOLDER_PATH = Path("Saved_Models")
SAVE_PATH  = FOLDER_PATH / "BinaryClassification_Model.pth"
loaded_model.load_state_dict(torch.load(SAVE_PATH))


# User inputs
glicose = float(input("Digite o nível de glicose: "))
imc = float(input("Digite o IMC: "))
pressao_arterial = float(input("Digite a pressão arterial: "))

# Normalizing the user inputs
mean = torch.tensor([120, 25, 80])  
std = torch.tensor([30, 5, 10]) 
user_inputs = torch.tensor([glicose, imc, pressao_arterial], dtype= torch.float32)
user_inputs_normalized = (user_inputs - mean) / std

# User inputs predictions
loaded_model.eval()
with torch.inference_mode():
    y_preds_input = loaded_model(user_inputs_normalized)

# Printing user inputs predicitions
preds = 0.5
if y_preds_input > preds:
    print(f"A pessoa possui diabetes, {y_preds_input.item():.3f}")
elif y_preds_input < preds:
    print(f"A pessoa não possui diabetes, {y_preds_input.item():.3f}")
else:
    print(f"Sistema está inpreciso, {y_preds_input.item():.3f}")
