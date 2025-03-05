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

loaded_model = MultipleLinearRegression()


# Get the model trained
FOLDER_PATH = Path("Saved_Models")
MODEL_SAVE_PATH = FOLDER_PATH / "MultipleLinearRegression_Model.pth"
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Get the user inputs
area = float(input("Enter the size of the apartament (m2): "))
numbers_of_bedrooms = float(input("Enter the numbers of bedroomns: "))
age_of_apartment = float(input("Enter the age of apartament: "))
user_inputs = torch.tensor([[area, numbers_of_bedrooms, age_of_apartment]], dtype=torch.float32)

# Do new predictions
loaded_model.eval()
with torch.inference_mode():
    y_new_pred = loaded_model(user_inputs)
    print(f"The price in thousand dollars it's: ${y_new_pred.item():.3f}")

