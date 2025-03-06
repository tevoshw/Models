import torch
from torch import nn
from pathlib import Path

class MultitaskRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    

loaded_model = MultitaskRegression()


FOLDER_PATH = Path("Saved_Models")
SAVE_PATH  = FOLDER_PATH / "MultitaskRegression_Model.pth"

loaded_model.load_state_dict(torch.load(SAVE_PATH)) 

age = float(input("Digite a temperatura: "))
imc = float(input("Digite a umidade: "))
exercises_per_week = float(input("Digite a velocidade do vento (km/h): "))
user_inputs = torch.tensor([age, imc, exercises_per_week], dtype= torch.float32)

loaded_model.eval()
with torch.inference_mode():
    y_preds_input = loaded_model(user_inputs)
    print(f"A prediçõa de precipitação é: {y_preds_input[0].item():.2f}%")
    print(f"A predição da velocidade do vento em rajada é de: {y_preds_input[1].item():.2f}%")