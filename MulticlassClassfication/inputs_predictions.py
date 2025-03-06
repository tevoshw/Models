from torch import nn
import torch
from pathlib import Path
from data import X

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

loaded_model = MulticlassClassification()

FOLDER = Path("Saved_Models")
PATH = FOLDER / "MulticlassesClassification_Model.pth"
loaded_model.load_state_dict(torch.load(PATH))


#'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)
lenght = float(input("Digite o comprimento da sépala (cm): "))
imc = float(input("Digite a largura da sépala (cm): "))
freq_exercicio = float(input("Dite o comprimento da pétala (cm): "))
freq_cardiaca = float(input("Digite a largura da pétala (cm): "))


user_inputs = torch.tensor([[lenght, imc, freq_exercicio, freq_cardiaca,]], dtype=torch.float)


with torch.inference_mode():
    loaded_model.eval()
    y_preds = loaded_model(user_inputs)
    predicted_class = torch.argmax(y_preds, dim=1)

class_mapping = {0: "Setosa", 1: "Versicolor", 2: "Viriginica"}
predicted_class_name = class_mapping[predicted_class.item()]
print(f"Classe prevista: {predicted_class_name}")