import torch

from torch import nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image

test_transformer = transforms.Compose([
                    transforms.Resize((64,64)),
                    transforms.ToTensor()
])
class Tinyvgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.firstlayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.secondlayer = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2560,out_features=2),
        )

    def forward(self, x):
        return self.classifier(self.secondlayer(self.firstlayer(x)))


device = torch.device("cuda")

model12 = Tinyvgg()
model12 = model12.to(device)

model12.load_state_dict(torch.load("C:/pytorchprojesi/model12_weights.pth", map_location=device))

model12.eval()
with torch.inference_mode():
    image_path = "C:/Users/ceyhu/Downloads/fe.png"
    image = Image.open(image_path).convert('RGB')  # PIL Image
    image = test_transformer(image).unsqueeze(0).to(device)
    output = model12(image)
    prediction = torch.softmax(output,dim=1)
    print(prediction)

