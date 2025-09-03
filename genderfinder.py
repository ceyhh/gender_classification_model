import torch

from torch import nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
device = torch.device("cuda")
train_transformer = transforms.Compose([
                    transforms.Resize((64,64)),
                    transforms.TrivialAugmentWide(num_magnitude_bins=31),
                    transforms.ToTensor()
])
test_transformer = transforms.Compose([
                    transforms.Resize((64,64)),
                    transforms.ToTensor()
])

data_train_dataset = datasets.ImageFolder(root= "the dataset's path",
                                     transform = train_transformer,
                                    target_transform = None
                                     )
data_test_dataset = datasets.ImageFolder(root="the dataset's path",

                                    transform = test_transformer)


train_dataloader = DataLoader(dataset=data_train_dataset, batch_size=16, shuffle=True)

test_dataloader = DataLoader(dataset=data_test_dataset, batch_size=16)


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


model12 = Tinyvgg().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model12.parameters(), lr=0.001)
epochs = 3

def accuracy_fn(y_true, y_pred):
    preds = torch.argmax(y_pred, dim=1)  # en yüksek skor hangi sınıf
    correct = (preds == y_true).sum().item()
    acc = correct / len(y_true)
    return acc

for epoch in range(epochs):
    tqdmloop = tqdm(train_dataloader,leave=True)

    for X,y in tqdmloop:
        X, y = X.to(device), y.to(device)
        model12.train()

        y_pred = model12(X)

        loss = loss_function(y_pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        acc = accuracy_fn(y, y_pred)

        # tqdm bar bilgileri
        tqdmloop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        tqdmloop.set_postfix(loss=loss.item(), acc=f"{acc:.2f}")


with torch.inference_mode():
    image_path =("your files path)
    image = Image.open(image_path).convert('RGB')
    image = test_transformer(image).unsqueeze(0)

    output = model12(image.to(device))
    prediction = output.argmax(dim=1).item()
    print("predicition:",prediction)

    savingpath = Path("./model12_weights.pth")

    torch.save(model12.state_dict(),savingpath)
