# -*- ecoding: utf-8 -*-
# @ModuleName: huigui1
# @Function:
# @Author: Wang Zhuo
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import  StepLR
from MedMamba import VSSM as medmamba
import openpyxl
from efficient_kan import KAN

Image.MAX_IMAGE_PIXELS = None

#keys=["satisfaction","functional","explanatory","aesthetic","reliability"]
key='reliability'
keys = '%s_normalized'%key

# Define dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, idx):
        map_path = self.data.iloc[idx]['map']
        label = float(self.data.iloc[idx][keys])
        map = Image.open('./data/map/' + map_path).convert('RGB')
        if self.transform:
            map = self.transform(map)
        return map, label

    def __len__(self):
        return len(self.data)


# Define KAN_VMamba model
class KAN_VMamba(nn.Module):
    def __init__(self):
        super(KAN_VMamba, self).__init__()
        # Define  pre-trained model and modifications
        self.cnn = medmamba(num_classes=3)
        self.cnn.head = nn.Identity()
        self.load_pretrained_cnn()
        # 定义自己的分类器，假设合并后的特征长度是2560（EfficientNet-B7的特征维度）
        self.regressor = KAN([768, 64, 1])

    def load_pretrained_cnn(self):
        cnn_state_dict = torch.load('./model/%s_best_model_KAN_VMamba.pth'%key)
        model_dict = self.cnn.state_dict()
        cnn_state_dict = {k: v for k, v in cnn_state_dict.items() if k in model_dict}
        model_dict.update(cnn_state_dict)
        self.cnn.load_state_dict(model_dict)

    def forward(self, map):
        x = self.cnn(map)
        return self.regressor(x).squeeze(1)


# Define the transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the transform
transform_val = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Training and evaluation function
def train_and_evaluate(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    # Excel setup
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Training Log"
    ws.append(["Epoch", "Phase", "Loss"])

    best_model_wts = None
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels).mean()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')
            ws.append([epoch + 1, phase, epoch_loss])

            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                best_epoch = epoch + 1

        # Save the training log to an Excel file
        scheduler.step()



    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Save best model weights
    torch.save(model.state_dict(), './model/%s_KAN_VMamba.pth'%key)




# Main function
def main():
    # Setup datasets and dataloaders
    train_dataset = MyDataset('./data/KAN_VMamba_train.csv', transform=transform)
    val_dataset = MyDataset('./data/KAN_VMamba_test.csv', transform=transform_val)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    }

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # Initialize model, criterion, optimizer, and scheduler
    model = KAN_VMamba().to(device)
    # criterion = nn.SmoothL1Loss()
    criterion = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    num_epochs = 300
    train_and_evaluate(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device)


if __name__ == '__main__':
    main()

