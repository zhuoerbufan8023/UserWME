# -*- ecoding: utf-8 -*-
# @ModuleName: S-VMamba
# @Function: 
# @Author: Wang Zhuo
# @Time: 2024-06-08 11:23
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from MedMamba import VSSM as medmamba

import openpyxl

Image.MAX_IMAGE_PIXELS = None

label_dic = {'AB': 0, 'A': 1, 'B': 2}
#keys=["satisfaction","functional","explanatory","aesthetic","reliability"]
keys='functional'

# Define dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, idx):
        map1_path = self.data.iloc[idx]['map_A']
        map2_path = self.data.iloc[idx]['map_B']
        label = int(label_dic[self.data.iloc[idx][keys]])
        map1 = Image.open('./data/map/' + map1_path).convert('RGB')
        map2 = Image.open('./data/map/' + map2_path).convert('RGB')
        if self.transform:
            image1 = self.transform(map1)
            image2 = self.transform(map2)
        return map1, map2, label

    def __len__(self):
        return len(self.data)


# Define Map_S_VMamba model
class Map_S_VMamba(nn.Module):
    def __init__(self):
        super(Map_S_VMamba, self).__init__()
        # Define your pre-trained model and modifications
        self.cnn = medmamba(num_classes=3)
        # print(self.cnn)# Example pre-trained model
        self.cnn.head = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加 Dropout
            nn.Linear(1024, 3)  # 假定有3个类别
        )



    def forward(self, img1, img2):
        x1 = self.cnn(img1)
        x2 = self.cnn(img2)
        x = torch.cat((x1, x2), dim=1)
        return self.classifier(x)


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
    ws.append(["Epoch", "Phase", "Loss", "Accuracy"])

    best_acc = 0.0
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs1, inputs2, labels in tqdm(dataloaders[phase]):
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs1, inputs2)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs1.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            ws.append([epoch + 1, phase, epoch_loss, epoch_acc])

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        scheduler.step()

    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, './model/%s_best_model_S_VMamba.pth'%keys)

    wb.save('./log/%s_training_log_S_VMamba.xlsx'%keys)


# Main function
def main():
    # Setup datasets and dataloaders
    train_dataset = MyDataset('./data/S_VMamba_train.csv', transform=transform)
    val_dataset = MyDataset('./data/S_VMamba_test.csv', transform=transform_val)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    }

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # Initialize model, criterion, optimizer, and scheduler
    model = Map_S_VMamba().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    num_epochs = 150
    train_and_evaluate(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device)


if __name__ == '__main__':
    main()
