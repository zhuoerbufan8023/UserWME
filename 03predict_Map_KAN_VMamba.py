import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from MedMamba import VSSM as medmamba
from efficient_kan import KAN

Image.MAX_IMAGE_PIXELS = None

# Load the trained model
def load_model(model_path):
    model = KAN_VMamba()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define KAN_VMamba model
class KAN_VMamba(nn.Module):
    def __init__(self):
        super(KAN_VMamba, self).__init__()
        self.cnn = medmamba(num_classes=3)
        self.cnn.head = nn.Identity()
        self.regressor = KAN([768, 64, 1])

    def forward(self, map):
        x = self.cnn(map)
        return self.regressor(x).squeeze(1)

# Define the transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 预测函数
def predict_data(model, csv_file, output_csv_file):
    dataset = MyDataset(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    original_data = pd.read_csv(csv_file)
    original_data['predicted_result'] = predictions
    original_data.to_csv(output_csv_file, index=False)
key='reliability'
keys = '%s_normalized'%key
# 定义数据集类
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
        return map,label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    model_path = './model/%s_KAN_VMamba.pth'%key
    model = load_model(model_path)
    input_csv = './data/KAN_VMamba_test.csv'
    output_csv = './data/%s_predictions.csv'%key
    predict_data(model, input_csv, output_csv)