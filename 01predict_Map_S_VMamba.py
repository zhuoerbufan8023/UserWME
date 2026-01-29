# @ModuleName: m1
# @Function:
# @Author: Wang Zhuo
# @Time: 2024-06-08 11:23
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from MedMamba import VSSM as medmamba
import csv
import torch.distributed as dist
import torch.multiprocessing as mp

Image.MAX_IMAGE_PIXELS = None

label_dic = {'AB': 0, 'A': 1, 'B': 2}


# Define dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, list_all, transform=None):
        self.data = list_all
        self.transform = transform

    def __getitem__(self, idx):
        map1_path, map2_path = self.data[idx]
        map1 = Image.open('./data/map/' + map1_path).convert('RGB')
        map2 = Image.open('./data/map/' + map2_path).convert('RGB')
        if self.transform:
            image1 = self.transform(map1)
            image2 = self.transform(map2)
        return map1, map2, map1_path, map2_path

    def __len__(self):
        return len(self.data)


# Define Map_S_VMamba model
class Map_S_VMamba(nn.Module):
    def __init__(self):
        super(Map_S_VMamba, self).__init__()
        # Define your pre-trained model and modifications
        self.cnn = medmamba(num_classes=3)
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
transform_val = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def compute(x):
    if x == 1:
        return 2
    elif x == 2:
        return 1
    elif x == 0:
        return 0
    else:
        print(type(x))


class Args:
    pass


# Main function for distributed prediction
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    print(f"Using GPU: {args.gpu} for prediction")

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(gpu)

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')

    # Set the device
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda', args.gpu)

    # Initialize the models
    model_satisfaction = Map_S_VMamba()
    model_satisfaction.load_state_dict(torch.load('./model/satisfaction_best_model_S_VMamba.pth', map_location=device))
    model_satisfaction.to(device)
    model_satisfaction = nn.parallel.DistributedDataParallel(model_satisfaction, device_ids=[args.gpu])

    model_functional = Map_S_VMamba()
    model_functional.load_state_dict(torch.load('./model/functional_best_model_S_VMamba.pth', map_location=device))
    model_functional.to(device)
    model_functional = nn.parallel.DistributedDataParallel(model_functional, device_ids=[args.gpu])

    model_explanatory = Map_S_VMamba()
    model_explanatory.load_state_dict(torch.load('./model/explanatory_best_model_S_VMamba.pth', map_location=device))
    model_explanatory.to(device)
    model_explanatory = nn.parallel.DistributedDataParallel(model_explanatory, device_ids=[args.gpu])

    model_aesthetic = Map_S_VMamba()
    model_aesthetic.load_state_dict(torch.load('./model/aesthetic_best_model_S_VMamba.pth', map_location=device))
    model_aesthetic.to(device)
    model_aesthetic = nn.parallel.DistributedDataParallel(model_aesthetic, device_ids=[args.gpu])

    model_reliability = Map_S_VMamba()
    model_reliability.load_state_dict(torch.load('./model/reliability_best_model_S_VMamba.pth', map_location=device))
    model_reliability.to(device)
    model_reliability = nn.parallel.DistributedDataParallel(model_reliability, device_ids=[args.gpu])

    dataset = MyDataset(args.list_all_map_c, transform=transform_val)
    sampler = DistributedSampler(dataset)
    dataloaders = DataLoader(dataset, batch_size=400, sampler=sampler, num_workers=8, pin_memory=True)

    with open("./data/file_out.csv", 'a', encoding='utf-8', newline="") as f:
        csv_writer = csv.writer(f)
        if args.gpu == 0:  # 只有主进程写入标题
            csv_writer.writerow(
                ["map1", "map2", "satisfaction", "functional", "explanatory", "aesthetic", "reliability"])

        with torch.no_grad():
            for batch in tqdm(dataloaders, desc=f"GPU {args.gpu} Predicting"):
                inputs1, inputs2, inputs1_name, inputs2_name = batch
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)

                outputs_satisfaction = model_satisfaction(inputs1, inputs2)
                _, pred_satisfaction = torch.max(outputs_satisfaction, 1)
                all_predictions_satisfaction = pred_satisfaction.cpu().numpy().flatten().tolist()

                outputs_functional = model_functional(inputs1, inputs2)
                _, pred_functional = torch.max(outputs_functional, 1)
                all_predictions_functional = pred_functional.cpu().numpy().flatten().tolist()

                outputs_explanatory = model_explanatory(inputs1, inputs2)
                _, pred_explanatory = torch.max(outputs_explanatory, 1)
                all_predictions_explanatory = pred_explanatory.cpu().numpy().flatten().tolist()

                outputs_aesthetic = model_aesthetic(inputs1, inputs2)
                _, pred_aesthetic = torch.max(outputs_aesthetic, 1)
                all_predictions_aesthetic = pred_aesthetic.cpu().numpy().flatten().tolist()

                outputs_reliability = model_reliability(inputs1, inputs2)
                _, pred_reliability = torch.max(outputs_reliability, 1)
                all_predictions_reliability = pred_reliability.cpu().numpy().flatten().tolist()

                for i in range(len(inputs1_name)):
                    csv_writer.writerow([inputs1_name[i], inputs2_name[i], all_predictions_satisfaction[i],
                                         all_predictions_functional[i], all_predictions_explanatory[i],
                                         all_predictions_aesthetic[i], all_predictions_reliability[i]])
                    csv_writer.writerow([inputs2_name[i], inputs1_name[i], compute(all_predictions_satisfaction[i]),
                                         compute(all_predictions_functional[i]),
                                         compute(all_predictions_explanatory[i]), compute(all_predictions_aesthetic[i]),
                                         compute(all_predictions_reliability[i])])

    f.close()


def main():
    args = Args()
    args.list_all_map_c = []
    map_all = os.listdir('./data/map')
    for i in range(0, len(map_all) - 1):
        for j in range(i + 1, len(map_all)):
            args.list_all_map_c.append([map_all[i], map_all[j]])

    # 设定使用的GPU为1和2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    ngpus_per_node = 4
    args.world_size = ngpus_per_node

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == '__main__':
    main()
