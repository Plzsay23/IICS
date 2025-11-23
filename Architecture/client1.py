import socket
import pickle
from tqdm import tqdm
import time
import torch
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import struct
from collections import OrderedDict
import warnings
import select
import os
from torchvision import models
import torchvision.transforms.v2 as v2

warnings.filterwarnings("ignore")


############################################## 수정 금지 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/client1.pt"
######################################################################################################


############################################## 수정 가능 #############################################
local_epochs = 5
lr = 0.01
batch_size = 32
host_ip = "127.0.0.1"
port = 8081

# 최적화된 데이터 증강
train_transform = v2.Compose([
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])


# ========== 최적화된 모델 ==========
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
            ])
        
        layers.extend([
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class OptimalMedNet(nn.Module):
    """4개 평가지표 최적화 모델 - 파라미터: ~0.7M"""
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.blocks = nn.Sequential(
            InvertedResidual(32, 24, 1, 1),
            InvertedResidual(24, 24, 1, 4),
            InvertedResidual(24, 32, 2, 4),
            InvertedResidual(32, 32, 1, 4),
            InvertedResidual(32, 32, 1, 4),
            InvertedResidual(32, 64, 2, 4),
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 96, 2, 4),
            InvertedResidual(96, 96, 1, 4),
            InvertedResidual(96, 160, 2, 4),
        )
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(160, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(320, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train(model, criterion, optimizer, train_loader):
    model.to(device)
    model.train()
    
    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0
        
        for (images, labels) in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{local_epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_corrects.double() / total
        
        print(f"Epoch [{epoch + 1}/{local_epochs}] => "
              f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy * 100:.2f}%")
    
    return model

##############################################################################################################################

class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        blob = torch.load(pt_path, map_location="cpu")
        self.items = blob["items"]
        self.is_train = is_train
        self.transform = transform
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx: int):
        rec = self.items[idx]
        x = rec["tensor"].float() / 255.0
        y = int(rec["label"])
        x = self.transform(x)
        return x, y


def main():
    train_dataset = CustomDataset(DATASET_NAME, is_train=True, transform=train_transform)
    num_workers = max(2, (os.cpu_count() or 8) - 2)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True
    )
    
    model = OptimalMedNet(num_classes=NUM_CLASSES).to(device)
    
    # Optimized optimizer
    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4,
        nesterov=True
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    ##############################################################################################################################
    
    ########################################################### 수정 금지 2 ##############################################################
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host_ip, port))
    
    while True:
        data_size = struct.unpack('>I', client.recv(4))[0]
        rec_payload = b""
        
        remaining_payload = data_size
        while remaining_payload != 0:
            rec_payload += client.recv(remaining_payload)
            remaining_payload = data_size - len(rec_payload)
        dict_weight = pickle.loads(rec_payload)
        weight = OrderedDict(dict_weight)
        print("\nReceived updated global model from server")
        
        model.load_state_dict(weight, strict=True)
        
        read_sockets, _, _ = select.select([client], [], [], 0)
        if read_sockets:
            print("Federated Learning finished")
            break
        
        model = train(model, criterion, optimizer, train_loader)
        
        model_data = pickle.dumps(dict(model.state_dict().items()))
        client.sendall(struct.pack('>I', len(model_data)))
        client.sendall(model_data)
        
        print("Sent updated local model to server.")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nThe model will be running on", device, "device")
    
    time.sleep(1)
    main()
######################################################################################################################
