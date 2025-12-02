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
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torch.cuda.amp import autocast, GradScaler


warnings.filterwarnings("ignore")


############################################## 수정 금지 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/client1.pt"
######################################################################################################


############################################# 수정 가능 #############################################
#######
local_epochs = 3       # (변경) 1 -> 3
#######
lr = 0.0008             # 0.0001 → 0.001 (Adam 기준)
batch_size = 32          # 서버와 맞추기
host_ip = "127.0.0.1"
port = 8081

WIDTH_MULT = 0.35

################# 전처리 코드 수정 가능하나 꼭 IMG_SIZE로 resize한 뒤 정규화 해야 함 #################
train_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BILINEAR),
    v2.RandomHorizontalFlip(p=0.5),
    #######
    v2.RandomRotation(degrees=2),  # (완화)
    # v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),  # 임시 비활성
    # v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),  # 임시 비활성
    # v2.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # 임시 비활성
    #######
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])





# client1.py, client2.py, server.py 공통 수정
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
    def __init__(self, num_classes=4, width_mult=WIDTH_MULT):
        super().__init__()

        def c(v):
            return max(8, int(v * width_mult))

        stem_out = c(32)
        c24 = c(24)
        c32 = c(32)
        c64 = c(64)
        c96 = c(96)
        c160 = c(160)
        c320 = c(320)

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            InvertedResidual(stem_out, c24, 1, 1),
            InvertedResidual(c24, c24, 1, 4),
            InvertedResidual(c24, c32, 2, 4),
            InvertedResidual(c32, c32, 1, 4),
            InvertedResidual(c32, c32, 1, 4),
            InvertedResidual(c32, c64, 2, 4),
            InvertedResidual(c64, c64, 1, 4),
            InvertedResidual(c64, c64, 1, 4),
            InvertedResidual(c64, c96, 2, 4),
            InvertedResidual(c96, c96, 1, 4),
            InvertedResidual(c96, c160, 2, 4),
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(c160, c320, 1, bias=False),
            nn.BatchNorm2d(c320),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(c320, num_classes)

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


class Network1(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = OptimalMedNet(num_classes=num_classes, width_mult=WIDTH_MULT)
    
    def forward(self, x):
        return self.model(x)


    


def train(model, criterion, optimizer, train_loader):
    """
    AMP(Automatic Mixed Precision)를 적용하여
    학습 속도를 높이고 메모리를 절약하는 전체 학습 함수
    """
    model.to(device)
    model.train()
    scaler = GradScaler()

    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        # Tqdm으로 진행률 표시
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{local_epochs}", unit="batch") as pbar:
            for images, labels in pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Autocast
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # Scaler: 손실 스케일링 후 역전파
                #######
                # (변경) 안정성 위해 gradient clipping 추가
                #######
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # 정확도 및 손실 계산 (로깅용)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_counts = torch.sum(preds == labels).item()
                running_corrects += correct_counts
                total += labels.size(0)

                current_loss = loss.item()
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        # 에포크 종료 후 전체 통계 출력
        epoch_loss = running_loss / total
        epoch_accuracy = running_corrects / total * 100.0
        print(f"Epoch [{epoch + 1}/{local_epochs}] => Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.2f}%")

    return model


##############################################################################################################################



####################################################### 수정 가능 ##############################################################


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
        x = rec["tensor"].float() / 255.0      # uint8 [C,H,W]
        y = int(rec["label"])

        x = self.transform(x)

        return x, y

def main():

    train_dataset = CustomDataset(DATASET_NAME, is_train=True, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,          
    num_workers=2,         
    pin_memory=True,       
    persistent_workers=True
)

    model = Network1().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    #######
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)  # (완화) 0.05 -> 0.02
    #######


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
