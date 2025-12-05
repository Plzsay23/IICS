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
from collections import OrderedDict, defaultdict
import warnings
import select
import os
from torchvision import models
import torchvision.transforms.v2 as v2
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
import random

warnings.filterwarnings("ignore")


############################################## 수정 금지 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/client1.pt"
######################################################################################################


############################################# 수정 가능 #############################################
local_epochs = 1       # 20 → 5 정도로 줄이고, 대신 global_round를 늘리는 쪽이 FL에 더 맞음
lr = 0.003             # 0.0001 → 0.001 (Adam 기준)
batch_size = 32          # 서버와 맞추기
host_ip = "127.0.0.1"
port = 8081

WIDTH_MULT = 0.2

################# 전처리 코드 수정 가능하나 꼭 IMG_SIZE로 resize한 뒤 정규화 해야 함 #################
common_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# 2. 증강 전처리 (복제된 데이터에만 적용: 회전 + 공통 전처리)
aug_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.RandomRotation(degrees=20),  # 회전 적용
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
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
        self.dropout = nn.Dropout(0.1)
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

                # [핵심] Autocast: 연산을 float16으로 수행하여 속도 향상
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # [핵심] Scaler: 손실 스케일링 후 역전파
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 정확도 및 손실 계산 (로깅용)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_counts = torch.sum(preds == labels).item()
                running_corrects += correct_counts
                total += labels.size(0)

                # Tqdm 바에 실시간 Loss/Acc 표시 (선택 사항)
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
        # 1. 데이터 로드
        blob = torch.load(pt_path, map_location="cpu")
        raw_items = blob["items"]
        self.is_train = is_train
        
        # transform 인자는 main에서 넘어오지만, 내부에서 aug_transform/common_transform을 분기 처리하므로
        # 여기서는 self.transform을 직접적으로 쓰지 않고 내부 로직을 따릅니다.
        
        self.final_items = []

        if is_train:
            # [학습 모드] 클래스 불균형 해소를 위한 오버샘플링 로직
            
            # (1) 클래스별 인덱스 분류
            class_indices = defaultdict(list)
            for idx, item in enumerate(raw_items):
                label = int(item["label"])
                class_indices[label].append(idx)
            
            # (2) 최대 데이터 개수 찾기 (목표 개수)
            # Client 2의 경우 Class 1이 0개일 수 있으므로, 존재하는 클래스 중에서만 Max 계산
            counts = [len(idxs) for idxs in class_indices.values() if len(idxs) > 0]
            max_count = max(counts) if counts else 0
            
            print(f"[{pt_path}] Max Class Count: {max_count}")

            # (3) 데이터 채우기
            for label, indices in class_indices.items():
                if len(indices) == 0:
                    continue # 데이터가 아예 없는 클래스(Client2의 Class 1)는 건너뜀
                
                # A. 원본 데이터 추가 (증강 False)
                for idx in indices:
                    self.final_items.append({
                        "data": raw_items[idx],
                        "is_aug": False
                    })
                
                # B. 부족한 만큼 무작위 복제하여 추가 (증강 True)
                num_needed = max_count - len(indices)
                if num_needed > 0:
                    print(f"  └ Class {label}: Add {num_needed} samples (Augmentation)")
                    aug_indices = random.choices(indices, k=num_needed) # 복원 추출
                    for idx in aug_indices:
                        self.final_items.append({
                            "data": raw_items[idx],
                            "is_aug": True # 이 플래그가 True면 회전을 먹임
                        })
        else:
            # [테스트 모드] 그냥 원본 그대로 사용
            for item in raw_items:
                self.final_items.append({
                    "data": item,
                    "is_aug": False
                })

    def __len__(self):
        return len(self.final_items)

    def __getitem__(self, idx: int):
        item_info = self.final_items[idx]
        rec = item_info["data"]
        is_aug = item_info["is_aug"]
        
        # 이미지 텐서 변환 (uint8 -> float division은 transform 내부 or 여기서 처리)
        # torchvision v2 transform은 uint8 입력도 받으므로 바로 넘겨도 되지만,
        # 기존 코드 호환성을 위해 float 변환 후 넘깁니다.
        x = rec["tensor"].float() / 255.0 
        y = int(rec["label"])

        # [핵심] 오버샘플링된 데이터면 회전 적용, 아니면 일반 전처리
        if self.is_train and is_aug:
            x = aug_transform(x)
        else:
            x = common_transform(x)

        return x, y

def main():

    train_dataset = CustomDataset(DATASET_NAME, is_train=True, transform=None)

    train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,          # Train은 꼭 Shuffle True!
    num_workers=2,         # [수정] 0 -> 2 (OS가 윈도우면 에러날 수 있으니 에러나면 0으로)
    pin_memory=True,       # [수정] False -> True
    persistent_workers=True # [수정] 에포크마다 워커 재생성 안함 (PyTorch 버전에 따라 지원)
)

    model = Network1().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)


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










