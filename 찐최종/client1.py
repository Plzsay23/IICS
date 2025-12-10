import socket
import pickle
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import struct
from collections import OrderedDict, defaultdict
import warnings
import select
import torchvision.transforms.v2 as v2
from torch.cuda.amp import autocast, GradScaler
import random

warnings.filterwarnings("ignore")

############################################## 수정 금지 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/client1.pt"
######################################################################################################

############################################# 수정 가능 #############################################
local_epochs = 1
lr = 0.003
batch_size = 32
host_ip = "127.0.0.1"
port = 8081

WIDTH_MULT = 0.2

################# 전처리 코드 수정 가능하나 꼭 IMG_SIZE로 resize한 뒤 정규화 해야 함 #################
pre_cache_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.Grayscale(num_output_channels=1)
])

runtime_aug_transform = v2.Compose([
    v2.RandomRotation(degrees=20),
    v2.Normalize(mean=[0.5], std=[0.5]), 
])

runtime_common_transform = v2.Compose([
    v2.Normalize(mean=[0.5], std=[0.5]),
])

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
            nn.Conv2d(1, stem_out, 3, 2, 1, bias=False),
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
    model.to(device)
    model.train()
    scaler = GradScaler()

    for epoch in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    return model

##############################################################################################################################

####################################################### 수정 가능 ##############################################################
class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        print(f"Loading {pt_path}...")
        blob = torch.load(pt_path, map_location="cpu")
        raw_items = blob["items"]
        self.is_train = is_train
        
        self.cached_data = [] 
        self.labels = []
        self.is_aug_flags = []

        temp_items = []

        if is_train:
            class_indices = defaultdict(list)
            for idx, item in enumerate(raw_items):
                label = int(item["label"])
                class_indices[label].append(idx)
            
            counts = [len(idxs) for idxs in class_indices.values() if len(idxs) > 0]
            max_count = max(counts) if counts else 0
            
            print(f"[{pt_path}] Max Class Count: {max_count} (Start Caching...)")

            for label, indices in class_indices.items():
                if len(indices) == 0: continue
                
                for idx in indices:
                    temp_items.append({"data": raw_items[idx], "is_aug": False})
                
                num_needed = max_count - len(indices)
                if num_needed > 0:
                    aug_indices = random.choices(indices, k=num_needed)
                    for idx in aug_indices:
                        temp_items.append({"data": raw_items[idx], "is_aug": True})
        else:
            for item in raw_items:
                temp_items.append({"data": item, "is_aug": False})
        
        print(f"Preprocessing {len(temp_items)} images into RAM...")
        
        for item_info in tqdm(temp_items, desc="Caching"):
            rec = item_info["data"]
            is_aug = item_info["is_aug"]

            img_tensor = rec["tensor"].float().div(255.0)
            img_tensor = pre_cache_transform(img_tensor)
            
            self.cached_data.append(img_tensor.clone())
            self.labels.append(int(rec["label"]))
            self.is_aug_flags.append(is_aug)
            
        print("Caching Completed. Ready to train.")

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx: int):
        x = self.cached_data[idx]
        y = self.labels[idx]
        is_aug = self.is_aug_flags[idx]

        if self.is_train and is_aug:
            x = runtime_aug_transform(x)
        else:
            x = runtime_common_transform(x)

        return x, y

def main():
    torch.backends.cudnn.benchmark = True
    train_dataset = CustomDataset(DATASET_NAME, is_train=True, transform=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True)

    model = Network1().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=True)
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

