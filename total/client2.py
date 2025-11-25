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


warnings.filterwarnings("ignore")


############################################## 수정 금지 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/client2.pt"
######################################################################################################


############################################# 수정 가능 #############################################
local_epochs = 2         # 20 → 5 정도로 줄이고, 대신 global_round를 늘리는 쪽이 FL에 더 맞음
lr = 0.001               # 0.0001 → 0.001 (Adam 기준)
batch_size = 32          # 서버와 맞추기
host_ip = "127.0.0.1"
port = 8081


################# 전처리 코드 수정 가능하나 꼭 IMG_SIZE로 resize한 뒤 정규화 해야 함 #################
train_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),       # 먼저 192x192로 맞추고
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=7),
    v2.ColorJitter(brightness=0.2, contrast=0.2,
                   saturation=0.2, hue=0.02),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])




# Network도 사용자 편의에 맞게 조정 (client와 server의 network와 같아야 함)
class Network1(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = models.mobilenet_v2(weights=None, width_mult=0.5)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def quantize_model(model, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        data = param.data
        max_val = data.abs().max()
        if max_val == 0:
            continue
        scale = max_val / qmax
        q = torch.round(data / scale).clamp(qmin, qmax)
        param.data = q * scale


def train(model, criterion, optimizer, train_loader):
    model.to(device)

    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{local_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = running_corrects.double() / total * 100.0
        print(f"Epoch [{epoch + 1}/{local_epochs}] => Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.2f}%")

    quantize_model(model, num_bits=8)
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                              shuffle=False, pin_memory=False)

    model = Network1().to(device)

    participation_weight = 1.3  # 클라2는 더 강하게 반영
    for param in model.parameters():
        param.data *= participation_weight

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

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










