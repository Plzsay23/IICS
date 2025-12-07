import threading
import socket
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

import struct
from tqdm import tqdm
import copy
import warnings
import random
import os
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
warnings.filterwarnings("ignore")

from torchvision import models
from torch.cuda.amp import autocast, GradScaler


############################################## 수정 불가 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/test.pt"
######################################################################################################

####################################################### 수정 가능 #######################################################
target_accuracy = 85.0  # 사용자 편의에 맞게 조정
global_round = 300      # 사용자 편의에 맞게 조정
batch_size = 32         # 사용자 편의에 맞게 조정
num_samples = 1280      # 사용자 편의에 맞게 조정
host = '127.0.0.1'      # loop back으로 연합학습 수행 시 사용될 ip
port = 8081             # 1024번 ~ 65535번

WIDTH_MULT = 0.2

# 테스트용 변환: 리사이즈 + 정규화(그대로 유지)
test_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BILINEAR),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# (추가) 베스트 모델 선택 임계값 및 상태 (최소 변경)
IMPROVE_DELTA_PP = 0.2   # (추가) 절대 향상폭 임계값(percentage points). 0.5 -> 0.2 로 완화
BEST_ACC = 0.0           # (추가) 현재까지 최고 정확도(%)
BEST_STATE = None        # (추가) 최고 성능 모델의 state_dict
####################################################### 수정 가능 끝 #######################################################


# Network도 사용자 편의에 맞게 조정 (client와 server의 network와 같아야 함)
# 기존 MobileNetV2 부분을 아래와 같이 수정
# client1.py, client2.py, server.py 모두 동일하게 수정

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




class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        print(pt_path)
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

def measure_accuracy(global_model, test_loader):  # pruning or quantization 적용시 필요한 경우 수정

    model = Network1().to(device)
    # model = Network2().to(device)

    model.load_state_dict(global_model)
    model.to(device)
    model.eval()

    accuracy = 0.0
    total = 0.0
    correct = 0


    inference_start = time.time()
    with torch.no_grad():
        print("\n")
        for inputs, labels in tqdm(test_loader, desc="Test"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)

            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        accuracy = (100 * correct / total)

    inference_end = time.time()
    inference_time = inference_end - inference_start

    # ===================== (추가) 베스트 모델 선택 로직 =====================
    global BEST_ACC, BEST_STATE  # (추가)
    picked = "KEEP"              # (추가)

    if (BEST_STATE is None) or (accuracy >= BEST_ACC + IMPROVE_DELTA_PP):  # (추가)
        BEST_ACC = accuracy                                                # (추가)
        BEST_STATE = copy.deepcopy(model.state_dict())                     # (추가)
        picked = "NEW"                                                     # (추가)
    else:
        model.load_state_dict(BEST_STATE)                                  # (추가)
        accuracy = BEST_ACC                                                # (추가)

    print(f"[Select] {picked} | best={BEST_ACC:.2f}% (delta_pp={IMPROVE_DELTA_PP:.2f})")  # (추가)
    # ===================== (추가) 끝 ========================================

    return accuracy, model, inference_time

##############################################################################################################################




####################################################### 수정 금지 ##############################################################
cnt = []
model_list = []  # 수신받은 model 저장할 리스트
semaphore = threading.Semaphore(0)

global_model = None
global_model_size = 0
global_accuracy = 0.0
current_round = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def handle_client(conn, addr, model, test_loader):
    global model_list, global_model, global_accuracy, global_model_size, current_round, cnt
    print(f"Connected by {addr}")

    while True:
        if len(cnt) < 2:
            cnt.append(1)
            weight = pickle.dumps(dict(model.state_dict().items()))
            # print(weight)
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)

        data_size = struct.unpack('>I', conn.recv(4))[0]
        received_payload = b""
        remaining_payload_size = data_size
        while remaining_payload_size != 0:
            received_payload += conn.recv(remaining_payload_size)
            remaining_payload_size = data_size - len(received_payload)
        model = pickle.loads(received_payload)

        model_list.append(model)
        # print(models)
        if len(model_list) == 2:
            current_round += 1
            global_model = average_models(model_list)
            global_accuracy, global_model, _ = measure_accuracy(global_model, test_loader)
            print(f"Global round [{current_round} / {global_round}] Accuracy : {global_accuracy}%")
            global_model_size = get_model_size(global_model)
            model_list = []
            semaphore.release()
        else:
            semaphore.acquire()

        if (current_round == global_round) or (global_accuracy >= target_accuracy):
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)
            conn.close()
            break
        else:
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)

def get_model_size(global_model):
    model_size = len(pickle.dumps(dict(global_model.state_dict().items())))
    model_size = model_size / (1024 ** 2)

    return model_size


def get_random_subset(dataset, num_samples):
    if num_samples > len(dataset):
        raise ValueError(f"num_samples should not exceed {len(dataset)} (total number of samples in test dataset).")

    indices = random.sample(range(len(dataset)), num_samples)
    subset = Subset(dataset, indices)

    return subset

def average_models(models):
    weight_avg = copy.deepcopy(models[0])

    for key in weight_avg.keys():
        for i in range(1, len(models)):
            weight_avg[key] += models[i][key]
        weight_avg[key] = torch.div(weight_avg[key], len(models))

    return weight_avg


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()
    connection = []
    address = []

    ############################ 수정 가능 ############################
    train_dataset = CustomDataset(DATASET_NAME, is_train=False, transform=test_transform)

    # (변경) 윈도우/TCP 안정성 우선: 워커 0, pin_memory False, persistent_workers False
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,            # (변경) 2 -> 0
        pin_memory=False,         # (변경) True -> False
        persistent_workers=False  # (변경) True -> False
    )

    # (추가) 미세 가속: cuDNN autotune + matmul 정밀도 설정 (GPU일 때 효과)
    try:
        torch.backends.cudnn.benchmark = True     # (추가)
        torch.set_float32_matmul_precision('high')# (추가; PyTorch 2.x 이상에서 유효)
    except Exception:
        pass

    model = Network1().to(device)
    ####################################################################

    print(f"Server is listening on {host}:{port}")

    while len(address) < 2 and len(connection) < 2:
        conn, addr = server.accept()
        connection.append(conn)
        address.append(addr)

    training_start = time.time()

    connection1 = threading.Thread(target=handle_client, args=(connection[0], address[0], model, test_loader))
    connection2 = threading.Thread(target=handle_client, args=(connection[1], address[1], model, test_loader))

    connection1.start();connection2.start()
    connection1.join();connection2.join()

    training_end = time.time()
    total_time = training_end - training_start

    # 평가지표 1
    print(f"\n학습 성능 : {global_accuracy} %")
    # 평가지표 2
    print(f"\n학습 소요 시간: {int(total_time // 3600)} 시간 {int((total_time % 3600) // 60)} 분 {(total_time % 60):.2f} 초")

    # 평가지표 3
    print(f"\n최종 모델 크기: {global_model_size:.4f} MB")

    final_model = dict(global_model.state_dict().items())
    _, _, inference_time = measure_accuracy(final_model, test_loader)
    # 평가지표 4
    print(f"\n예측 소요 시간 : {(inference_time):.2f} 초")

    print("연합학습 종료")


if __name__ == "__main__":

    main()
##############################################################################################################################
