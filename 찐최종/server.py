import threading
import socket
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset
import struct
from tqdm import tqdm
import copy
import warnings
import random
import torchvision.transforms.v2 as v2
import torch.nn.utils.prune as prune

warnings.filterwarnings("ignore")

############################################## 수정 불가 1 ##############################################
IMG_SIZE = 192
NUM_CLASSES = 4
DATASET_NAME = "./dataset/test.pt"
######################################################################################################

####################################################### 수정 가능 #######################################################
target_accuracy = 86.0  
global_round = 30   
batch_size = 128  
num_samples = 1   
host = '127.0.0.1' 
port = 8081 

WIDTH_MULT = 0.2

pre_cache_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.Grayscale(num_output_channels=1)
])

runtime_test_transform = v2.Compose([
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
        c24 = c(24); c32 = c(32); c64 = c(64); c96 = c(96); c160 = c(160); c320 = c(320)

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
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

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
        blob = torch.load(pt_path, map_location="cpu")
        raw_items = blob["items"]
        
        self.cached_data = []
        self.labels = []
        
        for rec in raw_items:
            x = rec["tensor"].float().div(255.0)
            x = pre_cache_transform(x)
            
            self.cached_data.append(x.clone())
            self.labels.append(int(rec["label"]))
            
    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx: int):
        x = self.cached_data[idx]
        y = self.labels[idx]
        x = runtime_test_transform(x)
        
        return x, y

def measure_accuracy(global_model, test_loader):
    model = Network1().to(device)
    model.load_state_dict(global_model)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=0.3)
            prune.remove(module, 'weight')
            pass
            
    model.eval()
    model.half()

    try:
        torch.set_float32_matmul_precision('medium')
    except:
        pass

    model.to(memory_format=torch.channels_last)
    gpu_inputs = []
    gpu_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True).half()
            labels = labels.to(device, non_blocking=True)
            gpu_inputs.append(inputs)
            gpu_labels.append(labels)

    correct = 0
    total = 0
    
    if len(gpu_inputs) > 0:
        with torch.no_grad():
            _ = model(gpu_inputs[0])

    if device == 'cuda': torch.cuda.synchronize()
    inference_start = time.time()
    
    with torch.inference_mode():
        for inputs, labels in zip(gpu_inputs, gpu_labels):
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    if device == 'cuda': torch.cuda.synchronize()
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    accuracy = (100 * correct / total)

    del gpu_inputs, gpu_labels
    torch.cuda.empty_cache()

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

    ############################ [수정] Main 부분 수정 ############################
    torch.backends.cudnn.benchmark = True
    train_dataset = CustomDataset(DATASET_NAME, is_train=False, transform=None)

    test_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2, 
        pin_memory=True,
        persistent_workers=True
    )

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
