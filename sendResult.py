import numpy as np
import os
import random
import time
import requests
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.join(script_directory, "..", "dfb")
sys.path.append(parent_directory)

# 모델 파일
from model.wdcnn3 import *

# 훈련을 수행할 모듈
from trainmodule import *

server_ip = '172.16.63.190'

# 기존 curl 명령어에서 사용된 헤더와 인자값들을 설정
headers = {
        'Content-Type': 'text/plain',
}
params = {
        'k': 'testkey2023',
        'i': 'acoustic01',
}

# 데이터 값 로드. 편의상 판별할 데이터가 장치에 저장되어 있다고 가정.
test_data_level3 = {"A":[0,0],"B":[0,0],"C":[0,0]}

# 데이터 값 (252, 4096) - 4096 개의 데이터 252개
test_data_level3["C"][0] = np.load("dataset/X_continual_test_data_C.npy")

# 라벨 값 - 252개
test_data_level3["C"][1] = np.load("dataset/y_continual_test_data_C.npy")

# 1. 훈련된 모델의 체크포인트 파일 경로
model_path = "trained_model/model-v2.ckpt"

# 2. 모델, 옵티마이저, 손실 함수 초기화
model = WDCNN3(n_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()

# 3. PlModule 초기화
training_module = PlModule(model, optimizer, loss)

# 4. 클래스 메서드를 사용하여 체크포인트에서 모델 로드
model = PlModule.load_from_checkpoint(
    model_path,
    model=model,
    optimizer=optimizer,
    loss_fn=loss,
    map_location=torch.device('cpu'))
    
num_classes = 3 
training_module.model.eval()
class_confidence_scores = [0] * num_classes

input_data = test_data_level3["C"][0]
# 모델 입력으로 사용하기 위해 (252, 1, 4096)으로 데이터 형태 변환
input = torch.FloatTensor(input_data).unsqueeze(1)

# 이 데이터의 절반씩만 사용하여 2048길이의 데이터를 모델에 입력
sliced_input = input[:, :, :2048]


for j in range(250):
    with torch.no_grad():
        outputs = model(sliced_input[j:j+1])
        # 클래스의 확률
        probabilities = torch.softmax(outputs, dim=1)
        # 가장 높은 확률을 가진 클래스와 그 확률 값
        confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
    sending_data = ''
    # c1,2,3: max,min,rms / c4,5,6: 각 클래스 확률값 / c7,8: 판별 클래스, 판별한 클래스 확률값
    sending_data += 'c1'+'|'+str(np.max(test_data_level3["C"][0][j]))+'|c2|'+str(np.min(test_data_level3["C"][0][j]))+'|c3|'+\
        str(np.sqrt(np.mean(np.square(test_data_level3["C"][0][j]))))+'|c4|'+str(probabilities[0][0].item())+\
            '|c5|'+str(probabilities[0][1].item())+'|c6|'+str(probabilities[0][2].item())+\
            '|c7|'+str(predicted_classes.item())+'|c8|'+str(confidence_scores.item())
    print(f'[{0}] [{j}] {sending_data}')
    response = requests.post(f'http://{server_ip}:7896/iot/d', params=params, headers=headers, data=sending_data)