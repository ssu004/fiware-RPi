# fiware-RPi
fiware 실습할 때 Rpi에 저장될 파일

<폴더>

- dataset: 장치에서 수집된 데이터셋으로 가정. 미리 저장되어 있는 상태
- model: 추론에 사용할 인공지능 모델 - 경랑 1D CNN을 사용
- trained_model: 이전에 모델 훈련한 결과 값을 파일(checkpoint)로 저장


<파일>

- sendResult.py: 모델에 훈련 결과값을 로드하여 해당 모델로 수집한 데이터를 추론하고 데이터의 특성값과 추론 결과를 서버로 전송

- trainmodule.py: pytorchLighting 사용하여 훈련에 필요한 값 초기화 및 훈련을 간소화시킨 모듈


