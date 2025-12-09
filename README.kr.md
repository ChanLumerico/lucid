# Lucid² 💎

**Lucid**는 순수 파이썬으로 처음부터 구현한 미니멀리스트 딥러닝 프레임워크이다. 오토디프, 신경망 모듈, GPU 가속까지 현대 딥러닝 시스템의 토대를 가볍고 읽기 쉬운 코드로 탐구할 수 있는 교육 친화적인 환경을 제공한다.

학생, 교육자, 연구자가 딥러닝 내부 동작을 투명하게 살펴볼 수 있도록, PyTorch와 같은 주요 프레임워크의 핵심 동작을 충실히 재현하면서도 줄 단위로 따라 읽을 만큼 단순한 API를 제공한다.

[📑 Lucid 문서](https://chanlumerico.github.io/lucid/build/html/index.html) | 
[🤗 Lucid Huggingface](https://huggingface.co/ChanLumerico/lucid)

### 🔥 새로운 소식

- [**`Safetensors`**](https://github.com/huggingface/safetensors)를 지원하여 기존 `.lcd` 포맷과 함께 Lucid 신경 모듈 포팅에 활용할 수 있다.
- 신경 모듈 카테고리 `nn.rnn`을 추가했다: `nn.RNNBase`, `nn.RNN`, `nn.LSTM`, `nn.GRU`, `nn.RNNCell`, `nn.LSTMCell`, `nn.GRUCell`

## 🔧 설치 방법

Lucid는 가볍고 이식성이 높으며 어떤 환경에서도 쉽게 사용할 수 있도록 설계되었다.

### ▶️ 기본 설치
Lucid는 PyPI에서 바로 설치할 수 있다:
```bash
pip install lucid-dl
```

GitHub의 최신 개발 버전을 설치할 수도 있다:
```bash
pip install git+https://github.com/ChanLumerico/lucid.git
```
이렇게 하면 NumPy 기반 CPU 모드로 Lucid를 사용하는 데 필요한 핵심 구성 요소가 설치된다.

### ⚡ GPU 활성화 (Metal / MLX 가속)
Apple Silicon(M1, M2, M3) Mac에서는 MLX 라이브러리를 통해 GPU 실행을 지원한다.

Metal 가속 사용 절차:
1. MLX 설치:
```bash
pip install mlx
```
2. 호환되는 장치(Apple Silicon)인지 확인한다.
3. 연산 시 `device="gpu"`를 지정한다.

### ✅ 동작 확인
```python
import lucid
x = lucid.ones((1024, 1024), device="gpu")
print(x.device)  # 'gpu'가 출력되어야 한다.
```

## 📐 Tensor: 핵심 추상화

Lucid의 중심에는 `Tensor` 클래스가 있다. 이는 NumPy 배열을 일반화하여 기울기 추적, 디바이스 배치, 연산 그래프 구축과 같은 기능을 지원한다.

각 Tensor는 다음을 담는다:
- 데이터 배열(`ndarray` 또는 `mlx.array`)
- 기울기(`grad`) 버퍼
- 자신을 만든 연산
- 부모 텐서 목록
- 계산 그래프 포함 여부(`requires_grad`)

### 🔁 생성과 설정
```python
from lucid import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device="gpu")
```

- `requires_grad=True`는 텐서를 오토디프 그래프에 포함시킨다.
- `device="gpu"`는 텐서를 Metal 백엔드에 할당한다.

### 🔌 디바이스 전환
Tensor는 `.to()`로 CPU와 GPU를 오갈 수 있다:
```python
x = x.to("gpu")
y = x.to("cpu")
```

현재 텐서의 디바이스는 다음과 같이 확인한다:
```python
print(x.device)  # 'cpu' 또는 'gpu'
```

## 📉 자동미분 (Autodiff)

Lucid는 **역전파 방식 자동미분**을 구현한다. 이는 스칼라 손실의 기울기를 효율적으로 계산할 때 사용된다.

순전파 동안 `requires_grad`가 필요한 모든 Tensor 연산을 기록하며, 각 노드는 사용자 정의 backward 함수를 저장해 체인 룰로 기울기를 전달한다.

### 📘 계산 그래프 내부
계산 그래프는 DAG(유향 비순환 그래프)이다:
- 각 `Tensor`가 노드 역할을 한다.
- 모든 연산은 입력과 출력 사이에 간선을 만든다.
- 각 Tensor는 부모에 대한 기울기를 계산하는 `_backward_op` 메서드를 가진다.

`.backward()` 과정:
1. 그래프를 위상 정렬한다.
2. 출력 기울기를 초기화한다(대개 1.0).
3. 모든 backward 연산을 역순으로 실행한다.

### 🧠 예시
```python
import lucid

x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1
z = y.sum()
z.backward()
print(x.grad)  # [2.0, 2.0, 2.0]
```

이는 체인 룰을 적용해 $\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y}\cdot\frac{\partial y}{\partial x} = [2, 2, 2]$를 계산한 결과이다.

### 🔄 훅 & 형태 정렬
Lucid는 다음을 지원한다:
- 기울기 관찰·수정을 위한 **훅(Hook)**
- 텐서 형태 간 브로드캐스팅과 정렬

## 🚀 Metal 가속 (MLX 백엔드)

Lucid는 Apple Silicon에서 [MLX](https://github.com/ml-explore/mlx)를 사용해 **Metal 가속**을 지원한다. 텐서 연산, 신경망 레이어, 역전파 계산을 GPU에서 수행할 수 있다.

### 📋 핵심 특징
- `device="gpu"`인 Tensor는 `mlx.core.array`로 할당된다.
- 핵심 수학 연산, 행렬 곱, backward 연산이 MLX API를 사용한다.
- API 변경 없이 `.to("gpu")` 또는 텐서 생성 시 `device="gpu"`로 전환한다.

### 💡 예시 1: 기본 가속
```python
import lucid

x = lucid.randn(1024, 1024, device="gpu", requires_grad=True)
y = x @ x.T
z = y.sum()
z.backward()
print(x.grad.device)  # 'gpu'
```

### 💡 예시 2: GPU 기반 모델
```python
import lucid.nn as nn
import lucid.nn.functional as F

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return F.relu(self.fc(x))

model = TinyNet().to("gpu")
data = lucid.randn(32, 100, device="gpu", requires_grad=True)
output = model(data)
loss = output.sum()
loss.backward()
```

GPU에서 모델을 학습할 때는 **각 순전파 이후 손실 텐서를 명시적으로 평가**해야 한다. 그렇지 않으면 MLX 계산 그래프가 커져 성능 저하나 메모리 오류가 발생할 수 있다. MLX는 평가를 지연하므로 `.eval()` 등으로 강제 평가해야 한다.

### 권장 GPU 학습 패턴
```python
loss = model(input).sum()
loss.eval()  # GPU에서 강제 평가
loss.backward()
```
이렇게 하면 역전파 전에 이전 GPU 연산이 모두 평가되어 그래프 성장을 통제할 수 있다.

## 🧱 `lucid.nn`으로 신경망 구축

Lucid는 `nn.Module`을 통해 PyTorch 스타일의 모듈식 인터페이스를 제공한다. 사용자는 `nn.Module`을 상속하고 매개변수와 레이어를 속성으로 선언해 모델을 만든다.

모든 모듈은 매개변수를 자동 등록하며, 디바이스 이동(`.to()`)과 오토디프 시스템 통합을 지원한다.

### 🧰 사용자 정의 모듈
```python
import lucid.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
```

### 🧩 파라미터 등록
모든 파라미터는 자동으로 등록되며 다음과 같이 접근할 수 있다:
```python
model = MLP()
print(model.parameters())
```

### 🧭 GPU로 이동
```python
model = model.to("gpu")
```
내부 파라미터가 모두 GPU 메모리로 이동한다.

## 🏋️‍♂️ 학습과 평가

Lucid는 표준 학습 루프, 맞춤형 옵티마이저, 배치 단위 기울기 추적을 지원한다.

### ✅ 전체 학습 루프
```python
import lucid
from lucid.nn.functional import mse_loss

model = MLP().to("gpu")
optimizer = lucid.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    preds = model(x_train)
    loss = mse_loss(preds, y_train)
    loss.eval()  # 평가 강제

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### 🧪 기울기 없이 평가
```python
with lucid.no_grad():
    out = model(x_test)
```
기울기 추적을 막아 메모리 사용을 줄인다.

## 📦 사전학습 가중치 로드

Lucid는 `lucid.weights` 모듈을 통해 표준 사전학습 초기화를 제공한다.
```python
from lucid.models import lenet_5
from lucid.weights import LeNet_5_Weights

# 사전학습 가중치로 LeNet-5 로드
model = lenet_5(weights=LeNet_5_Weights.DEFAULT)
```

가중치를 사용하지 않고 초기화하려면 `weights=None`을 전달하면 된다.

## 🧬 교육 중심 설계

Lucid는 블랙박스가 아니다. 모든 클래스와 함수, 코드 한 줄까지 읽고 수정하기 쉽게 설계되었다.

- 역전파를 직접 실험하며 직관을 쌓을 수 있다.
- 내부 연산을 수정해 맞춤형 오토그라드를 시험할 수 있다.
- CPU와 GPU 동작 차이를 원하는 모델에서 직접 벤치마크할 수 있다.
- 레이어별, 형태별, 기울기별로 디버깅할 수 있다.

신경망을 처음부터 만들든, 기울기 흐름을 관찰하든, 새로운 아키텍처를 설계하든 Lucid는 **투명한 실험장**이다.

## 🧠 마무리
Lucid는 강력한 교육용 리소스이자 미니멀한 실험용 샌드박스이다. 텐서, 기울기, 모델의 내부를 노출하고 GPU 가속을 통합해 딥러닝이 실제로 어떻게 동작하는지 **보고, 만지고, 이해**하도록 돕는다.

## 📜 기타

**Dependencies**:

| Library | Purpose |
| ------- | ------- |
| `numpy` | CPU용 핵심 Tensor 연산 |
| `mlx` | GPU(Apple Silicon)용 핵심 Tensor 연산 |
| `pandas`, `openml` | 데이터셋 다운로드 및 로딩 |
| `matplotlib` | 시각화 |
| `networkx` | Tensor·모듈 그래프 시각화용 구성 |

**Inspired By**:

![](https://skillicons.dev/icons?i=pytorch)
![](https://skillicons.dev/icons?i=tensorflow)
![](https://skillicons.dev/icons?i=stackoverflow)
