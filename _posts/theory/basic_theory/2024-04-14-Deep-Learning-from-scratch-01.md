---
title : 밑바닥부터 시작하는 딥러닝 정리 1편 - 신경망 쌓기
date : 2024-04-15 23:44:00 +09:00
categories : [basic theory]
tags : [base, deep learning, perceptron, layer] #소문자만 가능
description: 밑바닥부터 기초 쌓기
toc: true
toc_sticky: true
toc_label: 목차
math: true
mermaid: true
image: /assets/img/post/basic_theory/1.webp
---

> 밑바닥부터 다시 개념 정리를 해봅시다..

## 1. 퍼셉트론(perceptron)
---

- 퍼셉트론은 다수의 신호를 입력을 받아 하나의 신호를 출력
- 전류가 전선을 타고 흐르는 전자를 내보내듯, 퍼셉트론 신호도 흐름을 만들고 정보를 앞으로 전달함
- 흐른다 : 1, 안 흐른다 : 0

![퍼셉트론](/assets/img/post/basic_theory/2.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

- x1, x2 : 입력 신호
- y : 출력 신호
- w1, w2 : 가중치
- 그림의 원을 **뉴런** 혹은 **노드** 라고 부름
- 입력 신호가 뉴런에 보내질 때는 각각 고유한 **가중치** 가 곱해짐. (w1x1, w2x2)
- 가중치가 클수록 해당 신호가 그만큼 중요한 것. 
- 뉴런에서 보내온 신호의 총합이 정해진 한계를 넘어설 때만 1을 출력
- 그 한계를 **임계값** 이라 표현.


🤔 **가중치와 편향의 차이점**

- 가중치(w) : 각 입력 신호가 결과에 주는 영향력(중요도)를 조절하는 매개변수
- 편향(b) : 뉴런이 얼마나 쉽게 활성화(결과로 1을 출력)하느냐를 조정하는 매개변수


## 2. 신경망
---

![신경망](/assets/img/post/basic_theory/3.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }=

- 신경망은 **활성화 함수** 를 이용해 신호를 전달한다.
- 활성화 함수란 입력 신호의 총합이 활성화를 일으키는지를 정하는 역할을 한다.

## 3. 활성화 함수(activation function)

![활성화 함수](/assets/img/post/basic_theory/4.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

### 3-1. 계단함수

계단함수는 0을 경계로 출력이 0에서 1로 바뀐다.

```python
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int32)

x = np.arange(-5.0, 5.0, 0.1) # -5.0에서 5.0전까지 0.1 간격의 넘파이 배열 생성
y = step_function(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
```

![계단 함수](/assets/img/post/basic_theory/5.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

### 3-2. 시그모이드 함수

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1) # y축 범위 지정 
plt.show()
```

![계단 함수](/assets/img/post/basic_theory/6.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

🤔 `np.exp()`

numpy.exp() 함수는 밑이 자연상수 e인 지수함수(e^x)로 변환해준다.

```python
print(np.exp(0))    # e^0 와 동일
# 1.0

print(np.exp(1))    # e^1 와 동일
# 2.718281828459045

print(np.exp(10))   # e^10 와 동일
# 22026.465794806718
```

**차이점** 
- 계단 함수 : 0과 1 중 하나의 값만 돌려준다
- 시그모이드 함수 : 실수(0.72, 0.88 등)를 돌려준다


**공통점**
- 두 함수 모두 입력이 중요하면 큰 값을 출력하고, 입력이 중요하지 않으면 작은 값을 출력한다
- 두 함수 모두 **비선형 함수** 이다

<span style="color:violet">즉, **퍼셉트론** 에서는 뉴런 사이에 0 혹은 1이 흘렀다면, **신경망** 에서는 연속적인 실수가 흐른다.</span>

🤔 선형 함수 vs 비선형 함수

- 선형 함수 : 출력이 입력의 상수배만큼 변하는 함수. `f(x) = ax + b` 이고, 이 때 a와 b는 상수이다. 따라서 선형 함수는 곧은 1개의 직선이다.
- 비선형 함수 : 문자 그대로 '선형이 아닌 함수' 이다. 따라서 직선 1개로는 그릴 수 없는 함수이다.

신경망에서는 활성화 함수로 반드시 비선형 함수를 사용해야 한다. 선형 함수를 이용하면 신경망의 층을 깊게 하는 의미가 없어지기 때문이다.

예를 들어, 선형 함수인 `h(x) = cx` 를 활성화 함수로 사용한 3층 네트워크를 떠올려 보자. 식으로 나타내면 `y(x) = h(h(h(x)))` 가 된다. 이 계산은 `y(x) = c * c * c * x` 처럼 곱셈을 3번 수행하지만, 결국엔 `y(x) = ax` 와 같은 식이다. `a = c^3` 이라고 하면 되는 것이다.

즉, 은닉층이 없는 네트워크로 표현할 수 있다. 이 예처럼 선형 함수를 이용해서는 여러 층으로 구성하는 이점을 살릴 수 없다.

### 3-3. ReLU 함수 

최근에는 ReLU 함수가 많이 쓰이는 추세이다.

![ReLU 함수](/assets/img/post/basic_theory/7.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

```python
def relu(x):
  return np.maximum(0, x)
```

ReLU 함수가 Sigmoid 함수보다 많이 쓰이는 이유는 다음과 같다.

1) 계산 효율성 

Sigmoid 함수는 지수 함수가 포함되기 때문에 나눗셈 같은 복잡한 수학적 연산이 필요하지만, ReLU는 간단하게 계산할 수 있기 때문에 계산 효율성이 높다.

2) Gradient 소실 방지

Sigmoid 함수는 Gradient 소실 문제가 있다. 입력값이 매우 크거나 작다면 기울기도 0에 가까워진다. 기울기가 0에 가까워지면 가중치가 매우 느리게 업데이트 되기 때문에 학습 과정이 늦어지고, 신경망이 수렴하기 어렵게 된다. 또한 local minimum에 갇히게 될 수 있다.

ReLU는 양수 입력 값에 대해 일정한 기울기를 갖고 있으므로 이 문제를 방지할 수 있다.

3) sparse activation(희소 활성화)를 생성한다.

희소 활성화는 많은 뉴런이 0을 출력하게 해서 활성화를 적게 한다는 뜻이다. ReLU 함수의 그래프를 떠올려보면, 음수는 모두 0을 출력하고 양수는 그대로 출력하게 되는데 이를 희소 활성화라고 한다. 이렇게 하면, 계산 비용이 줄고 과적합을 줄이는 데도 도움이 된다. 

하지만 단점 또한 존재한다. DNN에서 일부 뉴런이 비활성화되어서 출력이 0이 되는 **dying ReLU** 문제를 겪는다. 이 문제를 해결하기 위해 **Leaky ReLU** 함수가 등장했다. ReLU 함수의 음수 부분에 작은 기울기를 추가해서 음수 값에 대한 기울기가 0인 문제를 방지하는 것이다.

🤔 그렇다면 더 이상 Sigmoid 함수는 사용하지 않게 된 것일까? 

**그렇지 않다. 사용하는 경우가 다르다**

- 시그모이드를 사용하는 경우

  1) RNN과 같은 일부 경우. 시그모이드가 0과 1 사이의 출력 범위를 가지면서도 확률이나 백분율을 나타내는데 유용하기 때문이다. RNN에서 hidden layer의 출력은 보통 확률을 나타내는데, 시그모이드 함수를 쓰면 출력값이 원하는 확률 범위 내에 있는지 확인할 수 있다.

  2) 실수 값이 0과 1 사이의 값으로 나오므로, 특정 클래스에 속할 확률로 해석할 수 있다는 장점이 있다. 따라서 데이터가 이진, 범주형이라면 시그모이드 함수를 사용하는 것이 좋다.

- ReLU를 사용하는 경우
  1) ReLU 함수의 출력값은 무한 범위다. 즉, 연속 변수에 적합하고 회귀 문제에 사용된다.


## 4. 3층 신경망 구현해보기

### 4-1. 입력층에서 1층으로의 신호 전달을 코드로 구현하기

![입력층 to 1층](/assets/img/post/basic_theory/8.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

```python
X = np.array([1.0, 0.5]) # 입력값
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 가중치
B1 = np.array([0.1, 0.2, 0.3]) # 편향

print(W1.shape) # (2,3)
print(X.shape)  # (2,)
print(B1.shape) # (3,)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print(A1) # [0.3 0.7 1.1]
print(Z1) # [0.57444252 0.66818777 0.75026011]
```

### 4-2. 1에서 2층으로의 신호 전달을 코드로 구현하기

![1층 to 2층](/assets/img/post/basic_theory/9.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

```python
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape) # (3,)
print(W2.shape) # (3, 2)
print(B2.shape) # (2,)

A2 = np.dot(Z1, W2) + B2 # [0.51615984 1.21402696]
Z2 = sigmoid(A2) # [0.62624937 0.7710107 ]
```

### 4-3. 2층에서 출력층으로의 신호 전달

![2층 to 출력층](/assets/img/post/basic_theory/10.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

```python
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(A3) # [0.31682708 0.69627909]
print(Y) # [0.31682708 0.69627909]
```

항등 함수인 `identity_function()` 을 정의하고, 이를 출력층의 활성화 함수로 이용했다. 항등 함수는 입력을 그대로 출력하는 함수이다.

✅ 출력층의 활성화 함수는 풀고자 하는 문제의 성질에 맞게 정의한다. 예를 들어, 회귀에는 항등 함수를, 이진 분류에는 시그모이드 함수를, 다중 분류에는 소프트맥스 함수를 사용하는 것이 일반적이다.

### 4-4. 구현 정리

```python

'''
가중치와 편향 초기화하고 딕셔너리 변수인 network에 저장
'''
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)

print(y) # [0.31682708 0.69627909]
```

## 5. softmax 함수

softmax() 함수는 지수 함수를 사용하는데, 지수 함수는 쉽게 아주 큰 값을 내뱉는다. 예를 들어, `e^1000` 은 무한대를 뜻하는 `inf` 가 되어 돌아온다. 즉, `오버플로우`가 발생한다.
해결책으로 소프트멕스 함수의 지수 함수를 계산할 때 어떤 정수를 더하거나 빼도 결과는 바뀌지 않는다는 것을 이용한다. 보통 입력 신호 중 **최댓값** 을 이용한다.

소프트멕스 함수 출력의 총합은 1 이다. 이 성질 덕분에 소프트멕스 함수의 출력을 '확률'로 해석할 수 있다.

```python

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로우 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y) # [0.01821127, 0.24519181, 0.73659691]

np.sum(y) # 1.0 
```

## 6. 출력층의 뉴런 수 정하기

출력층의 뉴런 수는 풀려는 문제에 맞게 적절히 정해야 한다. <span style="color:violet">분류에서는 분류하고 싶은 클래스 수로 설정하는 것이 일반적이다.</span>
예를 들어, 입력 이미지를 숫자 0부터 9 중 하나로 분류하는 문제라면 출력층의 뉴런을 10개로 설정한다.

## 6. 참조
---
* 밑바닥부터 시작하는 딥러닝
