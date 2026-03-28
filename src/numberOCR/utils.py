import random
import math

# =========================
# 工具函数
# =========================

def softmax(z):
    m = max(z)  # 防止溢出
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    return [e / s for e in exps]

def cross_entropy(probs, y):
    return -math.log(probs[y] + 1e-12)

def relu(z):
    return [max(0, v) for v in z]


# =========================
# 初始化模型
# =========================

def init_model(input_size=784, hidden_size=64, output_size=10):
    W1 = [[random.uniform(-0.01, 0.01) for _ in range(input_size)] for _ in range(hidden_size)]
    b1 = [0.0 for _ in range(hidden_size)]

    W2 = [[random.uniform(-0.01, 0.01) for _ in range(hidden_size)] for _ in range(output_size)]
    b2 = [0.0 for _ in range(output_size)]

    return W1, b1, W2, b2


# =========================
# 前向传播
# =========================

def forward(x, W1, b1, W2, b2):
    # z1 = W1·x + b1
    z1 = []
    for i in range(len(W1)):
        s = b1[i]
        for j in range(len(x)):
            s += W1[i][j] * x[j]
        z1.append(s)

    # a1 = ReLU(z1)
    a1 = relu(z1)

    # z2 = W2·a1 + b2
    z2 = []
    for i in range(len(W2)):
        s = b2[i]
        for j in range(len(a1)):
            s += W2[i][j] * a1[j]
        z2.append(s)

    probs = softmax(z2)

    return z1, a1, probs


# =========================
# 反向传播
# =========================

def backward(x, z1, a1, probs, y, W2):
    H = len(a1)
    O = len(probs)

    # ---- 1. 输出层梯度 ----
    grad_z2 = probs[:]   # dL/dz2
    grad_z2[y] -= 1

    # ---- 2. W2, b2 ----
    dW2 = [[0.0 for _ in range(H)] for _ in range(O)]
    db2 = [0.0 for _ in range(O)]

    for i in range(O):
        db2[i] = grad_z2[i]
        for j in range(H):
            dW2[i][j] = grad_z2[i] * a1[j]

    # ---- 3. 传回隐藏层 ----
    grad_a1 = [0.0 for _ in range(H)]

    for j in range(H):
        s = 0.0
        for i in range(O):
            s += grad_z2[i] * W2[i][j]
        grad_a1[j] = s

    # ---- 4. ReLU ----
    grad_z1 = [0.0 for _ in range(H)]
    for j in range(H):
        if z1[j] > 0:
            grad_z1[j] = grad_a1[j]
        else:
            grad_z1[j] = 0.0

    # ---- 5. W1, b1 ----
    dW1 = [[0.0 for _ in range(len(x))] for _ in range(H)]
    db1 = [0.0 for _ in range(H)]

    for i in range(H):
        db1[i] = grad_z1[i]
        for j in range(len(x)):
            dW1[i][j] = grad_z1[i] * x[j]

    return dW1, db1, dW2, db2


# =========================
# 更新参数
# =========================

def update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr=0.01):
    H = len(W1)
    O = len(W2)

    # 更新 W1, b1
    for i in range(H):
        b1[i] -= lr * db1[i]
        for j in range(len(W1[0])):
            W1[i][j] -= lr * dW1[i][j]

    # 更新 W2, b2
    for i in range(O):
        b2[i] -= lr * db2[i]
        for j in range(len(W2[0])):
            W2[i][j] -= lr * dW2[i][j]


# =========================
# 预测
# =========================

def predict(x, W1, b1, W2, b2):
    _, _, probs = forward(x, W1, b1, W2, b2)
    return probs.index(max(probs)), probs


# =========================
# 训练
# =========================

def train(dataset, W1, b1, W2, b2, epochs=50, lr=0.01):
    for epoch in range(epochs):
        total_loss = 0

        for x, y in dataset:
            random.shuffle(dataset)  # 每轮打乱数据
            z1, a1, probs = forward(x, W1, b1, W2, b2)
            loss = cross_entropy(probs, y)
            total_loss += loss

            dW1, db1, dW2, db2 = backward(x, z1, a1, probs, y, W2)
            update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        print(f"epoch {epoch}, loss={total_loss/len(dataset):.4f}")