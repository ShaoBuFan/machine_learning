import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# 定义模型
# =========================

class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 对应你的 W1, b1
        self.fc2 = nn.Linear(hidden_size, output_size) # 对应你的 W2, b2

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 对应你的 z1, a1
        x = self.fc2(x)              # 对应你的 z2（softmax 在 loss 里算）
        return x


# =========================
# 造假数据（模拟 MNIST 格式）
# =========================

N = 100  # 100 个样本
x = torch.randn(N, 784)           # 随机输入
y = torch.randint(0, 10, (N,))    # 随机标签 0~9


# =========================
# 初始化模型、损失函数、优化器
# =========================

model = SimpleNet()
criterion = nn.CrossEntropyLoss()       # 对应你的 softmax + cross_entropy
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 对应你的 update()


# =========================
# 训练
# =========================

for epoch in range(300):
    optimizer.zero_grad()          # 清空上一步的梯度
    outputs = model(x)             # 前向传播
    loss = criterion(outputs, y)   # 计算 loss
    loss.backward()                # 反向传播（自动求梯度）
    optimizer.step()               # 更新参数

    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss={loss.item():.4f}")