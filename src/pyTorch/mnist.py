import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# 数据加载
# =========================

transform = transforms.ToTensor()  # 把图片转成 tensor，像素值归一化到 0~1

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)


# =========================
# 模型（和之前一样）
# =========================

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 把 28x28 的图片展平成 784 维向量
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================
# 初始化
# =========================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用驱动: {device}")
print(f"使用设备: {torch.cuda.get_device_name(0)}")
print(f"使用版本: {torch.version.cuda}")

model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# =========================
# 训练
# =========================

for epoch in range(5):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # 每轮结束后测试准确率
    model.eval()
    correct = 0
    with torch.no_grad():  # 测试时不需要计算梯度
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()

    acc = correct / len(test_dataset)
    print(f"epoch {epoch+1}, loss={avg_loss:.4f}, 测试准确率={acc:.4f}")