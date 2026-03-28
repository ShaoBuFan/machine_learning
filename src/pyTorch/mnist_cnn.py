import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# 数据加载
# =========================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)


# =========================
# CNN 模型
# =========================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1通道->32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32通道->64通道
        self.pool  = nn.MaxPool2d(2, 2)                           # 2x2 池化，尺寸减半

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.pool(torch.relu(self.conv1(x)))  # -> (batch, 32, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)

        x = x.view(-1, 64 * 7 * 7)               # 展平
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# =========================
# 初始化
# =========================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = CNN().to(device)
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

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()

    acc = correct / len(test_dataset)
    print(f"epoch {epoch+1}, loss={avg_loss:.4f}, 测试准确率={acc:.4f}")