import torch # 导入 PyTorch 深度学习框架
import torch.nn as nn # 导入神经网络模块
import torch.nn.functional as F # 导入常用的神经网络函数（如激活函数）
import torch.optim as optim # 导入优化器模块
from torchvision import datasets, transforms # 导入视觉数据库和图像变换工具
from torch.utils.data import DataLoader # 导入数据加载器

# 1. 硬件配置：检查是否有可用的 GPU (CUDA)，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 超参数设置：这些参数会影响训练速度和效果
batch_size = 64      # 每批处理的样本数量
learning_rate = 0.01 # 学习率：模型学习的“步伐”大小
epochs = 1           # 训练总轮次：模型把整个数据集完整看多少遍

# 3. 数据预处理 & 加载
# Compose 可以将多个变换操作组合在一起
transform = transforms.Compose([
    transforms.ToTensor(), # 将图片转换为张量 (Tensor)，并将像素值缩放到 [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # 使用 MNIST 数据集的均值和标准差进行标准化
])

# 下载并加载 MNIST 训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# DataLoader 负责打乱数据、分批次读取，是训练循环的核心工具
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 4. 定义 CNN 模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层 1：输入 1 个通道（灰度图），输出 32 个特征图，卷积核大小 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # 卷积层 2：输入 32 个通道，输出 64 个特征图，卷积核大小 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # 最大池化层：2x2 窗口，步长为 2，图像尺寸会减半
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层 1：输入数据经过卷积和池化后，尺寸变为 64*5*5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        # 全连接层 2：输出层，10 个节点对应 0-9 这十个数字
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播流程：卷积 -> ReLU 激活 -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平 (Flatten)：将多维张量拉伸为一维，为进入全连接层做准备
        x = x.view(-1, 64 * 5 * 5)
        # 全连接层运算 -> ReLU 激活
        x = F.relu(self.fc1(x))
        # 得到最终输出（未经过 Softmax，因为 CrossEntropyLoss 内部自带了）
        x = self.fc2(x)
        return x

# 实例化模型并迁移到指定的计算设备 (CPU/GPU)
model = SimpleCNN().to(device)

# 5. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失：常用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam 优化器：一种自适应学习率的算法

# 6. 训练函数
def train(epoch):
    model.train() # 将模型设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据迁移到对应的设备
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()    # 1. 梯度清零（PyTorch 默认会累加梯度）
        output = model(data)     # 2. 前向传播：计算模型输出
        loss = criterion(output, target) # 3. 计算损失值
        loss.backward()          # 4. 反向传播：计算所有权重的梯度
        optimizer.step()         # 5. 更新权重：根据梯度调整参数
        
        # 每隔 100 个批次打印一次进度
        if batch_idx % 100 == 0:
            print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

# 7. 测试函数
def test():
    model.eval() # 将模型设置为评估模式（例如关闭随机性的 Dropout）
    test_loss = 0
    correct = 0
    # 在测试阶段不需要计算梯度，这能节省内存和计算资源
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # 累计测试损失
            pred = output.argmax(dim=1, keepdim=True)    # 找到预测概率最大的数字
            correct += pred.eq(target.view_as(pred)).sum().item() # 统计预测正确的数量

    test_loss /= len(test_loader.dataset)
    print(f'\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')

# 8. 执行主程序
if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch) # 训练一轮
        test()  # 测试准确率
    
    # 全部训练完成后，保存模型的权重信息到本地文件
    torch.save(model.state_dict(), "CNN_01/mnist_cnn.pt")
    print("模型已保存为 CNN_01/mnist_cnn.pt")
