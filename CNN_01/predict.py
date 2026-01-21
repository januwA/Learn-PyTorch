import torch # 导入 PyTorch 深度学习框架
import torch.nn as nn # 导入神经网络模块
import torch.nn.functional as F # 导入常用的神经网络函数（如激活函数）
from torchvision import transforms # 导入图像预处理工具
from PIL import Image # 导入 Python 图像处理库 PIL
import cv2 # 导入 OpenCV 库，用于传统的计算机视觉操作
import numpy as np # 导入数值计算库 numpy
import os # 导入操作系统接口，用于路径检查

# 定义模型结构 (必须与训练脚本 main.py 中的 SimpleCNN 结构完全对齐)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层：输入1通道（灰度图），输出32通道，卷积核大小3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # 第二个卷积层：输入32通道，输出64通道，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # 2x2 最大池化层：将图像长宽各减半
        self.pool = nn.MaxPool2d(2, 2)
        # 第一个全连接层：输入特征维度 64*5*5，输出128维度
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        # 第二个全连接层：最终输出10个维度（对应数字 0-9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化 (连续两次)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 将多维特征图展平为一维向量，方便喂入全连接层
        x = x.view(-1, 64 * 5 * 5)
        # 全连接 -> 激活
        x = F.relu(self.fc1(x))
        # 最后的输出层
        x = self.fc2(x)
        return x

def predict_multi_digits(image_path, model_path):
    # 检查模型权重文件是否存在
    if not os.path.exists(model_path):
        print("错误: 找不到模型文件")
        return

    # 确定计算设备：有显卡用显卡，没有用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device) # 初始化模型结构
    model.load_state_dict(torch.load(model_path, map_location=device)) # 加载已训练好的权重
    model.eval() # 设置为评估模式（关闭 Dropout 等训练专用组件）

    # 使用 OpenCV 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    # 将彩色图片转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 预处理之高斯迷糊：减少图像噪点
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 预处理之二值化：将图像变成纯黑底白字 (MNIST 风格)
    # OTSU 算法会自动寻找最佳阈值
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 展示中间处理结果窗口（本地运行有用）
    # cv2.imshow('blur', blur) # 显示模糊后图像
    # cv2.imshow('thresh', thresh) # 显示二值化（黑白）图像
    # cv2.waitKey(0) # 等待用户按键
    # cv2.destroyAllWindows() # 关闭所有窗口
    
    # 3. 膨胀处理：让笔画稍微变粗，防止细微线条断裂导致的识别失败
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # 4. 轮廓查找：寻找图像中所有独立的白色块（数字）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按照 X 轴坐标从左向右排序，保证识别顺序符合书写习惯
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    results = []
    # 遍历每一个寻找到的轮廓
    for cnt in contours:
        # 获取轮廓的外接矩形（位置和宽高）
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤掉太小的噪点（比如小于 5 像素的杂点不认为是数字）
        if w < 2 or h < 5: continue 
        
        # 从二值化大图中剪切出该数字区域
        digit_roi = thresh[y:y+h, x:x+w]
        
        # --- 核心处理：将切出来的数字模拟成 MNIST 风格的排列 ---
        size = max(w, h)
        # 创建一个黑色的正方形背景
        square_digit = np.zeros((size, size), dtype=np.uint8)
        # 将提取出的数字贴到正方形背景中心
        dx = (size - w) // 2
        dy = (size - h) // 2
        square_digit[dy:dy+h, dx:dx+w] = digit_roi
        
        # 给数字周围添加 40% 的宽度的黑色边框 (MNIST 喜欢留白)
        pad = int(size * 0.4)
        digit_roi_padded = cv2.copyMakeBorder(square_digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        
        # 将 OpenCV 的 numpy 数组转回 PIL 图片，以便使用 torchvision 的各种变换
        pil_img = Image.fromarray(digit_roi_padded)
        
        # 定义转换流水线：缩放到 28x28 -> 转为张量 -> 分部标准化
        transform = transforms.Compose([
            transforms.Resize((28, 28)), # 网络输入必须是 28x28
            transforms.ToTensor(), # 转化为 [0, 1] 之间的浮点数张量
            transforms.Normalize((0.1307,), (0.3081,)) # MNIST 官方均值和标准差
        ])
        
        # 增加 Batch 维度，因为 PyTorch 需要输入为 [BatchSize, Channels, H, W]
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # 进入推理环境，不计算梯度（为了提速）
        with torch.no_grad():
            output = model(input_tensor) # 模型预测
            pred = output.argmax(dim=1).item() # 找到概率最大的类别索引
            prob = F.softmax(output, dim=1)[0][pred].item() # 获取该预测的置信度百分比
            results.append((pred, prob)) # 保存结果

    # 输出显示
    if not results:
        print("未能识别。请检查 test.jpg 是否为白底黑字，且数字清晰。")
    else:
        # 将所有识别到的数字连成一个字符串
        full_number = "".join(str(r[0]) for r in results)
        print(f"识别到: {full_number}")
        for i, (digit, conf) in enumerate(results):
            # 打印每个数字的明细
            print(f"  [{i+1}] 预测: {digit}, 置信度: {conf:.2f}")

if __name__ == "__main__":
    # 配置输入图片路径和权重路径
    img_path = "CNN_01/test.jpg"
    model_path = "CNN_01/mnist_cnn.pt"
    predict_multi_digits(img_path, model_path)
