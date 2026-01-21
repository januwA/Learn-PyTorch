# Learn PyTorch

这是一个记录 PyTorch 学习历程的项目。包含模型训练、图像处理以及深度学习相关的实验心得。

## 🖥️ 本地环境记录

为了方便以后配置环境，记录当前的硬件与驱动状态：

- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)
- **显卡驱动版本**: 581.32
- **支持的最高 CUDA 版本**: 13.0
- **Python 版本**: 3.14.2 (使用 `uv` 管理)

### GPU 开发环境配置参考

> https://pytorch.org/get-started/locally/

```powershell
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

---

## 📚 项目目录

### 1. [CNN_01 (手写数字识别)](./CNN_01/README.md)
- **目标**: 使用简单的卷积神经网络识别 MNIST 数据集。
- **核心点**: 
  - PyTorch 基础训练循环（5步法）。
  - 使用 OpenCV 对真实图片进行预处理（模糊、二值化、轮廓分割）。
  - 支持多数字图片的分割识别。
- **状态**: ✅ 已完成

---

## 💡 常用命令手册

- **查看 GPU 状态**: `nvidia-smi`
- **运行练习脚本**: `uv run <文件夹>/<脚本名>.py`
- **检查 PyTorch 是否支持 CUDA**:
  ```python
  uv run python -c "import torch; print(torch.cuda.is_available())"

  uv run python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda}')"
  ```
