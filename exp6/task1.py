import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm  # 用于显示进度条

# ==========================================
# 1. 定义超参数与配置
# ==========================================
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 30         # 根据电脑性能调整，建议至少10轮
LR = 0.001          # 学习率
IMG_SIZE = 64       # 图片统一缩放尺寸 (指导书提及Resize)
NOISE_FACTOR = 0.2  # 训练时手动添加噪声的系数

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. 数据读取与预处理
# ==========================================
# 定义数据变换：转灰度 -> 调整尺寸 -> 转张量
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
    transforms.RandomRotation(10),           # 随机旋转 +/- 10度
    transforms.ToTensor(),
])

# 路径配置
train_dir = os.path.join('data', 'covid19', 'train')
test_dir = os.path.join('data', 'covid19', 'noisy_test')

# 加载数据集
# ImageFolder 会自动根据文件夹名称 (Covid, Normal...) 生成标签
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")
print(f"类别映射: {train_dataset.class_to_idx}")

# ==========================================
# 3. 定义模型结构
# ==========================================

# --- 模型 A: 自编码器 (Autoencoder) ---
# 用于去噪：输入含噪图片 -> 压缩 -> 还原 -> 输出干净图片
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # [batch, 16, 32, 32]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),                      # [batch, 64, 10, 10]
            nn.ReLU()
        )
        # 解码器 (Decoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),             # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # [batch, 16, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # [batch, 1, 64, 64]
            nn.Sigmoid() # 输出范围控制在 0-1 之间
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 模型 B: 卷积神经网络 (CNN) ---
# 用于分类：输入去噪后的图片 -> 卷积特征提取 -> 全连接分类
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 初始化模型
autoencoder = Autoencoder().to(device)
cnn = CNN(num_classes=3).to(device)

# 定义优化器和损失函数
criterion_ae = nn.MSELoss()         # 自编码器用均方误差
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=LR)

criterion_cnn = nn.CrossEntropyLoss() # 分类用交叉熵
optimizer_cnn = optim.Adam(cnn.parameters(), lr=LR)

# ==========================================
# 4. 训练流程
# ==========================================

# 记录训练过程中的数据用于绘图
history = {
    'ae_loss': [],
    'cnn_loss': [],
    'cnn_acc': []
}

def add_noise(img):
    """手动添加高斯噪声 """
    noise = torch.randn_like(img) * NOISE_FACTOR
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

print("\n========== 阶段一：训练自编码器 (去噪) ==========")
for epoch in range(EPOCHS):
    autoencoder.train()
    running_loss = 0.0

    for imgs, _ in tqdm(train_loader, desc=f'AE Epoch {epoch+1}/{EPOCHS}'):
        imgs = imgs.to(device)

        # 1. 制造带噪输入
        noisy_imgs = add_noise(imgs)

        # 2. 前向传播
        outputs = autoencoder(noisy_imgs)

        # 3. 计算损失 (目标是还原成原始干净图片 imgs)
        loss = criterion_ae(outputs, imgs)

        # 4. 反向传播与优化
        optimizer_ae.zero_grad()
        loss.backward()
        optimizer_ae.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    history['ae_loss'].append(avg_loss)
    print(f"AE Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

print("\n========== 阶段二：训练 CNN (分类) ==========")
# 注意：CNN 的输入是经过 Autoencoder 去噪后的图片
for epoch in range(EPOCHS):
    cnn.train()
    autoencoder.eval() # 固定自编码器，不训练它

    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc=f'CNN Epoch {epoch+1}/{EPOCHS}'):
        imgs, labels = imgs.to(device), labels.to(device)

        # 1. 制造带噪输入
        noisy_imgs = add_noise(imgs)

        # 2. 通过自编码器去噪 (不需要计算 AE 的梯度)
        with torch.no_grad():
            denoised_imgs = autoencoder(noisy_imgs)

        # 3. 将去噪后的图输入 CNN
        outputs = cnn(denoised_imgs)
        loss = criterion_cnn(outputs, labels)

        # 4. 反向传播 (只更新 CNN)
        optimizer_cnn.zero_grad()
        loss.backward()
        optimizer_cnn.step()

        running_loss += loss.item()

        # 计算正确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    history['cnn_loss'].append(avg_loss)
    history['cnn_acc'].append(accuracy)

    print(f"CNN Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

# ==========================================
# 5. 模型测试与结果可视化
# ==========================================
print("\n========== 最终测试 (在 noisy_test 上) ==========")
cnn.eval()
autoencoder.eval()

test_correct = 0
test_total = 0

# 这里使用 next(iter(...)) 获取一批数据用于展示去噪效果
sample_imgs, sample_labels = next(iter(test_loader))
sample_imgs = sample_imgs.to(device)

with torch.no_grad():
    # 注意：测试集本身已经是 noisy_test，所以不需要手动 add_noise
    denoised_samples = autoencoder(sample_imgs)

    # 全集测试准确率
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        # 这里的输入本身就是带噪的 (来自 noisy_test)
        denoised = autoencoder(imgs)
        outputs = cnn(denoised)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

print(f"测试集最终准确率: {100 * test_correct / test_total:.2f}%")

# --- 绘图 1: 训练过程 ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history['ae_loss'], label='AE Loss')
plt.title('Autoencoder Loss')
plt.xlabel('Epoch')

plt.subplot(1, 3, 2)
plt.plot(history['cnn_loss'], label='CNN Loss', color='orange')
plt.title('CNN Loss')
plt.xlabel('Epoch')

plt.subplot(1, 3, 3)
plt.plot(history['cnn_acc'], label='CNN Accuracy', color='green')
plt.title('CNN Accuracy')
plt.xlabel('Epoch')
plt.show()

# --- 绘图 2: 去噪效果对比 (取前 5 张) ---
# 将 Tensor 转回 numpy 用于绘图
noisy_np = sample_imgs.cpu().numpy()
denoised_np = denoised_samples.cpu().numpy()

n_show = 5
plt.figure(figsize=(10, 4))
for i in range(n_show):
    # 显示带噪原图
    ax = plt.subplot(2, n_show, i + 1)
    plt.imshow(noisy_np[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title("Noisy Input")
    plt.axis('off')

    # 显示去噪后图
    ax = plt.subplot(2, n_show, i + 1 + n_show)
    plt.imshow(denoised_np[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')
plt.show()