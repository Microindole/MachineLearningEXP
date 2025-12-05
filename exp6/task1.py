import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  # 用于画好看的混淆矩阵，如果没有安装请 pip install seaborn

# ==========================================
# 1. 定义超参数与配置
# ==========================================
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 50  # 建议提升到 50 轮以观察极限
LR = 0.001  # 学习率
IMG_SIZE = 64
NOISE_FACTOR = 0.2

# 结果保存路径
RESULT_DIR = 'results'
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. 数据读取与预处理
# ==========================================
transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# 测试集不应该做旋转翻转，只做基础处理
transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dir = os.path.join('data', 'covid19', 'train')
test_dir = os.path.join('data', 'covid19', 'noisy_test')

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print(f"类别: {class_names}")


# ==========================================
# 3. 定义模型结构
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


autoencoder = Autoencoder().to(device)
cnn = CNN(len(class_names)).to(device)

criterion_ae = nn.MSELoss()
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=LR)
criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(cnn.parameters(), lr=LR)


# ==========================================
# 4. 辅助函数：记录梯度
# ==========================================
def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# ==========================================
# 5. 训练流程
# ==========================================
history = {'ae_loss': [], 'cnn_loss': [], 'cnn_acc': [], 'cnn_grad': []}


def add_noise(img):
    noise = torch.randn_like(img) * NOISE_FACTOR
    return torch.clamp(img + noise, 0., 1.)


print("\n========== 阶段一：训练自编码器 ==========")
for epoch in range(EPOCHS):
    autoencoder.train()
    running_loss = 0.0
    for imgs, _ in tqdm(train_loader, desc=f'AE Epoch {epoch+1}/{EPOCHS}'):
        imgs = imgs.to(device)
        noisy_imgs = add_noise(imgs)

        optimizer_ae.zero_grad()
        outputs = autoencoder(noisy_imgs)
        loss = criterion_ae(outputs, imgs)
        loss.backward()
        optimizer_ae.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    history['ae_loss'].append(avg_loss)
    if (epoch+1) % 5 == 0:
        print(f"AE Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

print("\n========== 阶段二：训练 CNN ==========")
for epoch in range(EPOCHS):
    cnn.train()
    autoencoder.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_grad = 0.0  # 记录本轮平均梯度

    for imgs, labels in tqdm(train_loader, desc=f'CNN Epoch {epoch+1}/{EPOCHS}'):
        imgs, labels = imgs.to(device), labels.to(device)
        noisy_imgs = add_noise(imgs)

        with torch.no_grad():
            denoised_imgs = autoencoder(noisy_imgs)

        optimizer_cnn.zero_grad()
        outputs = cnn(denoised_imgs)
        loss = criterion_cnn(outputs, labels)
        loss.backward()

        epoch_grad += get_grad_norm(cnn)

        optimizer_cnn.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    acc = 100 * correct / total
    avg_grad = epoch_grad / len(train_loader)

    history['cnn_loss'].append(avg_loss)
    history['cnn_acc'].append(acc)
    history['cnn_grad'].append(avg_grad)

    if (epoch+1) % 5 == 0:
        print(f"CNN Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, Grad: {avg_grad:.4f}")

# ==========================================
# 6. 最终评估与可视化
# ==========================================
print("\n========== 最终测试与详细指标 ==========")
cnn.eval()
autoencoder.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        denoised = autoencoder(imgs)  # 测试集本身带噪
        outputs = cnn(denoised)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 1. 打印详细分类报告 (Precision, Recall, F1)
print("\n分类报告 (Classification Report):")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 2. 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix3.png'))  # 保存图片
plt.show()

# 3. 绘制训练曲线 + 梯度变化
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history['cnn_loss'], label='Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['cnn_acc'], label='Accuracy', color='green')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history['cnn_grad'], label='Gradient Norm', color='red')
plt.title('Gradient Flow (Check for vanishing/exploding)')
plt.xlabel('Epoch')
plt.legend()

plt.savefig(os.path.join(RESULT_DIR, 'training_metrics3.png'))  # 保存图片
plt.show()

# 4. 保存去噪对比图
sample_imgs, _ = next(iter(test_loader))
sample_imgs = sample_imgs.to(device)
with torch.no_grad():
    denoised_samples = autoencoder(sample_imgs)

noisy_np = sample_imgs.cpu().numpy()
denoised_np = denoised_samples.cpu().numpy()

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(noisy_np[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.axis('off')
    if i == 2: plt.title("Original Noisy")

    plt.subplot(2, 5, i + 6)
    plt.imshow(denoised_np[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.axis('off')
    if i == 2: plt.title("Denoised")
plt.savefig(os.path.join(RESULT_DIR, 'denoising_result3.png'))  # 保存图片
plt.show()

# ==========================================
# 7. 保存模型 (新增部分)
# ==========================================
# 保存自编码器权重
ae_path = os.path.join(RESULT_DIR, 'autoencoder.pth')
torch.save(autoencoder.state_dict(), ae_path)
print(f"自编码器模型已保存至: {ae_path}")

# 保存 CNN 权重
cnn_path = os.path.join(RESULT_DIR, 'cnn.pth')
torch.save(cnn.state_dict(), cnn_path)
print(f"CNN 模型已保存至: {cnn_path}")

print(f"所有结果图片已保存至 {os.path.abspath(RESULT_DIR)} 文件夹")