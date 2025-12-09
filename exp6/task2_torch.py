import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ==========================================
# 1. 配置与环境检测 (CPU 优化)
# ==========================================
# 强制使用 CPU (因为你的环境没有显卡，且我们要保证和 Keras 比较时的公平性)
device = torch.device('cpu')
print(f"使用设备: {device}")

# 结果保存路径
RESULT_DIR = 'results_task2_pytorch'
os.makedirs(RESULT_DIR, exist_ok=True)

# 超参数 (保持与 Keras 优化版一致，方便对比)
VOCAB_SIZE = 5000
MAX_LEN = 100
EMBED_DIM = 64
HIDDEN_DIM = 32
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001


# ==========================================
# 2. 数据预处理工具类
# ==========================================
class TextPipeline:
    """
    手动实现 Tokenizer，不依赖 keras，展示对底层原理的理解
    """

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}  # 0用于填充，1用于未知词
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def fit(self, texts):
        # 统计词频
        all_words = []
        for text in texts:
            all_words.extend(str(text).lower().split())

        counts = Counter(all_words)
        # 取最高频的 N 个词
        common_words = counts.most_common(self.vocab_size - 2)

        for word, _ in common_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=100):
        words = str(text).lower().split()
        # 截断或填充
        indices = [self.word2idx.get(w, 1) for w in words[:max_len]]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))  # Post-padding
        return torch.tensor(indices, dtype=torch.long)


# ==========================================
# 3. 自定义 Dataset
# ==========================================
class DrugReviewDataset(Dataset):
    def __init__(self, csv_path, pipeline, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.pipeline = pipeline
        self.is_train = is_train

        # 预处理文本
        print(f"正在加载 {'训练集' if is_train else '测试集'}...")
        self.encoded_texts = [self.pipeline.encode(text, MAX_LEN) for text in self.df['review']]

        # 处理标签
        self.labels = self.df['rating'].apply(self.process_rating).values
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def process_rating(self, rating):
        if rating <= 4:
            return 0
        elif rating <= 6:
            return 1
        else:
            return 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.encoded_texts[idx], self.labels[idx]


# ==========================================
# 4. 定义模型 (PyTorch 风格)
# ==========================================
class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentGRU, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # GRU 层 (batch_first=True 使得输入维度为 [batch, seq, feature])
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # Dropout 用于防止过拟合
        self.dropout = nn.Dropout(0.5)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, max_len]

        # 1. Embedding
        embedded = self.embedding(x)  # [batch_size, max_len, embed_dim]

        # 2. GRU
        # output: 每个时间步的输出, hidden: 最后一个时间步的隐藏状态
        _, hidden = self.gru(embedded)

        # hidden shape: [1, batch_size, hidden_dim] (因为是单向单层)
        # 我们取最后一个层的 hidden state
        hidden = hidden.squeeze(0)  # [batch_size, hidden_dim]

        # 3. Dropout & FC
        out = self.dropout(hidden)
        out = self.fc(out)  # [batch_size, output_dim]

        return out


# ==========================================
# 5. 训练与评估函数
# ==========================================
def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    # 使用 tqdm 包装 iterator，显示进度条
    # desc='Training' 设置前缀，leave=False 跑完一轮后移除进度条保持清爽
    progress_bar = tqdm(iterator, desc='Training', leave=False)

    for texts, labels in progress_bar:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        acc = (predictions.argmax(1) == labels).float().mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # 在进度条末尾实时显示当前 Loss
        progress_bar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{acc.item():.2f}'})

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []

    # 同样给评估过程加上进度条
    progress_bar = tqdm(iterator, desc='Evaluating', leave=False)

    with torch.no_grad():
        for texts, labels in progress_bar:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_preds.extend(predictions.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), all_labels, all_preds


# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    TRAIN_PATH = os.path.join('data', 'review', 'drugsComTrain_raw.csv')
    TEST_PATH = os.path.join('data', 'review', 'drugsComTest_raw.csv')

    # --- 1. 构建 Pipeline ---
    print("构建词典...")
    raw_train = pd.read_csv(TRAIN_PATH)
    pipeline = TextPipeline(vocab_size=VOCAB_SIZE)
    pipeline.fit(raw_train['review'])

    # --- 2. 加载数据 (DataLoader) ---
    train_dataset = DrugReviewDataset(TRAIN_PATH, pipeline, is_train=True)
    test_dataset = DrugReviewDataset(TEST_PATH, pipeline, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 初始化模型 ---
    model = SentimentGRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, output_dim=3)
    model = model.to(device)

    # 优化器与损失函数
    # 使用 weight_decay (L2正则化) 来进一步抑制过拟合
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # --- 4. 训练循环 ---
    print("\n开始 PyTorch 训练...")
    best_valid_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, _, _ = evaluate_model(model, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(valid_loss)
        history['val_acc'].append(valid_acc)

        print(
            f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} '
            f'| Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc * 100:.2f}%')

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'best_model.pt'))

    # --- 5. 结果可视化 ---
    print("\n正在绘制图表...")

    # 训练曲线
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.title('PyTorch: Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('PyTorch: Loss')
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, 'pytorch_curves.png'))

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, 'best_model.pt')))
    _, _, y_true, y_pred = evaluate_model(model, test_loader, criterion)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    class_names = ['Negative', 'Neutral', 'Positive']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('PyTorch: Confusion Matrix')
    plt.savefig(os.path.join(RESULT_DIR, 'pytorch_confusion_matrix.png'))

    # 分类报告
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(RESULT_DIR, 'pytorch_report.txt'), 'w') as f:
        f.write(report)

    print(f"PyTorch 复现完成！结果已保存至 {RESULT_DIR}")
