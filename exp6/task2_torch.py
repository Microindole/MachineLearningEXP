import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# [新增] 引入更多 sklearn 指标用于绘图
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from itertools import cycle
from tqdm import tqdm

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

RESULT_DIR = 'results_task2_pytorch_final'
os.makedirs(RESULT_DIR, exist_ok=True)

VOCAB_SIZE = 5000
MAX_LEN = 150
EMBED_DIM = 64
HIDDEN_DIM = 32
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

# ==========================================
# 2. 数据工具
# ==========================================
class TextPipeline:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def fit(self, texts):
        all_words = []
        for text in texts:
            all_words.extend(str(text).lower().split())
        counts = Counter(all_words)
        common_words = counts.most_common(self.vocab_size - 2)
        for word, _ in common_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=100):
        words = str(text).lower().split()
        indices = [self.word2idx.get(w, 1) for w in words[:max_len]]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

class DrugReviewDataset(Dataset):
    def __init__(self, csv_path, pipeline, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.pipeline = pipeline
        self.is_train = is_train

        # [新增] 仅在训练集加载时生成数据分析图
        if is_train:
            self.plot_data_analysis(self.df)

        print(f"正在加载 {'训练集' if is_train else '测试集'}...")
        self.encoded_texts = [self.pipeline.encode(text, MAX_LEN) for text in self.df['review']]
        self.labels = self.df['rating'].apply(self.process_rating).values

    def process_rating(self, rating):
        if rating <= 4: return 0
        elif rating <= 6: return 1
        else: return 2

    def plot_data_analysis(self, df):
        """[新增] 绘制数据分布，用于报告"""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        label_counts = df['rating'].apply(lambda x: 'Negative' if x<=4 else ('Neutral' if x<=6 else 'Positive')).value_counts()
        plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
        plt.title('训练集情感标签分布')

        plt.subplot(1, 2, 2)
        seq_lengths = df['review'].apply(lambda x: len(str(x).split()))
        plt.hist(seq_lengths, bins=50, color='lightgreen', edgecolor='black', range=(0, 400))
        plt.axvline(x=MAX_LEN, color='r', linestyle='--', label=f'截断 ({MAX_LEN})')
        plt.title('评论长度分布')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'report_data_analysis.png'))
        plt.close()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.encoded_texts[idx], torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 3. 定义模型 (升级为 Bi-GRU)
# ==========================================
class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # [修改点] 开启 bidirectional=True
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.5)

        # [修改点] 因为是双向，全连接层的输入维度变为 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        # output, hidden
        _, hidden = self.gru(embedded)

        # [修改点] 拼接双向的 hidden state
        # hidden shape: [num_layers * num_directions, batch, hidden_dim]
        # 取最后两个维度的 hidden state (前向最后一层 + 后向最后一层) 进行拼接
        # hidden[-2, :, :] 是前向，hidden[-1, :, :] 是后向
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        out = self.dropout(hidden_cat)
        out = self.fc(out)
        return out

# ==========================================
# 4. 训练与评估
# ==========================================
def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss, epoch_acc = 0, 0
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
        progress_bar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{acc.item():.2f}'})
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    all_labels = []
    all_preds = []
    all_probs = [] # [新增] 用于画 ROC/PR 曲线

    with torch.no_grad():
        for texts, labels in iterator:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.argmax(1).cpu().numpy())
            # Softmax 获取概率用于绘图
            probs = torch.softmax(predictions, dim=1)
            all_probs.extend(probs.cpu().numpy())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    TRAIN_PATH = os.path.join('data', 'review', 'drugsComTrain_raw.csv')
    TEST_PATH = os.path.join('data', 'review', 'drugsComTest_raw.csv')

    print("构建词典...")
    raw_train = pd.read_csv(TRAIN_PATH)
    pipeline = TextPipeline(vocab_size=VOCAB_SIZE)
    pipeline.fit(raw_train['review'])

    train_dataset = DrugReviewDataset(TRAIN_PATH, pipeline, is_train=True)
    test_dataset = DrugReviewDataset(TEST_PATH, pipeline, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("计算类别权重...")
    y_train_numpy = train_dataset.labels
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train_numpy), y=y_train_numpy)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = SentimentGRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, output_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print("\n开始 PyTorch 训练 (Bi-GRU)...")
    best_valid_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion)
        # 注意这里接收 5 个返回值
        valid_loss, valid_acc, _, _, _ = evaluate_model(model, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(valid_loss)
        history['val_acc'].append(valid_acc)

        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.2f}%')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'best_model.pt'))

    # ==========================================
    # 6. 绘图 (与 Keras 保持一致)
    # ==========================================
    print("\n正在绘制图表...")

    # (1) 训练曲线
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.title('PyTorch: 准确率')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('PyTorch: 损失值')
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, 'pytorch_curves.png'))
    plt.close()

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, 'best_model.pt')))
    _, _, y_true, y_pred, y_probs = evaluate_model(model, test_loader, criterion)
    class_names = ['Negative', 'Neutral', 'Positive']

    # (2) 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('PyTorch: 混淆矩阵')
    plt.savefig(os.path.join(RESULT_DIR, 'pytorch_confusion_matrix.png'))
    plt.close()

    # (3) ROC 曲线
    y_test_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, roc_auc = dict(), dict(), dict()
    plt.figure(figsize=(8, 6))
    colors = cycle(['blue', 'green', 'red'])
    for i, color in zip(range(3), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC {0} (AUC={1:0.2f})'.format(class_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.legend()
    plt.title('PyTorch: ROC 曲线')
    plt.savefig(os.path.join(RESULT_DIR, 'pytorch_roc.png'))
    plt.close()

    # (4) PR 曲线
    precision, recall, average_precision = dict(), dict(), dict()
    plt.figure(figsize=(8, 6))
    for i, color in zip(range(3), colors):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_probs[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_probs[:, i])
        plt.plot(recall[i], precision[i], color=color, lw=2, label='PR {0} (AP={1:0.2f})'.format(class_names[i], average_precision[i]))
    plt.legend()
    plt.title('PyTorch: PR 曲线')
    plt.savefig(os.path.join(RESULT_DIR, 'pytorch_pr_curve.png'))
    plt.close()

    # 文本报告
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(RESULT_DIR, 'pytorch_report.txt'), 'w') as f:
        f.write(report)

    print(f"PyTorch 复现完成！结果位于: {RESULT_DIR}")