import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import cycle

# ==========================================
# 1. 解决依赖与导入
# ==========================================
try:
    from keras_preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences
except ImportError:
    print("正在尝试从 tensorflow 直接导入...")
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, SpatialDropout1D, GRU
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping

# 解决中文乱码问题 (Windows下通常需要)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果保存目录
RESULT_DIR = 'results_task2_final'
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================
# 2. 配置与超参数 (CPU 极致优化版)
# ==========================================
TRAIN_PATH = os.path.join('data', 'review', 'drugsComTrain_raw.csv')
TEST_PATH = os.path.join('data', 'review', 'drugsComTest_raw.csv')

# --- 针对 CPU 和 过拟合 的关键修改 ---
VOCAB_SIZE = 5000  # 降维：只保留最高频的5000词
MAX_LEN = 100  # 提速：序列长度缩短，计算量减少30%
EMBED_DIM = 64  # 降维
BATCH_SIZE = 64  # 适合 CPU 的批次大小
EPOCHS = 15

print(f"结果将保存至: {os.path.abspath(RESULT_DIR)}")

# ==========================================
# 3. 数据读取与预处理
# ==========================================
if not os.path.exists(TRAIN_PATH):
    print(f"错误：找不到文件 {TRAIN_PATH}")
    exit()

print("1. 正在读取数据...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# --- 图表1：数据类别分布 (写报告用：分析样本不平衡) ---
def plot_label_distribution(df, title, filename):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df['rating'].apply(lambda x: 'Negative' if x <= 4 else ('Neutral' if x <= 6 else 'Positive')))
    plt.title(title)
    plt.ylabel('样本数量')
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()


print("   正在绘制数据分布图...")
plot_label_distribution(train_df, '训练集情感类别分布', 'dist_train_labels.png')


# 处理标签
def process_rating(rating):
    if rating <= 4:
        return 0
    elif rating <= 6:
        return 1
    else:
        return 2


y_train_raw = train_df['rating'].apply(process_rating).values
y_test_raw = test_df['rating'].apply(process_rating).values

y_train = to_categorical(y_train_raw, num_classes=3)
y_test = to_categorical(y_test_raw, num_classes=3)

# 处理文本
print("2. 正在进行文本序列化...")
X_train_text = train_df['review'].astype(str).values
X_test_text = test_df['review'].astype(str).values


# --- 图表2：文本长度分布 (写报告用：证明 MAX_LEN=100 是合理的) ---
def plot_length_distribution(texts, filename):
    lengths = [len(s.split()) for s in texts]
    plt.figure(figsize=(10, 5))
    sns.histplot(lengths, bins=50, kde=True)
    plt.axvline(x=MAX_LEN, color='r', linestyle='--', label=f'截断长度 ({MAX_LEN})')
    plt.title('评论文本长度分布')
    plt.xlabel('单词数量')
    plt.ylabel('频次')
    plt.xlim(0, 500)  # 限制显示范围以便看清主体
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()


print("   正在绘制长度分布图...")
plot_length_distribution(X_train_text, 'dist_text_length.png')

# Tokenizer 处理
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_text)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post')

# ==========================================
# 4. 搭建模型 (GRU + SpatialDropout)
# ==========================================
print("3. 搭建模型...")
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM))

# 抗过拟合层
model.add(SpatialDropout1D(0.4))

# 使用 GRU (比 LSTM 快，效果相当)
model.add(GRU(32, dropout=0.3, recurrent_dropout=0.0))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, MAX_LEN))
model.summary()

# 尝试绘制模型结构图 (需要安装 graphviz，如果报错会自动跳过)
try:
    plot_model(model, to_file=os.path.join(RESULT_DIR, 'model_structure.png'),
               show_shapes=True, show_layer_names=True)
    print("   模型结构图已保存。")
except Exception as e:
    print("   提示：未安装 Graphviz，跳过绘制模型结构图。")

# ==========================================
# 5. 训练模型
# ==========================================
print("\n4. 开始训练...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    verbose=1)

# ==========================================
# 6. 评估与可视化 (升级版)
# ==========================================
print("\n5. 生成评估图表...")

# --- 图表3：训练曲线 (Accuracy & Loss) ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='训练集准确率')
plt.plot(epochs_range, val_acc, label='验证集准确率')
plt.title('准确率变化曲线')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='训练集 Loss')
plt.plot(epochs_range, val_loss, label='验证集 Loss')
plt.title('Loss 变化曲线')
plt.legend()
plt.savefig(os.path.join(RESULT_DIR, 'curve_training.png'))
plt.close()

# 预测
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
class_names = ['Negative', 'Neutral', 'Positive']

# --- 图表4：混淆矩阵 ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.savefig(os.path.join(RESULT_DIR, 'matrix_confusion.png'))
plt.close()

# --- 图表5：ROC 曲线 (高端图表，适合报告) ---
# 计算每个类别的 ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3
y_test_bin = label_binarize(y_true, classes=[0, 1, 2])  # 二值化用于 ROC 计算

plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC class {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('多分类 ROC 曲线')
plt.legend(loc="lower right")
plt.savefig(os.path.join(RESULT_DIR, 'curve_roc.png'))
plt.close()

# 保存分类报告文本
report = classification_report(y_true, y_pred, target_names=class_names)
with open(os.path.join(RESULT_DIR, 'report_classification.txt'), 'w') as f:
    f.write(report)

print(f"\n所有任务完成！请查看文件夹: {RESULT_DIR}")
print("包含以下图表用于实验报告：")
print("1. dist_train_labels.png (数据分布)")
print("2. dist_text_length.png (文本长度分布)")
print("3. curve_training.png (训练曲线)")
print("4. matrix_confusion.png (混淆矩阵)")
print("5. curve_roc.png (ROC 曲线)")
