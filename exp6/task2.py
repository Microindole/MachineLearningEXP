import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import cycle
from sklearn.utils import class_weight

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

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from keras.models import Sequential
# [修改点] 引入 Bidirectional
from keras.layers import Embedding, Dense, Dropout, SpatialDropout1D, GRU, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = 'results_task2_keras_final'
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================
# 2. 配置与超参数
# ==========================================
TRAIN_PATH = os.path.join('data', 'review', 'drugsComTrain_raw.csv')
TEST_PATH = os.path.join('data', 'review', 'drugsComTest_raw.csv')

VOCAB_SIZE = 5000
MAX_LEN = 150
EMBED_DIM = 64
BATCH_SIZE = 64
EPOCHS = 15

# ==========================================
# [新增] 辅助绘图函数 (为报告增加素材)
# ==========================================
def plot_data_analysis(train_df, result_dir):
    """绘制数据分布图和长度分布图，用于实验报告的'数据分析'部分"""
    print("正在生成数据分析图表...")

    # 图1：情感标签分布（饼图）
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    label_counts = train_df['rating'].apply(lambda x: 'Negative' if x<=4 else ('Neutral' if x<=6 else 'Positive')).value_counts()
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('训练集情感标签分布')

    # 图2：评论长度分布（直方图）
    plt.subplot(1, 2, 2)
    seq_lengths = train_df['review'].apply(lambda x: len(str(x).split()))
    plt.hist(seq_lengths, bins=50, color='skyblue', edgecolor='black', range=(0, 400))
    plt.axvline(x=MAX_LEN, color='r', linestyle='--', label=f'截断长度 ({MAX_LEN})')
    plt.title('评论单词数分布')
    plt.xlabel('单词数量')
    plt.ylabel('样本数')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'report_data_analysis.png'))
    plt.close()
    print(f"  -> 数据分析图已保存: report_data_analysis.png")

# ==========================================
# 3. 数据读取与预处理
# ==========================================
if not os.path.exists(TRAIN_PATH):
    print(f"错误：找不到文件 {TRAIN_PATH}")
    exit()

print("1. 正在读取数据...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# [新增] 调用绘图函数
plot_data_analysis(train_df, RESULT_DIR)

def process_rating(rating):
    if rating <= 4: return 0
    elif rating <= 6: return 1
    else: return 2

y_train_raw = train_df['rating'].apply(process_rating).values
y_test_raw = test_df['rating'].apply(process_rating).values

y_train = to_categorical(y_train_raw, num_classes=3)
y_test = to_categorical(y_test_raw, num_classes=3)

print("2. 正在进行文本序列化...")
X_train_text = train_df['review'].astype(str).values
X_test_text = test_df['review'].astype(str).values

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_text)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post')

print("3. 计算类别权重 (Class Weights)...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_raw),
    y=y_train_raw
)
class_weights_dict = dict(enumerate(class_weights))
print(f"   权重策略: {class_weights_dict}")

# ==========================================
# 4. 搭建模型 (已升级为双向 GRU)
# ==========================================
print("4. 搭建模型 (Bi-GRU)...")
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM))
model.add(SpatialDropout1D(0.4))

# [关键修改] 使用 Bidirectional 包裹 GRU
# 这能让模型同时利用上下文，显著提升对长句的理解能力
model.add(Bidirectional(GRU(32, dropout=0.3, recurrent_dropout=0.0)))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, MAX_LEN))
model.summary()

# ==========================================
# 5. 训练模型
# ==========================================
print("\n5. 开始训练...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(RESULT_DIR, 'best_model.keras'),
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop, checkpoint],
                    class_weight=class_weights_dict,
                    verbose=1)

# ==========================================
# 6. 评估与可视化 (增强版)
# ==========================================
print("\n6. 生成评估图表...")

# (1) 训练曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.title('准确率变化曲线')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('损失值变化曲线')
plt.legend()
plt.savefig(os.path.join(RESULT_DIR, 'keras_curves.png'))
plt.close()

# 预测
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
class_names = ['Negative', 'Neutral', 'Positive']

# (2) 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('混淆矩阵 (Confusion Matrix)')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.savefig(os.path.join(RESULT_DIR, 'keras_confusion_matrix.png'))
plt.close()

# (3) ROC 曲线
y_test_bin = label_binarize(y_true, classes=[0, 1, 2])
fpr, tpr, roc_auc = dict(), dict(), dict()
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'green', 'red'])
for i, color in zip(range(3), colors):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC {0} (AUC = {1:0.2f})'.format(class_names[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.legend(loc="lower right")
plt.title('多分类 ROC 曲线')
plt.savefig(os.path.join(RESULT_DIR, 'keras_roc.png'))
plt.close()

# [新增] (4) 精确率-召回率曲线 (PR Curve) - 专门用于展示不平衡数据效果
precision, recall, average_precision = dict(), dict(), dict()
plt.figure(figsize=(8, 6))
for i, color in zip(range(3), colors):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_probs[:, i])
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='PR {0} (AP = {1:0.2f})'.format(class_names[i], average_precision[i]))
plt.xlabel('Recall (召回率)')
plt.ylabel('Precision (精确率)')
plt.title('精确率-召回率曲线 (PR Curve)')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULT_DIR, 'keras_pr_curve.png'))
plt.close()

# 保存文本报告
report = classification_report(y_true, y_pred, target_names=class_names)
with open(os.path.join(RESULT_DIR, 'keras_report.txt'), 'w') as f:
    f.write(report)

print(f"Keras 任务完成！所有图表已保存至: {RESULT_DIR}")