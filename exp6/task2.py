import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import cycle
from sklearn.utils import class_weight  # [关键新增] 用于计算类别权重

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
from keras.callbacks import ModelCheckpoint

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# [修改点] 结果保存到独立文件夹
RESULT_DIR = 'results_task2_keras_final'
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================
# 2. 配置与超参数 (最终旗舰版)
# ==========================================
TRAIN_PATH = os.path.join('data', 'review', 'drugsComTrain_raw.csv')
TEST_PATH = os.path.join('data', 'review', 'drugsComTest_raw.csv')

# --- 基于数据探测报告的最终决策 ---
VOCAB_SIZE = 5000  # 最佳性价比：覆盖92%内容，去除绝大多数噪音
MAX_LEN = 150  # 最佳拐点：覆盖98%样本，保留99.5%信息量
EMBED_DIM = 64
BATCH_SIZE = 64  # CPU 友好设置
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


# 处理标签函数
def process_rating(rating):
    if rating <= 4:
        return 0
    elif rating <= 6:
        return 1
    else:
        return 2


# 获取原始标签 (0, 1, 2) 用于计算权重
y_train_raw = train_df['rating'].apply(process_rating).values
y_test_raw = test_df['rating'].apply(process_rating).values

# 转为 One-hot
y_train = to_categorical(y_train_raw, num_classes=3)
y_test = to_categorical(y_test_raw, num_classes=3)

# 处理文本
print("2. 正在进行文本序列化...")
X_train_text = train_df['review'].astype(str).values
X_test_text = test_df['review'].astype(str).values

# Tokenizer 处理
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_text)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post')

# [关键新增] 计算类别权重，解决 Neutral 样本过少的问题
print("3. 计算类别权重 (Class Weights)...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_raw),
    y=y_train_raw
)
class_weights_dict = dict(enumerate(class_weights))
print(f"   权重策略: {class_weights_dict}")
# 预期结果: 类别1(Neutral)的权重会非常大，强迫模型关注它

# ==========================================
# 4. 搭建模型 (GRU + SpatialDropout)
# ==========================================
print("4. 搭建模型...")
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM))

# 抗过拟合层
model.add(SpatialDropout1D(0.4))

# 使用 GRU (CPU上比LSTM快30%)
model.add(GRU(32, dropout=0.3, recurrent_dropout=0.0))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, MAX_LEN))
model.summary()

# ==========================================
# 5. 训练模型 (带权重 + 保存文件)
# ==========================================
print("\n5. 开始训练...")


# 1. 早停：防止过拟合
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 2. 保存：只保存最好的模型到硬盘
checkpoint = ModelCheckpoint(
    filepath=os.path.join(RESULT_DIR, 'best_model.keras'), # 保存路径
    monitor='val_loss',
    save_best_only=True,    # 只保存最好的
    save_weights_only=False, # 保存整个模型（结构+权重）
    mode='min',
    verbose=1
)

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop, checkpoint],
                    class_weight=class_weights_dict,
                    verbose=1)

# ==========================================
# 6. 评估与可视化
# ==========================================
print("\n6. 生成评估图表...")

# 训练曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.title('Keras: Accuracy Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Keras: Loss Curve')
plt.legend()
plt.savefig(os.path.join(RESULT_DIR, 'keras_curves.png'))
plt.close()

# 预测与混淆矩阵
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
class_names = ['Negative', 'Neutral', 'Positive']

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Keras: Confusion Matrix')
plt.savefig(os.path.join(RESULT_DIR, 'keras_confusion_matrix.png'))
plt.close()

# ROC 曲线
y_test_bin = label_binarize(y_true, classes=[0, 1, 2])
fpr, tpr, roc_auc = dict(), dict(), dict()
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'green', 'red'])
for i, color in zip(range(3), colors):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.legend(loc="lower right")
plt.title('Keras: Multi-class ROC')
plt.savefig(os.path.join(RESULT_DIR, 'keras_roc.png'))
plt.close()

# 保存报告
report = classification_report(y_true, y_pred, target_names=class_names)
with open(os.path.join(RESULT_DIR, 'keras_report.txt'), 'w') as f:
    f.write(report)

print(f"Keras 任务完成！结果位于: {RESULT_DIR}")
