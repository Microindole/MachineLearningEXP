import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 解决依赖与导入
# ==========================================
try:
    from keras_preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences
except ImportError:
    print("错误：未安装 keras-preprocessing 库。")
    print("请运行: pip install keras-preprocessing")
    exit()

from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 创建结果保存目录
RESULT_DIR = 'results_task2'
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================
# 2. 配置与超参数 (升级版)
# ==========================================
TRAIN_PATH = os.path.join('data', 'review', 'drugsComTrain_raw.csv')
TEST_PATH = os.path.join('data', 'review', 'drugsComTest_raw.csv')

# --- 改进点：增大模型容量 ---
VOCAB_SIZE = 10000  # 词典大小 (从 5000 -> 10000)
MAX_LEN = 150  # 序列长度 (从 100 -> 150，覆盖更多长评论)
EMBED_DIM = 128  # 词向量维度 (从 64 -> 128)
LSTM_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 20  # 设大一点，配合早停法使用

print(f"结果将保存至: {os.path.abspath(RESULT_DIR)}")

# ==========================================
# 3. 数据处理
# ==========================================
if not os.path.exists(TRAIN_PATH):
    print(f"找不到文件: {TRAIN_PATH}")
    exit()

print("正在读取数据...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# 处理标签: 1-4(消极0), 5-6(中性1), 7-10(积极2)
def process_rating(rating):
    if rating <= 4:
        return 0
    elif rating <= 6:
        return 1
    else:
        return 2


y_train = to_categorical(train_df['rating'].apply(process_rating).values, num_classes=3)
y_test = to_categorical(test_df['rating'].apply(process_rating).values, num_classes=3)

# 处理文本
print("正在构建词典与序列化...")
X_train_text = train_df['review'].astype(str).values
X_test_text = test_df['review'].astype(str).values

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_text)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post')

print(f"数据准备就绪。输入: {X_train.shape}")

# ==========================================
# 4. 搭建模型 (升级为双向 LSTM)
# ==========================================
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM))  # Keras 3 移除了 input_length 参数，可直接省略

# --- 改进点：使用 Bidirectional LSTM ---
# 双向 LSTM 可以同时捕捉上下文信息
model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)))  # 第一层 LSTM，返回序列给下一层
model.add(Dropout(0.3))  # 防止过拟合
model.add(Bidirectional(LSTM(32)))  # 第二层 LSTM
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, MAX_LEN))  # 显式构建模型以显示 summary
model.summary()

# ==========================================
# 5. 训练模型 (加入回调函数)
# ==========================================
# 早停法：如果验证集 Loss 在 3 轮内不下降，就停止训练
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\n开始训练 (Epochs设为20，但可能会提前停止)...")
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],  # 使用回调
                    verbose=1)

# ==========================================
# 6. 保存模型
# ==========================================
model_save_path = os.path.join(RESULT_DIR, 'sentiment_model.keras')
model.save(model_save_path)
print(f"\n模型已保存至: {model_save_path}")

# ==========================================
# 7. 详细评估与可视化
# ==========================================
print("\n正在生成详细评估报告...")

# 预测
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
class_names = ['Negative', 'Neutral', 'Positive']

# --- 7.1 分类报告 (F1, Precision, Recall) ---
report = classification_report(y_true, y_pred, target_names=class_names)
print("\n========== 分类报告 ==========")
print(report)
# 将报告保存到文本文件
with open(os.path.join(RESULT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# --- 7.2 混淆矩阵热力图 ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'))
plt.show()

# --- 7.3 训练曲线 (Accuracy & Loss) ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Loss Curve')
plt.legend()

plt.savefig(os.path.join(RESULT_DIR, 'training_curves.png'))
plt.show()

print(f"所有结果已保存到 {RESULT_DIR} 文件夹中。")
