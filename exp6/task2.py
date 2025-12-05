import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 解决 Keras 3 兼容性导入
# ==========================================
# 实验指导书基于旧版 Keras，新版需要单独导入处理工具
try:
    from keras_preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences
except ImportError:
    print("错误：未安装 keras-preprocessing 库。")
    print("请在终端运行: pip install keras-preprocessing")
    exit()

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical

# ==========================================
# 2. 数据读取与配置
# ==========================================
# 根据你的截图，数据在 data/review 目录下
TRAIN_PATH = os.path.join('data', 'review', 'drugsComTrain_raw.csv')
TEST_PATH = os.path.join('data', 'review', 'drugsComTest_raw.csv')

# 超参数
VOCAB_SIZE = 5000   # 词典大小（保留最常出现的5000个词）
MAX_LEN = 100       # 每个评论截断/补齐为100个词
EMBED_DIM = 64      # 词向量维度
LSTM_UNITS = 64     # LSTM 单元数
BATCH_SIZE = 128
EPOCHS = 5          # 训练轮数

print("正在读取数据...")
if not os.path.exists(TRAIN_PATH):
    print(f"找不到文件: {TRAIN_PATH}，请检查路径！")
    exit()

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 只取部分数据进行快速实验（可选，如果跑得慢可以取消注释下面两行）
# train_df = train_df.iloc[:10000]
# test_df = test_df.iloc[:2000]

print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")

# ==========================================
# 3. 数据预处理
# ==========================================

# --- 3.1 标签处理 (Rating -> Sentiment) ---
# 1-4分: 消极(0), 5-6分: 中性(1), 7-10分: 积极(2)
def process_rating(rating):
    if rating <= 4: return 0
    elif rating <= 6: return 1
    else: return 2

print("正在处理标签...")
y_train = train_df['rating'].apply(process_rating).values
y_test = test_df['rating'].apply(process_rating).values

# 转为独热编码 (One-hot)
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# --- 3.2 文本处理 (Review -> Sequence) ---
print("正在处理文本 (这可能需要几秒钟)...")
X_train_text = train_df['review'].astype(str).values
X_test_text = test_df['review'].astype(str).values

# 初始化 Tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
# 仅在训练集上构建词典
tokenizer.fit_on_texts(X_train_text)

# 文本转数字序列
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# 填充序列 (Padding)
X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"数据准备就绪。输入形状: {X_train.shape}, 标签形状: {y_train.shape}")

# ==========================================
# 4. 搭建 LSTM 模型
# ==========================================
model = Sequential()

# 嵌入层: 将整数索引映射为密集向量
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN))

# LSTM 层: 处理序列信息
model.add(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.0))
# 注意: recurrent_dropout 在 GPU 上可能导致无法使用 CuDNN 加速，设为 0 以保证速度

# 输出层: 3分类
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# ==========================================
# 5. 训练与评估
# ==========================================
print("\n开始训练模型...")
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    verbose=1)

# ==========================================
# 6. 结果可视化
# ==========================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()