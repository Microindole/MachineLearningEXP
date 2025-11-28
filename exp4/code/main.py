import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import sys
import os

# 定义一个同时往“屏幕”和“文件”输出的类
class Logger(object):
    def __init__(self, filename='run.log'):
        self.terminal = sys.stdout  # 记录原来的屏幕输出句柄
        self.log = open(filename, 'w', encoding='utf-8') # 打开文件

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- 核心操作：劫持系统的输出通道 ---
sys.stdout = Logger('run.log')
sys.stderr = sys.stdout

# ==========================================
#  实验指导书 第3部分：数据预处理
# ==========================================
print("正在加载本地数据...")
path = '../data/'  # 确保路径正确

# 3.1 加载数据 [cite: 2]
# (使用本地文件代替 mnist.load_data 是合理的变通)
X_train = np.load(path + 'x_train.npy')
y_train = np.load(path + 'y_train.npy')
X_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\n--- 开始数据预处理 ---")

# 3.2 重塑 (Reshape) 和 类型转换 [cite: 2]
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
print(f"1. 训练集已重塑: {X_train.shape}")

# 3.3 归一化 (Normalize)
X_train /= 255
X_test /= 255
print("2. 像素归一化完成")

# 【必须补充】查看X_train第2个例子的第100个到150个像素点的值
print("\n--- [实验要求] 查看 X_train[1] 的 100-150 像素值 ---")
# 注意：指导书原文输出结果是一串 0. 和部分非零小数
print(X_train[1, 100:150])

# 3.4 One-Hot 编码 [cite: 8]
num_classes = 10
Y_train = keras.utils.to_categorical(y_train, num_classes)
Y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"3. 标签 One-Hot 编码完成, 示例: {Y_train[0]}")

# ==========================================
#  实验指导书 第4部分：搭建与训练神经网络
# ==========================================

print("\n--- 开始搭建神经网络 (Model 1) ---")

# 4.1 添加层 [cite: 10]
# 指导书要求：先 add(Dense) 再 add(Activation)
model = models.Sequential()
# 第一层需要指定 input_shape
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

# 查看模型摘要，验证 param # 是否为 7850 [cite: 11]
model.summary()

# 4.2 编译神经网络 [cite: 11]
# loss用categorical_crossentropy, optimizer用SGD
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=0.01),
              metrics=['accuracy'])

# 4.3 训练神经网络
print("\n--- 开始训练 Model 1 (严格按照指导书: 200 epochs) ---")
# 注意：这里改回了 200，这可能需要几分钟时间
history = model.fit(X_train, Y_train,
                    batch_size=128,
                    epochs=200,
                    verbose=1,
                    validation_split=0.2)

# 4.4 评估神经网络 [cite: 14]
print("\n--- 在测试集上评估 Model 1 ---")
score = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test score (Loss): {score[0]}")
print(f"Test accuracy: {score[1]}")

# ==========================================
#  实验指导书 第5部分：优化神经网络
# ==========================================
print("\n========== 进入实验第5部分：优化模型 ==========")

# 5.1 模型改进 [cite: 15]
# 指导书要求：增加隐藏层(128, relu)，输出层(10, softmax)
model_v2 = models.Sequential()
model_v2.add(layers.Dense(128, input_shape=(784,), activation='relu')) # 隐藏层
model_v2.add(layers.Dense(10, activation='softmax')) # 输出层

print("--- 新模型结构 (增加隐藏层) ---")
model_v2.summary()
# 参数数量应为 118,282 [cite: 15]

# 5.2 编译新模型
model_v2.compile(loss='categorical_crossentropy',
                 optimizer=optimizers.SGD(learning_rate=0.01),
                 metrics=['accuracy'])

# 5.3 训练新模型 [cite: 16]
# 指导书明确指出这里 epochs 设置为 20
print("\n--- 开始训练改进后的 Model 2 (20 epochs) ---")
history_v2 = model_v2.fit(X_train, Y_train,
                          batch_size=128,
                          epochs=20,
                          verbose=1,
                          validation_split=0.2)

# 5.4 评估新模型 [cite: 17]
print("\n--- 评估改进后的 Model 2 ---")
score_v2 = model_v2.evaluate(X_test, Y_test, verbose=1)
print(f"Test score: {score_v2[0]}")
print(f"Test accuracy: {score_v2[1]}")

# 5.5 总结 [cite: 17]
acc_improvement = (score_v2[1] - score[1]) * 100
print(f"\n实验结论：通过增加隐藏层，准确率提升了约 {acc_improvement:.2f}%")
