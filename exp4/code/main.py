import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import sys
import os
import matplotlib.pyplot as plt
import itertools

class Logger(object):
    def __init__(self, filename='run.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger('run.log')
sys.stderr = sys.stdout

output_dirs = ['../model', '../images']
for d in output_dirs:
    if not os.path.exists(d):
        os.makedirs(d)


# 数据预处理
print("正在加载本地数据...")
path = '../data/'
X_train = np.load(path + 'x_train.npy')
y_train = np.load(path + 'y_train.npy')
X_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

demo_matrix = X_train[0][12:17, 12:17]
print(demo_matrix)

plt.figure()
plt.imshow(X_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]} (Matrix Principle Demo)")
plt.savefig('../images/demo_matrix_principle.png')
plt.close()

# 数据集全貌图
print("\n--- 生成数据集全貌图 (10x10) ---")
plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap='gray')
plt.suptitle("Dataset Overview (First 100 Samples)", fontsize=16)
plt.savefig('../images/data_overview_10x10.png')
plt.close()
print(">> 已保存数据集全貌图: ../images/data_overview_10x10.png")

print("\n--- 开始数据预处理 ---")
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
print(f"1. 训练集已重塑: {X_train.shape}")

# 归一化
X_train /= 255
X_test /= 255
print("2. 像素归一化完成")

# 查看X_train第2个例子的第100个到150个像素点的值
print("\n--- [实验要求] 查看 X_train[1] (第2个例子) 的 100-150 像素值 ---")
print(X_train[1, 100:150])

# One-Hot 编码
num_classes = 10
Y_train = keras.utils.to_categorical(y_train, num_classes)
Y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"3. 标签 One-Hot 编码完成, 示例: {Y_train[0]}")

#  搭建与训练神经网络
print("\n--- 开始搭建神经网络 (Model 1) ---")

# 添加层
model = models.Sequential()
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))
model.summary()

# 编译神经网络
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=0.01),
              metrics=['accuracy'])

# 训练神经网络
print("\n--- 开始训练 Model 1 ---")
history = model.fit(X_train, Y_train,
                    batch_size=128,
                    epochs=200,
                    verbose=1,
                    validation_split=0.2)
print("\n--- 在测试集上评估 Model 1 ---")
score = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test score (Loss): {score[0]}")
print(f"Test accuracy: {score[1]}")

# 优化神经网络
print("\n========== 优化模型 ==========")
model_v2 = models.Sequential()
model_v2.add(layers.Input(shape=(784,)))
model_v2.add(layers.Dense(128, activation='relu'))
model_v2.add(layers.Dense(10, activation='softmax'))

print("--- 增加隐藏层 ---")
model_v2.summary()

model_v2.compile(loss='categorical_crossentropy',
                 optimizer=optimizers.SGD(learning_rate=0.01),
                 metrics=['accuracy'])

print("\n--- 开始训练改进后的 Model 2 (20 epochs) ---")
history_v2 = model_v2.fit(X_train, Y_train,
                          batch_size=128,
                          epochs=20,
                          verbose=1,
                          validation_split=0.2)

print("\n--- 评估改进后的 Model 2 ---")
score_v2 = model_v2.evaluate(X_test, Y_test, verbose=1)
print(f"Test score: {score_v2[0]}")
print(f"Test accuracy: {score_v2[1]}")
acc_improvement = (score_v2[1] - score[1]) * 100
if acc_improvement > 0:
    print(f"\n实验结论：通过增加隐藏层，准确率提升了约 {acc_improvement:.2f}%")
else:
    print(f"\n实验结论：通过增加隐藏层，准确率下降了约 {acc_improvement:.2f}%")


# 训练曲线对比图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Model 1 (No Hidden)')
plt.plot(history_v2.history['accuracy'], label='Model 2 (With Hidden, 20 epochs)')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Model 1 Loss')
plt.plot(history_v2.history['loss'], label='Model 2 Loss')
plt.title('Loss Comparison')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('../images/training_comparison.png')
plt.close()
print("训练曲线对比图: ../images/training_comparison.png")

# 真实预测结果可视化
indices = np.random.randint(0, 10000, 9)
sample_images = X_test[indices]
sample_labels = np.argmax(Y_test[indices], axis=1)

predictions = model_v2.predict(sample_images, verbose=0)
pred_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = sample_images[i].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    color = 'green' if pred_labels[i] == sample_labels[i] else 'red'
    plt.title(f"True: {sample_labels[i]} | Pred: {pred_labels[i]}", color=color)
    plt.axis('off')

plt.suptitle("Model Prediction Results", fontsize=16)
plt.savefig('../images/prediction_result.png')
plt.close()
print("预测结果演示图: ../images/prediction_result.png")

print("\n正在生成混淆矩阵...")
all_preds = model_v2.predict(X_test, verbose=0)
all_pred_labels = np.argmax(all_preds, axis=1)
true_labels = np.argmax(Y_test, axis=1)
cm = tf.math.confusion_matrix(true_labels, all_pred_labels).numpy()
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Model 2)')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('../images/confusion_matrix.png')
plt.close()
print("混淆矩阵: ../images/confusion_matrix.png")

print("\n错题集分析...")
wrong_indices = np.where(all_pred_labels != true_labels)[0]
if len(wrong_indices) > 0:
    # 随机选 9 个错误的案例
    select_num = min(len(wrong_indices), 9)
    selected_wrong = np.random.choice(wrong_indices, select_num, replace=False)

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(selected_wrong):
        plt.subplot(3, 3, i + 1)
        img = X_test[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        # 红色标题显示 真实值 vs 错误预测
        plt.title(f"True: {true_labels[idx]} | Pred: {all_pred_labels[idx]}", color='red', fontsize=12)
        plt.axis('off')

    plt.suptitle(f"Error Analysis (Total Errors: {len(wrong_indices)}/{len(X_test)})", fontsize=16)
    plt.savefig('../images/error_analysis.png')
    plt.close()
    print("错题集分析图: ../images/error_analysis.png")
else:
    print("模型准确率100%，没有错误案例可生成。")

# 保存最终模型
model_save_path = '../model/mnist_model_v2.keras'
model_v2.save(model_save_path)
print(f"\n5. 最终模型: {model_save_path}")