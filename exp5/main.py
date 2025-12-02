import os
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential, Model, load_model
from keras.layers import InputLayer, Input, Reshape, MaxPooling2D, Conv2D, Dense, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical

from mnist import MNIST

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

save_dir = "images"
os.makedirs(save_dir, exist_ok=True)
print(f"--- 图片保存目录已准备: {os.path.abspath(save_dir)} ---")

print("正在检查/下载数据集...")
data = MNIST(data_dir="data/MNIST/")

# 配置神经网络参数
img_size = data.img_size
img_size_flat = data.img_size_flat
img_shape = data.img_shape
img_shape_full = data.img_shape_full
num_classes = data.num_classes
num_channels = data.num_channels


# 绘制图像的辅助函数
def plot_images(images, cls_true, cls_pred=None, filename=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    # 保存图片
    if filename:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"图片已保存: {save_path}")

    plt.show()


def plot_example_errors(cls_pred, correct, filename=None):
    incorrect = (correct == False)
    images = data.x_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.y_test_cls[incorrect]
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9], filename=filename)


# ------------------------------------------
# 序列模型 (Sequential Model)
# ------------------------------------------
print("\n--- Training Sequential Model ---")

model = Sequential()
model.add(InputLayer(shape=(img_size_flat,)))
model.add(Reshape(img_shape_full))
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 数据预处理
x_train = data.x_train.reshape(-1, img_size_flat) / 255.0
x_test = data.x_test.reshape(-1, img_size_flat) / 255.0
y_train_encoded = to_categorical(data.y_train, num_classes)
y_test_encoded = to_categorical(data.y_test, num_classes)

# 训练
model.fit(x_train, y_train_encoded, epochs=1, batch_size=128)

# 评估
result = model.evaluate(x_test, y_test_encoded)
print("Loss:", result[0])
print("Accuracy:", result[1])

# 预测
predict_prob = model.predict(x_test)
predict_cls = np.argmax(predict_prob, axis=1)
plot_images(images=data.x_test[0:9], cls_true=data.y_test_cls[0:9], cls_pred=predict_cls[0:9],
            filename="seq_model_predict.png")

# 错分类图片
correct = (predict_cls == data.y_test_cls)
plot_example_errors(cls_pred=predict_cls, correct=correct, filename="seq_model_errors.png")

# ------------------------------------------
# 功能模型 (Functional Model)
# ------------------------------------------
print("\n--- Training Functional Model ---")

inputs = Input(shape=(img_size_flat,))
net = inputs
net = Reshape(img_shape_full)(net)
net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Flatten()(net)
net = Dense(128, activation='relu')(net)
net = Dense(num_classes, activation='softmax')(net)
outputs = net

model2 = Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train_encoded, epochs=1, batch_size=128)

print("\n--- Saving and Loading Model ---")
path_model = 'model.keras'
model2.save(path_model)
del model2
model3 = load_model(path_model)

# ------------------------------------------
# 权重和输出的可视化
# ------------------------------------------
print("\n--- Visualization ---")


def plot_conv_weights(weights, input_channel=0, filename=None):
    w_min = np.min(weights)
    w_max = np.max(weights)
    num_filters = weights.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = weights[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

    if filename:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"图片已保存: {save_path}")
    plt.show()


# 获取层
layer_input = model3.layers[0]
layer_conv1 = model3.layers[2]
layer_conv2 = model3.layers[4]

weights_conv1 = layer_conv1.get_weights()[0]
plot_conv_weights(weights=weights_conv1, input_channel=0, filename="conv1_weights.png")

weights_conv2 = layer_conv2.get_weights()[0]
plot_conv_weights(weights=weights_conv2, input_channel=0, filename="conv2_weights.png")


def plot_conv_output(values, filename=None):
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])

    if filename:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"图片已保存: {save_path}")
    plt.show()


def plot_image(image, filename=None):
    plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
    if filename:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"图片已保存: {save_path}")
    plt.show()


# 可视化第一张测试图
image1 = data.x_test[0]
plot_image(image1, filename="test_image_0.png")

# 方法一：建立临时 Model 获取 Conv1 输出
print("Visualization Method 1: Layer Conv1 Output")
temp_model_1 = Model(inputs=model3.input, outputs=layer_conv1.output)
layer_output1 = temp_model_1.predict(np.array([x_test[0]]))
plot_conv_output(values=layer_output1, filename="conv1_output.png")

# 方法二：建立临时 Model 获取 Conv2 输出
print("Visualization Method 2: Layer Conv2 Output")
output_conv2_model = Model(inputs=model3.input, outputs=layer_conv2.output)
layer_output2 = output_conv2_model.predict(np.array([x_test[0]]))
plot_conv_output(values=layer_output2, filename="conv2_output.png")
