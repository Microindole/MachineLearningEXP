import csv
import numpy as np
import pandas as pd
import math
import os
import sys

class Logger(object):
    def __init__(self, filename='train.log'):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger('train.log')

# 路径配置
TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/test.csv'
PREDICT_FILE = '../data/predict.csv'
MODEL_DIR = '../model'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 超参数
LEARNING_RATE = 0.05
ITERATIONS = 300000
VALIDATION_SPLIT = 0.2


# ==========================================
# 数据处理
# ==========================================
def load_train_data(filename):
    print(f"正在读取数据: {filename} ...")
    try:
        df = pd.read_csv(filename, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filename, encoding='gbk')
        except:
            df = pd.read_csv(filename, encoding='big5')

    df.replace('NR', 0, inplace=True)
    raw_data = df.iloc[:, 3:].to_numpy()

    data = {}
    for i in range(18):
        data[i] = []

    for i in range(raw_data.shape[0]):
        row_values = raw_data[i, :].astype(float)
        row_values[row_values < 0] = 0
        data[i % 18].extend(row_values)

    full_data = np.array([data[i] for i in range(18)])
    return full_data


def extract_features(data):
    x_list = []
    y_list = []
    hours_per_month = 480
    for month in range(12):
        month_data = data[:, month * hours_per_month: (month + 1) * hours_per_month]
        for i in range(hours_per_month - 9):
            features = month_data[:, i: i + 9].flatten()
            target = month_data[9, i + 9]
            x_list.append(features)
            y_list.append(target)
    return np.array(x_list), np.array(y_list)


def normalize(x, mean=None, std=None):
    if mean is None:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std[std == 0] = 1
    x_norm = (x - mean) / std
    return x_norm, mean, std


def add_bias(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)


# ==========================================
# 模型训练
# ==========================================
def train(x, y, lr, iter_num):
    dim = x.shape[1]
    w = np.zeros(dim)
    s_grad = np.zeros(dim)

    print(f"开始训练 (样本数: {len(x)})...")
    for i in range(iter_num):
        y_pred = np.dot(x, w)
        loss = y_pred - y
        gradient = 2 * np.dot(x.T, loss)
        s_grad += gradient ** 2
        ada = np.sqrt(s_grad)
        w = w - lr * gradient / (ada + 1e-8)

        if i % 50000 == 0:
            print(f"  Iter {i}: RMSE = {math.sqrt(np.mean(loss ** 2)):.4f}")

    return w


if __name__ == "__main__":
    # 准备数据
    full_data = load_train_data(TRAIN_FILE)
    x_all, y_all = extract_features(full_data)

    # 验证集切分
    np.random.seed(42)
    indices = np.arange(len(x_all))
    np.random.shuffle(indices)
    split_idx = int(len(x_all) * (1 - VALIDATION_SPLIT))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    # 验证流程
    x_train_norm, mean_train, std_train = normalize(x_train)
    x_val_norm, _, _ = normalize(x_val, mean_train, std_train)
    w = train(add_bias(x_train_norm), y_train, LEARNING_RATE, ITERATIONS)

    y_val_pred = np.dot(add_bias(x_val_norm), w)
    val_loss = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
    print(f"\n=== 验证集 RMSE: {val_loss:.4f} ===\n")

    # 全量训练与保存
    print("正在使用全量数据重新训练最终模型...")
    x_all_norm, mean_all, std_all = normalize(x_all)
    w_final = train(add_bias(x_all_norm), y_all, LEARNING_RATE, ITERATIONS)

    # 保存模型参数
    print(f"正在保存模型至 {MODEL_DIR} ...")
    np.save(os.path.join(MODEL_DIR, 'model.npy'), w_final)
    np.save(os.path.join(MODEL_DIR, 'mean.npy'), mean_all)
    np.save(os.path.join(MODEL_DIR, 'std.npy'), std_all)
    print("模型保存完成！")

    # 预测
    if os.path.exists(TEST_FILE):
        print("\n正在生成最终 predict.csv ...")
        df_test = pd.read_csv(TEST_FILE, header=None)
        df_test.replace('NR', 0, inplace=True)
        raw_test = df_test.iloc[:, 2:].to_numpy().astype(float)
        raw_test[raw_test < 0] = 0

        x_test = []
        for i in range(len(raw_test) // 18):
            sample = raw_test[i * 18: (i + 1) * 18, :].flatten()
            x_test.append(sample)
        x_test = np.array(x_test)

        # 必须用全量数据的 mean/std
        x_test_norm = (x_test - mean_all) / std_all
        y_pred_final = np.dot(add_bias(x_test_norm), w_final)

        with open(PREDICT_FILE, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'value'])
            for i, pred in enumerate(y_pred_final):
                writer.writerow([f'id_{i}', pred])
        print(f"预测完成，结果已保存至 {PREDICT_FILE}")
