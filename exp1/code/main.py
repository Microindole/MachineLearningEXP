import csv
import numpy as np
import pandas as pd
import math
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # 用于更复杂的排版


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
IMG_DIR = '../images'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# 超参数
LEARNING_RATE = 0.05
ITERATIONS = 300000
VALIDATION_SPLIT = 0.2

# 污染物特征列表
FEATURE_NAMES = [
    "AMB_TEMP", "CH4", "CO", "NHMC", "NO", "NO2", "NOx", "O3",
    "PM10", "PM2.5", "RAINFALL", "RH", "SO2", "THC",
    "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR"
]


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


def plot_all_analysis(loss_history, y_true, y_pred, weights, filename='experiment_report_plots.png'):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(loss_history, color='#1f77b4', linewidth=2)
    ax1.set_title(f'Training Loss Curve (Final RMSE: {loss_history[-1]:.4f})', fontsize=14)
    ax1.set_xlabel('Iterations (x100)')
    ax1.set_ylabel('RMSE Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = plt.subplot(gs[0, 1])
    ax2.scatter(y_true, y_pred, alpha=0.5, s=10, color='#2ca02c', label='Data Points')

    limit_max = max(y_true.max(), y_pred.max())
    ax2.plot([0, limit_max], [0, limit_max], 'r--', linewidth=2, label='Ideal Prediction (y=x)')
    ax2.set_title('Regression Analysis: True vs Predicted', fontsize=14)
    ax2.set_xlabel('True PM2.5')
    ax2.set_ylabel('Predicted PM2.5')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    ax3 = plt.subplot(gs[1, 0])
    subset_n = 80
    indices = range(subset_n)
    ax3.plot(indices, y_true[:subset_n], 'b-o', label='True Value', markersize=4, alpha=0.7)
    ax3.plot(indices, y_pred[:subset_n], 'r--x', label='Prediction', markersize=4, alpha=0.7)
    ax3.set_title(f'Detailed Comparison (First {subset_n} Samples)', fontsize=14)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('PM2.5 Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 权重热力图
    ax4 = plt.subplot(gs[1, 1])
    w_matrix = weights[1:].reshape(18, 9)

    # 归一化权重以便绘图更清晰
    w_visual = np.abs(w_matrix)

    im = ax4.imshow(w_visual, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax4, label='Weight Magnitude (Importance)')

    ax4.set_title('Model Feature Importance (Weights Heatmap)', fontsize=14)
    ax4.set_yticks(range(18))
    ax4.set_yticklabels(FEATURE_NAMES, fontsize=8)
    ax4.set_xticks(range(9))
    ax4.set_xticklabels([f'H-{9 - i}' for i in range(9)])
    ax4.set_xlabel('Hours Before Prediction')

    plt.tight_layout()
    save_path = os.path.join(IMG_DIR, filename)
    plt.savefig(save_path, dpi=150)
    print(f"综合分析图表已保存: {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, filename='residuals.png'):
    """
    单独生成残差分布图
    """
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.title('Residual Distribution (Error Histogram)')
    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(IMG_DIR, filename)
    plt.savefig(save_path)
    print(f"残差分析图已保存: {save_path}")
    plt.close()


# ==========================================
# 模型训练
# ==========================================
def train(x, y, lr, iter_num):
    dim = x.shape[1]
    w = np.zeros(dim)
    s_grad = np.zeros(dim)
    loss_history = []

    print(f"开始训练 (样本数: {len(x)})...")
    for i in range(iter_num):
        y_pred = np.dot(x, w)
        loss = y_pred - y
        gradient = 2 * np.dot(x.T, loss)
        s_grad += gradient ** 2
        ada = np.sqrt(s_grad)
        w = w - lr * gradient / (ada + 1e-8)

        current_rmse = math.sqrt(np.mean(loss ** 2))

        if i % 100 == 0:
            loss_history.append(current_rmse)

        if i % 1000 == 0:
            print(f"  Iter {i}: RMSE = {current_rmse:.4f}")

    return w, loss_history


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

    # 训练
    w, loss_hist = train(add_bias(x_train_norm), y_train, LEARNING_RATE, ITERATIONS)

    # 验证集预测
    y_val_pred = np.dot(add_bias(x_val_norm), w)
    val_loss = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
    print(f"\n=== 验证集 RMSE: {val_loss:.4f} ===\n")

    print("正在生成高级分析图表...")
    # 图1: 综合面板 (包含 Loss, 回归分析, 细节对比, 权重热力图)
    plot_all_analysis(loss_hist, y_val, y_val_pred, w, filename='Result_Analysis_Dashboard.png')
    # 图2: 残差分析 (单独一张，适合放在“结果分析”中讨论误差分布)
    plot_residuals(y_val, y_val_pred, filename='Result_Residuals.png')

    # 全量训练与保存
    print("正在使用全量数据重新训练最终模型...")
    x_all_norm, mean_all, std_all = normalize(x_all)
    w_final, _ = train(add_bias(x_all_norm), y_all, LEARNING_RATE, ITERATIONS)

    # 保存模型
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

        x_test_norm = (x_test - mean_all) / std_all
        y_pred_final = np.dot(add_bias(x_test_norm), w_final)

        with open(PREDICT_FILE, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'value'])
            for i, pred in enumerate(y_pred_final):
                writer.writerow([f'id_{i}', pred])
        print(f"预测完成，结果已保存至 {PREDICT_FILE}")
