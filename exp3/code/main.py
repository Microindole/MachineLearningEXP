import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 数据预处理
def load_and_process_data(train_path, test_path):
    print(f"正在读取数据: {train_path}, {test_path} ...")
    df_train = pd.read_csv(train_path, skipinitialspace=True, na_values='?')
    df_test = pd.read_csv(test_path, skipinitialspace=True, na_values='?')

    # 提取标签
    Y_train = (df_train['income'] == '>50K').astype(int).values

    # 构建特征矩阵
    X_train_raw = df_train.drop('income', axis=1)
    X_test_raw = df_test
    X_train_raw['dataset_type'] = 'train'
    X_test_raw['dataset_type'] = 'test'
    full_data = pd.concat([X_train_raw, X_test_raw], axis=0)
    continuous_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    categorical_cols = [col for col in full_data.columns if col not in continuous_cols and col != 'dataset_type']

    # One-Hot 编码
    full_data_encoded = pd.get_dummies(full_data, columns=categorical_cols, dummy_na=True)
    feature_names = full_data_encoded.drop('dataset_type', axis=1).columns.tolist()

    # 拆分回训练集和测试集
    X_train = full_data_encoded[full_data_encoded['dataset_type'] == 'train'].drop('dataset_type', axis=1).values
    X_test = full_data_encoded[full_data_encoded['dataset_type'] == 'test'].drop('dataset_type', axis=1).values

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    continuous_indices = [full_data_encoded.columns.get_loc(c) for c in continuous_cols if
                          c in full_data_encoded.columns]

    train_mean = np.mean(X_train[:, continuous_indices], axis=0)
    train_std = np.std(X_train[:, continuous_indices], axis=0) + 1e-8  # 防止除零

    X_train[:, continuous_indices] = (X_train[:, continuous_indices] - train_mean) / train_std
    X_test[:, continuous_indices] = (X_test[:, continuous_indices] - train_mean) / train_std

    return X_train, Y_train, X_test, feature_names

# 逻辑回归
def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)

def get_prob(X, w, b):
    return _sigmoid(np.matmul(X, w) + b)

def infer(X, w, b):
    # 概率 > 0.5 输出 1
    return (get_prob(X, w, b) > 0.5).astype(int)

def _cross_entropy_loss(y_pred, y_true, w, lamda=0):
    # 交叉熵损失函数 + L2 正则化
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if lamda > 0:
        loss += lamda * np.sum(np.square(w))
    return loss

def _gradient(X, y_true, y_pred, w, lamda=0):
    m = len(y_true)
    error = y_pred - y_true
    w_grad = np.matmul(X.T, error) / m
    if lamda > 0:
        w_grad += 2 * lamda * w
    b_grad = np.mean(error)
    return w_grad, b_grad

def accuracy(Y_pred, Y_true):
    return np.mean(Y_pred == Y_true)

# 交叉验证训练
def train_k_fold(X, Y, k=5, learning_rate=0.2, epochs=1000, lamda=0):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)  # 打乱顺序

    all_scores = []
    best_acc = 0
    best_model = (None, None)
    best_history = None
    best_X_val = None
    best_Y_val = None

    print(f"开始 {k} 折交叉验证 (Learning Rate={learning_rate}, Epochs={epochs})...")

    for i in range(k):
        # 划分训练集和验证集
        val_idx = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        n_features = X.shape[1]
        w = np.zeros(n_features)  # 初始化权重
        b = 0.0

        train_loss_hist = []
        val_loss_hist = []
        train_acc_hist = []
        val_acc_hist = []

        for epoch in range(epochs):
            # Forward
            y_pred = get_prob(X_tr, w, b)

            # Loss & Gradient
            w_grad, b_grad = _gradient(X_tr, Y_tr, y_pred, w, lamda)

            # Update
            w -= learning_rate * w_grad
            b -= learning_rate * b_grad

            # 记录过程 (每50轮记录一次)
            if epoch % 50 == 0:
                y_val_pred_prob = get_prob(X_val, w, b)

                loss_tr = _cross_entropy_loss(y_pred, Y_tr, w, lamda)
                loss_val = _cross_entropy_loss(y_val_pred_prob, Y_val, w, lamda)

                acc_tr = accuracy(infer(X_tr, w, b), Y_tr)
                acc_val = accuracy(infer(X_val, w, b), Y_val)

                train_loss_hist.append(loss_tr)
                val_loss_hist.append(loss_val)
                train_acc_hist.append(acc_tr)
                val_acc_hist.append(acc_val)

        final_val_acc = val_acc_hist[-1]
        print(f"Fold {i + 1}/{k} - Final Val Acc: {final_val_acc:.4f}")
        all_scores.append(final_val_acc)

        # 保存最佳模型
        if final_val_acc > best_acc:
            best_acc = final_val_acc
            best_model = (w, b)
            best_history = (train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist)
            best_X_val = X_val
            best_Y_val = Y_val

    print(f"\n{k} 折平均准确率: {np.mean(all_scores):.4f}")
    return best_model, best_history, best_X_val, best_Y_val

# 绘图
def save_plots(history, X_val, Y_val, w, b, feature_names):
    t_loss, v_loss, t_acc, v_acc = history

    # 保存 Loss 和 Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(t_loss, label='Train Loss', color='blue', alpha=0.6)
    plt.plot(v_loss, label='Val Loss', color='orange', linestyle='--')
    plt.plot(t_acc, label='Train Acc', color='green', alpha=0.6)
    plt.plot(v_acc, label='Val Acc', color='red', linestyle='--')
    plt.title('Training Loss & Accuracy Curve')
    plt.xlabel('Steps (x50)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_acc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'loss_acc_curve.png'")

    # 保存混淆矩阵
    plt.figure(figsize=(6, 5))
    Y_pred = infer(X_val, w, b)
    tp = np.sum((Y_pred == 1) & (Y_val == 1))
    tn = np.sum((Y_pred == 0) & (Y_val == 0))
    fp = np.sum((Y_pred == 1) & (Y_val == 0))
    fn = np.sum((Y_pred == 0) & (Y_val == 1))

    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'confusion_matrix.png'")

    # 保存特征重要性图
    plt.figure(figsize=(10, 6))
    indices = np.argsort(np.abs(w))[-15:]  # 前15个重要特征
    plt.barh(range(len(indices)), w[indices], align='center', color='purple', alpha=0.7)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('Top 15 Feature Weights')
    plt.xlabel('Weight Value')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'feature_importance.png'")

# 主程序
if __name__ == "__main__":
    train_file = "../data/train.csv"
    test_file = "../data/test.csv"
    if not os.path.exists(train_file):
        print(f"Error: 找不到文件 {train_file}，请确保数据文件在当前目录下。")
        exit()

    print("Loading Data...")
    X_train, Y_train, X_test, feature_names = load_and_process_data(train_file, test_file)

    # 训练模型
    (w_opt, b_opt), best_history, X_val_best, Y_val_best = train_k_fold(
        X_train, Y_train, k=5, learning_rate=0.1, epochs=1000, lamda=0.001
    )
    save_plots(best_history, X_val_best, Y_val_best, w_opt, b_opt, feature_names)

    # 预测并保存结果 CSV
    print("Generating predictions...")
    preds = infer(X_test, w_opt, b_opt)
    pd.DataFrame({'id': range(1, len(preds) + 1), 'label': preds}).to_csv('output.csv', index=False)
    print("Predictions saved to output.csv")

    # 保存模型参数
    if not os.path.exists('../model'):
        os.makedirs('../model')
    np.savez('../model/logistic_model.npz', w=w_opt, b=b_opt)
    print("Model saved to model/logistic_model.npz")
