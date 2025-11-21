import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.metrics import confusion_matrix

# 确保模型保存目录存在
if not os.path.exists('../model'):
    os.makedirs('../model')


# ================= 1. 数据预处理 =================
def preprocess_data(train_path, test_path):
    print("正在加载和预处理数据...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    def clean_df(df):
        cat_columns = df.select_dtypes(include=['object']).columns
        for col in cat_columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        df.replace('?', np.nan, inplace=True)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        # 优化：去除 fnlwgt
        if 'fnlwgt' in df.columns:
            df = df.drop('fnlwgt', axis=1)

        # 优化：对数变换，让数据分布更接近高斯分布 (实验核心目的)
        for col in ['capital_gain', 'capital_loss']:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        return df

    df_train = clean_df(df_train)
    df_test = clean_df(df_test)

    y_train = df_train['income'].apply(lambda x: 1 if x == '>50K' else 0).values
    X_train_raw = df_train.drop('income', axis=1)

    # 对齐特征
    X_train_raw['is_train'] = 1
    df_test['is_train'] = 0
    full_data = pd.concat([X_train_raw, df_test], axis=0)

    if 'education' in full_data.columns:
        full_data = full_data.drop('education', axis=1)

    continuous_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    categorical_cols = [c for c in full_data.columns if c not in continuous_cols and c != 'is_train']

    full_data_encoded = pd.get_dummies(full_data, columns=categorical_cols)
    full_data_encoded = full_data_encoded.astype(float)

    # 标准化
    for col in continuous_cols:
        if col in full_data_encoded.columns:
            mean = full_data_encoded[col].mean()
            std = full_data_encoded[col].std()
            full_data_encoded[col] = (full_data_encoded[col] - mean) / (std + 1e-8)

    X_train = full_data_encoded[full_data_encoded['is_train'] == 1].drop('is_train', axis=1).values
    X_test = full_data_encoded[full_data_encoded['is_train'] == 0].drop('is_train', axis=1).values

    return X_train, y_train, X_test


# ================= 2. 概率生成模型类 (含 Loss 计算) =================
class ProbabilisticGenerativeModel:
    def __init__(self):
        self.w = None
        self.b = None
        self.mu1 = None
        self.mu2 = None
        self.sigma = None
        self.phi = None

    def fit(self, X, y):
        # 1. 统计高斯分布参数 (mu, sigma)
        N = X.shape[0]
        D = X.shape[1]
        X1 = X[y == 1]
        X2 = X[y == 0]

        self.phi = X1.shape[0] / N
        self.mu1 = np.mean(X1, axis=0)
        self.mu2 = np.mean(X2, axis=0)

        # 共享协方差矩阵 (LDA)
        X1_centered = X1 - self.mu1
        X2_centered = X2 - self.mu2
        X_centered = np.vstack((X1_centered, X2_centered))

        self.sigma = np.dot(X_centered.T, X_centered) / N
        self.sigma += np.eye(D) * 1e-5  # 防止奇异矩阵

        inv_sigma = np.linalg.inv(self.sigma)

        # 2. 计算权重 w 和 偏置 b (Closed-form solution)
        self.w = np.dot(inv_sigma, self.mu1 - self.mu2)
        term1 = -0.5 * np.dot(np.dot(self.mu1.T, inv_sigma), self.mu1)
        term2 = 0.5 * np.dot(np.dot(self.mu2.T, inv_sigma), self.mu2)
        term3 = np.log(self.phi / (1 - self.phi))
        self.b = term1 + term2 + term3

    def predict_proba(self, X):
        # 计算 Sigmoid 概率 P(y=1|x) = 1 / (1 + exp(-z))
        z = np.dot(X, self.w) + self.b
        # 限制 z 的范围防止 exp 溢出
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        # 如果 z > 0 (即 P > 0.5), 预测为 1
        z = np.dot(X, self.w) + self.b
        return np.where(z > 0, 1, 0)

    def calculate_loss(self, X, y):
        # 计算交叉熵损失 (Cross Entropy Loss)
        # Loss = -1/N * sum( y*log(p) + (1-y)*log(1-p) )
        y_pred_proba = self.predict_proba(X)
        epsilon = 1e-15  # 防止 log(0)
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba))
        return loss

    def save(self, filepath):
        params = {'w': self.w, 'b': self.b, 'mu1': self.mu1, 'mu2': self.mu2, 'sigma': self.sigma}
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)


# ================= 3. 主程序 =================
if __name__ == "__main__":
    X_train, y_train, X_test = preprocess_data('../data/train.csv', '../data/test.csv')

    # 划分验证集
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    X_sub_train, y_sub_train = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # --- A. 学习曲线 (带 Loss 打印) ---
    print("\n开始模拟训练过程 (Learning Curve)...")
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_accs = []
    val_accs = []

    print(f"{'Data %':<10} | {'Train Acc':<12} | {'Val Acc':<12} | {'Train Loss':<12} | {'Val Loss':<12}")
    print("-" * 70)

    for frac in train_sizes:
        size = int(frac * X_sub_train.shape[0])
        X_part = X_sub_train[:size]
        y_part = y_sub_train[:size]

        temp_model = ProbabilisticGenerativeModel()
        temp_model.fit(X_part, y_part)

        # Acc
        t_acc = np.mean(temp_model.predict(X_part) == y_part)
        v_acc = np.mean(temp_model.predict(X_val) == y_val)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        # Loss (现在有了!)
        t_loss = temp_model.calculate_loss(X_part, y_part)
        v_loss = temp_model.calculate_loss(X_val, y_val)

        print(f"{frac * 100:5.0f}%     | {t_acc:.4f}       | {v_acc:.4f}       | {t_loss:.4f}       | {v_loss:.4f}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes * 100, train_accs, 'o-', label='Training Accuracy')
    plt.plot(train_sizes * 100, val_accs, 'o-', label='Validation Accuracy')
    plt.xlabel('Training Data Percentage (%)')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve: Probabilistic Generative Model')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    print("\n学习曲线已保存为 learning_curve.png")

    # --- B. 训练最终模型 ---
    print("\n正在全量数据上训练最终模型...")
    model = ProbabilisticGenerativeModel()
    model.fit(X_sub_train, y_sub_train)
    model.save('../model/generative_model.pkl')

    final_train_loss = model.calculate_loss(X_sub_train, y_sub_train)
    final_val_loss = model.calculate_loss(X_val, y_val)
    print(f"最终训练集 Loss: {final_train_loss:.4f}")
    print(f"最终验证集 Loss: {final_val_loss:.4f}")

    # --- C. 混淆矩阵 ---
    y_pred_val = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # --- D. 预测并生成 Submission (修正ID从1开始) ---
    print("\n正在生成 submission.csv ...")
    model_full = ProbabilisticGenerativeModel()
    model_full.fit(X_train, y_train)
    test_preds = model_full.predict(X_test)

    submission = pd.DataFrame({
        'id': range(1, len(test_preds) + 1),  # 修正: ID 从 1 开始
        'label': test_preds
    })
    submission.to_csv('submission.csv', index=False)
    print(f"完成。文件已保存，共 {len(submission)} 行。")
    print("前5行预览:")
    print(submission.head())
