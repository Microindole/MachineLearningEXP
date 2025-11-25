import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

if not os.path.exists('../model'):
    os.makedirs('../model')


# 数据预处理
def preprocess_data(train_path, test_path):
    print("正在加载和预处理数据 (Optimized LDA Mode)...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    def clean_df(df):
        # 去除字符串空格
        cat_columns = df.select_dtypes(include=['object']).columns
        for col in cat_columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        # 填充缺失值
        df.replace('?', np.nan, inplace=True)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        # 去除 fnlwgt 噪音
        if 'fnlwgt' in df.columns:
            df = df.drop('fnlwgt', axis=1)

        # Log变换
        for col in ['capital_gain', 'capital_loss']:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        # 5. 合并稀有类别
        if 'native_country' in df.columns:
            df['native_country'] = df['native_country'].apply(
                lambda x: 'United-States' if x == 'United-States' else 'Other')

        return df

    df_train = clean_df(df_train)
    df_test = clean_df(df_test)

    # 提取标签
    y_train = df_train['income'].apply(lambda x: 1 if x == '>50K' else 0).values
    X_train_raw = df_train.drop('income', axis=1)

    # 合并数据以统一 One-Hot 编码
    X_train_raw['is_train'] = 1
    df_test['is_train'] = 0
    full_data = pd.concat([X_train_raw, df_test], axis=0)

    # 去除 education
    if 'education' in full_data.columns:
        full_data = full_data.drop('education', axis=1)

    continuous_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    categorical_cols = [c for c in full_data.columns if c not in continuous_cols and c != 'is_train']

    # One-Hot Encoding
    full_data_encoded = pd.get_dummies(full_data, columns=categorical_cols)
    full_data_encoded = full_data_encoded.astype(float)

    # 获取特征名称
    feature_names = full_data_encoded.drop('is_train', axis=1).columns.tolist()

    # 标准化
    for col in continuous_cols:
        if col in full_data_encoded.columns:
            mean = full_data_encoded[col].mean()
            std = full_data_encoded[col].std()
            full_data_encoded[col] = (full_data_encoded[col] - mean) / (std + 1e-8)

    # 拆分回训练集和测试集
    X_train = full_data_encoded[full_data_encoded['is_train'] == 1].drop('is_train', axis=1).values
    X_test = full_data_encoded[full_data_encoded['is_train'] == 0].drop('is_train', axis=1).values

    return X_train, y_train, X_test, feature_names


# 概率生成模型
class ProbabilisticGenerativeModel:
    def __init__(self):
        self.w = None
        self.b = None
        self.mu1 = None
        self.mu2 = None
        self.sigma = None
        self.phi = None

    def fit(self, X, y):
        N = X.shape[0]
        D = X.shape[1]
        X1 = X[y == 1]
        X2 = X[y == 0]

        self.phi = X1.shape[0] / N
        self.mu1 = np.mean(X1, axis=0)
        self.mu2 = np.mean(X2, axis=0)

        # 共享协方差矩阵
        X1_centered = X1 - self.mu1
        X2_centered = X2 - self.mu2
        X_centered = np.vstack((X1_centered, X2_centered))

        self.sigma = np.dot(X_centered.T, X_centered) / N
        self.sigma += np.eye(D) * 1e-4

        inv_sigma = np.linalg.inv(self.sigma)

        # 计算权重 w 和 偏置 b
        self.w = np.dot(inv_sigma, self.mu1 - self.mu2)

        term1 = -0.5 * np.dot(np.dot(self.mu1.T, inv_sigma), self.mu1)
        term2 = 0.5 * np.dot(np.dot(self.mu2.T, inv_sigma), self.mu2)
        term3 = np.log(self.phi / (1 - self.phi))
        self.b = term1 + term2 + term3

    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return np.where(z > 0, 1, 0)

    def calculate_loss(self, X, y):
        p = self.predict_proba(X)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)


def plot_class_distribution(y_train):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_train, hue=y_train, palette='viridis', legend=False)
    plt.xticks([0, 1], ['<=50K', '>50K'])
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Income Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('class_distribution_bar.png')
    print("Class distribution bar chart saved.")


def plot_feature_importance(model, feature_names):
    if model.w is None:
        return

    weights = model.w
    fi_df = pd.DataFrame({'Feature': feature_names, 'Weight': weights})
    fi_df['AbsWeight'] = fi_df['Weight'].abs()

    top_features = fi_df.sort_values(by='AbsWeight', ascending=False).head(15)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Weight', y='Feature', hue='Feature', data=top_features, palette='coolwarm', legend=False)
    plt.title('Top 15 Feature Importance (LDA Weights)')
    plt.xlabel('Weight (Positive means correlates with >50K)')
    plt.tight_layout()
    plt.savefig('feature_importance_bar.png')
    print("Feature importance bar chart saved.")


def plot_probability_histogram(model, X_val, y_val):
    probs = model.predict_proba(X_val)

    plt.figure(figsize=(8, 6))
    plt.hist(probs[y_val == 0], bins=50, alpha=0.5, label='<=50K', color='blue')
    plt.hist(probs[y_val == 1], bins=50, alpha=0.5, label='>50K', color='red')
    plt.title('Distribution of Predicted Probabilities (>50K)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('probability_histogram.png')
    print("Probability histogram saved.")


def plot_scatter_pca(X_val, y_val):
    # PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_val)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_val, cmap='coolwarm', alpha=0.5, s=10)
    plt.colorbar(scatter, ticks=[0, 1], label='Income Class (0: <=50K, 1: >50K)')
    plt.title('Data Distribution (PCA Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig('scatter_plot_pca.png')
    print("Scatter plot (PCA) saved.")


if __name__ == "__main__":
    X_train, y_train, X_test, feature_names = preprocess_data('../data/train.csv', '../data/test.csv')

    # 划分验证集
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    X_sub_train, y_sub_train = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # 类别分布柱状图
    plot_class_distribution(y_train)

    # 训练模型
    print("\nTraining Model...")
    model = ProbabilisticGenerativeModel()
    model.fit(X_sub_train, y_sub_train)

    # 验证
    acc = np.mean(model.predict(X_val) == y_val)
    print(f"Validation Accuracy: {acc * 100:.2f}%")

    # 特征重要性柱状图
    plot_feature_importance(model, feature_names)

    # 概率直方图
    plot_probability_histogram(model, X_val, y_val)

    # 散点图
    plot_scatter_pca(X_val, y_val)

    # 混淆矩阵
    cm = confusion_matrix(y_val, model.predict(X_val))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # 生成 Submission
    print("\nGenerating Submission...")
    model_full = ProbabilisticGenerativeModel()
    model_full.fit(X_train, y_train)

    model_full.save('../model/generative_model.pkl')
    print("Model saved to model/generative_model.pkl")

    test_preds = model_full.predict(X_test)

    submission = pd.DataFrame({
        'id': range(1, len(test_preds) + 1),
        'label': test_preds
    })
    submission.to_csv('../data/submission.csv', index=False)
    print("Done.")
