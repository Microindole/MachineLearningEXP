import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# ================= 配置 =================
# 这里填你实验中设定的参数
TARGET_VOCAB_SIZE = 5000
TARGET_MAX_LEN = 100

DATA_PATH = os.path.join('data', 'review', 'drugsComTrain_raw.csv')
RESULT_DIR = 'data_analysis_report'
os.makedirs(RESULT_DIR, exist_ok=True)

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_probe():
    print(f"正在读取数据: {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        print("错误：找不到数据文件。")
        return

    df = pd.read_csv(DATA_PATH)
    total_samples = len(df)
    print(f"原始数据总行数: {total_samples}")

    # ================= 1. 长度分析 (多尺度梯度探测) =================
    print("\n[1/3]正在分析评论长度 (梯度探测)...")

    # 1. 基础计算
    # 将 review 列转为字符串并计算长度
    df['word_count'] = df['review'].astype(str).apply(lambda x: len(x.split()))
    total_words_all = df['word_count'].sum()

    # 2. 设定探测梯度 (从 50 到 300，步长 25，再加上 500)
    test_lengths = list(range(50, 301, 25)) + [400, 500]

    print(f"\n   {'截断长度':<10} | {'样本覆盖率':<12} | {'信息保留率':<12} | {'计算负载倍数'}")
    print("-" * 65)

    results_sample_cov = []
    results_info_ret = []

    base_load = 50 # 假设以 50 为基准计算量

    for length in test_lengths:
        # A. 样本覆盖率：有多少条评论长度完全 <= length
        covered_samples = len(df[df['word_count'] <= length])
        sample_cov = (covered_samples / total_samples) * 100

        # B. 信息保留率：截断后保留了多少单词总量
        # 使用 clip 函数模拟截断：超过 length 的变成 length
        retained_words = df['word_count'].clip(upper=length).sum()
        info_ret = (retained_words / total_words_all) * 100

        # C. 计算负载（相对值）：长度越长，LSTM计算量线性增加
        load_factor = length / 50.0

        print(f"   {length:<14} | {sample_cov:.2f}%       | {info_ret:.2f}%       | {load_factor:.1f}x")

        results_sample_cov.append(sample_cov)
        results_info_ret.append(info_ret)

    print("-" * 65)
    print(" [分析指导]：")
    print(" 1. '样本覆盖率'低不要紧，只要'信息保留率'高就行。")
    print(" 2. 寻找'拐点'：当长度增加，但信息保留率提升不明显时，就是最佳点。")

    # 3. 绘制双轴图表
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('截断长度 (MAX_LEN)')
    ax1.set_ylabel('样本覆盖率 (Sample Coverage %)', color=color)
    ax1.plot(test_lengths, results_sample_cov, color=color, marker='o', label='样本覆盖率')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
    color = 'tab:orange'
    ax2.set_ylabel('信息保留率 (Info Retention %)', color=color)
    ax2.plot(test_lengths, results_info_ret, color=color, marker='s', linestyle='--', label='信息保留率')
    ax2.tick_params(axis='y', labelcolor=color)

    # 标记当前的推荐值
    plt.axvline(x=140, color='green', linestyle=':', label='推荐值 (140)')

    plt.title('长度截断对信息量的影响 (收益 vs 成本)')
    fig.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'analysis_length_gradient.png'))
    print(f"    梯度分析图已保存: {RESULT_DIR}/analysis_length_gradient.png")

    # ================= 2. 词汇分析 (多尺度对比) =================
    print("\n[2/3]正在分析词汇覆盖率 (多尺度对比)...")

    # 统计所有单词
    all_text = " ".join(df['review'].astype(str).tolist()).lower()
    words = all_text.split()
    total_word_occurrences = len(words)
    word_counts = Counter(words)
    unique_words = len(word_counts)

    print(f" -> 原始词汇表大小: {unique_words}")

    # --- 核心修改：循环对比不同词表大小 ---
    test_sizes = [5000, 10000, 15000, 20000] # 我们探测这几个档位
    print(f"\n   {'词表大小':<10} | {'覆盖单词数':<12} | {'内容覆盖率':<10} | {'边际提升'}")
    print("-" * 55)

    prev_coverage = 0
    formatted_results = [] # 存起来画图用

    for size in test_sizes:
        most_common = word_counts.most_common(size)
        covered_count = sum(count for word, count in most_common)
        coverage = (covered_count / total_word_occurrences) * 100

        improvement = coverage - prev_coverage if prev_coverage > 0 else 0
        print(f"   {size:<14} | {covered_count:<16} | {coverage:.2f}%      | +{improvement:.2f}%")

        formatted_results.append((size, coverage))
        prev_coverage = coverage

    print("-" * 55)
    print(" [分析] 注意看‘边际提升’。如果提升很小（比如 <1%），说明增加词表纯属浪费资源。")

    # 绘制对比图
    sizes, coverages = zip(*formatted_results)
    plt.figure(figsize=(10, 6))

    # 画完整的曲线
    top_n = 25000
    counts = [count for word, count in word_counts.most_common(top_n)]
    cumulative_coverage = np.cumsum(counts) / total_word_occurrences * 100
    plt.plot(range(1, top_n + 1), cumulative_coverage, label='全量曲线', color='gray', alpha=0.5)

    # 标记我们测试的点
    plt.scatter(sizes, coverages, color='red', zorder=5)
    for s, c in zip(sizes, coverages):
        plt.annotate(f'{c:.1f}%', (s, c), xytext=(5, -10), textcoords='offset points')

    plt.axvline(x=TARGET_VOCAB_SIZE, color='green', linestyle='--', label=f'最终选择 ({TARGET_VOCAB_SIZE})')

    plt.title('不同词表大小的内容覆盖率对比')
    plt.xlabel('词表大小')
    plt.ylabel('覆盖率 (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULT_DIR, 'analysis_vocab_compare.png'))
    print(f"    对比图已保存: {RESULT_DIR}/analysis_vocab_compare.png")

    # ================= 3. 情感分布 (类别失衡) =================
    print("\n[3/3]正在分析情感标签分布...")
    def get_sentiment(rating):
        if rating <= 4: return 'Negative'
        if rating <= 6: return 'Neutral'
        return 'Positive'

    df['sentiment'] = df['rating'].apply(get_sentiment)
    sentiment_counts = df['sentiment'].value_counts()

    print(" -> 各类别样本数:")
    print(sentiment_counts)

    # 打印比例
    total = sum(sentiment_counts)
    print(f" -> Positive 占比: {sentiment_counts.get('Positive',0)/total*100:.1f}%")
    print(f" -> Negative 占比: {sentiment_counts.get('Negative',0)/total*100:.1f}%")
    print(f" -> Neutral  占比: {sentiment_counts.get('Neutral',0)/total*100:.1f}% (最难分类)")

    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#99ff99','#ffcc99'])
    plt.title('情感类别占比')
    plt.savefig(os.path.join(RESULT_DIR, 'analysis_labels.png'))
    print(f"    图表已保存: {RESULT_DIR}/analysis_labels.png")

    print(f"\n分析完成！报告图表已保存在 {RESULT_DIR} 文件夹中。")
    print("你可以直接把生成的数字和图片填进实验报告里。")

if __name__ == '__main__':
    run_probe()