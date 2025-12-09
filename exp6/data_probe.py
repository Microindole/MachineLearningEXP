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

    # ================= 1. 长度分析 (MAX_LEN 合理性) =================
    print("\n[1/3]正在分析评论长度...")
    # 简单分词计算长度
    df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

    # 计算有多少样本在 MAX_LEN 范围内
    covered_samples = len(df[df['word_count'] <= TARGET_MAX_LEN])
    coverage_rate = (covered_samples / total_samples) * 100

    # 计算统计指标
    avg_len = df['word_count'].mean()
    median_len = df['word_count'].median()
    percentile_90 = np.percentile(df['word_count'], 90)

    print(f" -> 平均长度: {avg_len:.1f} 词")
    print(f" -> 中位数长度: {median_len:.1f} 词")
    print(f" -> 90% 的评论都在 {percentile_90:.1f} 词以内")
    print(f" -> 设定 MAX_LEN={TARGET_MAX_LEN} 时，完整覆盖了 {coverage_rate:.2f}% 的评论")

    if coverage_rate > 80:
        print("    [结论] ✅ 长度截断合理，绝大多数信息被保留。")
    else:
        print("    [警告] ⚠️ 可能截断了过多信息，建议适当增加 MAX_LEN。")

    # 绘制长度分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=100, kde=True)
    plt.axvline(x=TARGET_MAX_LEN, color='r', linestyle='--', label=f'当前设定 ({TARGET_MAX_LEN})')
    plt.axvline(x=percentile_90, color='g', linestyle=':', label=f'90%分位点 ({int(percentile_90)})')
    plt.title(f'评论长度分布与截断位置 (覆盖率: {coverage_rate:.1f}%)')
    plt.xlabel('单词数量')
    plt.legend()
    plt.xlim(0, 500)
    plt.savefig(os.path.join(RESULT_DIR, 'analysis_length.png'))
    print(f"    图表已保存: {RESULT_DIR}/analysis_length.png")

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