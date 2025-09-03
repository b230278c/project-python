# visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df: pd.DataFrame):
    """
    绘制情感类别的分布和情感分数的分布
    """
    # 设置 Seaborn 样式
    sns.set(style="whitegrid")

    # 情感类别分布
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='sentiment_label', palette='pastel')
    plt.title("Sentiment Label Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 情感分数分布
    plt.figure(figsize=(6,4))
    sns.histplot(df['sentiment_score'], bins=20, kde=True, color='skyblue')
    plt.title("Sentiment Score Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 按日期分组，取平均情感分数
    df['date'] = df['Timestamp'].dt.date
    trend = df.groupby('date')['sentiment_score'].mean().reset_index()

    plt.figure(figsize=(10,5))
    sns.lineplot(data=trend, x='date', y='sentiment_score', marker="o")
    plt.title("Average Sentiment Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    