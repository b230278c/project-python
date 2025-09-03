# sentiment_analyzer.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 初始化 VADER 分析器
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(df: pd.DataFrame, text_column: str = "Text") -> pd.DataFrame:
    """
    对 DataFrame 中指定列的文本进行情感分析
    返回带有 sentiment_score 和 sentiment_label 的 DataFrame
    """
    scores = []
    labels = []

    for text in df[text_column]:
        # 确保文本是字符串
        text = str(text)
        vs = analyzer.polarity_scores(text)
        score = vs['compound']  # 复合分数，范围 -1 ~ 1
        scores.append(score)

        # 根据分数判断情感类别
        if score >= 0.05:
            label = 'Positive'
        elif score <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
        labels.append(label)

    df['sentiment_score'] = scores
    df['sentiment_label'] = labels
    return df
