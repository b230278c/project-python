# sentiment_analyzer.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initializing the VADER analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(df: pd.DataFrame, text_column: str = "Text") -> pd.DataFrame:

    scores = []
    labels = []

    for text in df[text_column]:
        # Make sure the text is a string
        text = str(text)
        vs = analyzer.polarity_scores(text)
        score = vs['compound']  # Composite score, range -1 ~ 1
        scores.append(score)

        # Determine the emotional category based on the score
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
