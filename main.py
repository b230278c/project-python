# main.py
import pandas as pd
from data_loader import load_data
from sentiment_analyzer import analyze_sentiment
from visualizer import plot_sentiment_distribution

def main():
    try:
        # Reading CSV
        df = load_data("sentimentdataset.csv")
        df.columns = df.columns.str.strip()

        # Handling missing and empty values ​​in Text columns
        df = df.dropna(subset=["Text"])
        df = df[df["Text"].str.strip() != ""]

        # Unified time format
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

        print("Data cleaning completed:")
        print(df.info())

        # Sentiment Analysis
        df = analyze_sentiment(df, text_column="Text")

        # Visualization: Sentiment Distribution
        plot_sentiment_distribution(df)

        # Save the results
        output_file = "sentiment_report.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Results saved to {output_file}")

    except FileNotFoundError:
        print("Error: Input file sentimentdataset.csv not found. Please confirm that the file path is correct.")
    except KeyError as e:
        print(f"Error: Required column missing - {e}")
    except Exception as e:
        print(f"An error occurred during runtime: {e}")

if __name__ == "__main__":
    main()
