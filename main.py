# main.py
import pandas as pd
from data_loader import load_data
from sentiment_analyzer import analyze_sentiment
from visualizer import plot_sentiment_distribution, plot_sentiment_over_time

def main():
    try:
        # 读取 CSV
        df = load_data("sentimentdataset.csv")
        df.columns = df.columns.str.strip()

        # 处理 Text 列缺失和空值
        df = df.dropna(subset=["Text"])
        df = df[df["Text"].str.strip() != ""]

        # 统一时间格式
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

        print("数据清理完成：")
        print(df.info())

        # 情感分析
        df = analyze_sentiment(df, text_column="Text")

        # 可视化：情感分布
        plot_sentiment_distribution(df)

        # 保存结果
        output_file = "sentiment_report.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"结果已保存到 {output_file}")

    except FileNotFoundError:
        print("错误：未找到输入文件 sentimentdataset.csv,请确认文件路径是否正确。")
    except KeyError as e:
        print(f"错误：缺少必要的列 - {e}")
    except Exception as e:
        print(f"运行时发生错误：{e}")

if __name__ == "__main__":
    main()
