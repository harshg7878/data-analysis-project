import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def basic_stats(df, col):
    mean_val   = df[col].mean()
    median_val = df[col].median()
    std_val    = df[col].std()
    print(f"Statistics for column '{col}':")
    print(f"  Mean   = {mean_val:.2f}")
    print(f"  Median = {median_val:.2f}")
    print(f"  Std Dev= {std_val:.2f}")
    return mean_val, median_val, std_val

def plot_bar(df, cat_col, num_col, output_file=None):
    grouped = df.groupby(cat_col)[num_col].mean().reset_index()
    plt.figure(figsize=(8,5))
    sns.barplot(data=grouped, x=cat_col, y=num_col, palette="Blues_d")
    plt.xlabel(cat_col)
    plt.ylabel(f"Average of {num_col}")
    plt.title(f"Average {num_col} by {cat_col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print(f"Saved bar chart to {output_file}")
    else:
        plt.show()
    plt.close()

def plot_scatter(df, x_col, y_col, hue_col=None, output_file=None):
    plt.figure(figsize=(7,5))
    if hue_col:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.7)
    else:
        plt.scatter(df[x_col], df[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print(f"Saved scatter plot to {output_file}")
    else:
        plt.show()
    plt.close()

def plot_heatmap(df, output_file=None):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation matrix of numeric variables")
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print(f"Saved heatmap to {output_file}")
    else:
        plt.show()
    plt.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python data_analysis.py <csv_filepath>")
        sys.exit(1)
    filepath = sys.argv[1]
    df = load_data(filepath)
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nData info:")
    print(df.info())
    print("\nData description (numeric columns):")
    print(df.describe())

    numeric_col    = "TotalRevenue"
    category_col   = "ProductCategory"
    x_col          = "UnitsSold"
    y_col          = "TotalProfit"
    hue_col        = "Region"     
    
    basic_stats(df, numeric_col)
    plot_bar(df, category_col, numeric_col, output_file="bar_chart.png")
    plot_scatter(df, x_col, y_col, hue_col=hue_col, output_file="scatter_plot.png")
    plot_heatmap(df, output_file="heatmap.png")
    print("Plots saved to files.")

if __name__ == "__main__":
    main()
