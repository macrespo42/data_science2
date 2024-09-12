import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    df = pd.read_csv("Train_knight.csv")
    df["knight"] = [1 if x == "Jedi" else 0 for x in df["knight"]]

    sns.heatmap(df.corr(numeric_only=True))
    plt.title("heatmap of knight features correlation")
    plt.show()


if __name__ == "__main__":
    main()
