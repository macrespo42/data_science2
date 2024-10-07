import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    df = None
    try:
        df = pd.read_csv("Train_knight.csv")
    except Exception:
        print("Please run the script from the root of the project")
        exit(1)
    df["knight"] = [1 if x == "Jedi" else 0 for x in df["knight"]]

    sns.heatmap(df.corr(numeric_only=True))
    plt.title("heatmap of knight features correlation")
    plt.show()


if __name__ == "__main__":
    main()
