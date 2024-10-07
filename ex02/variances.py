import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main() -> None:
    df = None
    try:
        df = pd.read_csv("Train_knight.csv")
    except Exception:
        print("Please run the script from the root of the project")
        exit(1)

    X = df.drop(columns=["knight"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    explained_variance = pca.explained_variance_ratio_ * 100
    print("Variances (Percentage):")
    print(explained_variance)

    cumulative_variance = np.cumsum(explained_variance)
    print("Cumulative Variances (Percentage):")
    print(cumulative_variance)

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(explained_variance) + 1),
        cumulative_variance,
    )

    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("Cumulative Explained Variance vs Number of Components")

    plt.show()


if __name__ == "__main__":
    main()
