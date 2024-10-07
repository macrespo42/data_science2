import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def plot_knn_accuracy(k_values, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker="o", linestyle="--")
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs K values")
    plt.grid(True)
    plt.show()


def main() -> None:
    if len(sys.argv) != 3:
        print("provide 2 arguments: truth and predictions files")
        sys.exit(1)
    training = sys.argv[1]
    validation = sys.argv[2]

    try:
        training_df = pd.read_csv(training)
        validation_df = pd.read_csv(validation)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    X = training_df.drop("knight", axis=1)
    y = training_df["knight"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=6)

    model = knn.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    f1 = f1_score(y_test.values, y_pred, pos_label='Sith')
    print(f"f1_score: {f1 * 100}%")

    k_values = range(1, 31)
    accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)

        model = knn.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))

    plot_knn_accuracy(k_values, accuracies)

    with open("KNN.txt", "w+") as f:
        for pred in y_pred:
            f.write(f"{pred}\n")


if __name__ == "__main__":
    main()
