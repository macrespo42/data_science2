import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def preprocess_data(training, validation):
    features = ["Push", "Lightsaber", "Friendship", "Attunement"]

    X_train = training[features]
    X_test = validation[features]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    y_train = [0 if x == "Jedi" else 1 for x in training["knight"]]
    y_test = [0 if x == "Jedi" else 1 for x in validation["knight"]]

    return X_train, X_test, y_train, y_test


def plot_knn_accuracy(k_values, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker="o", linestyle="--")
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs K values")
    plt.grid(True)
    plt.show()


def main() -> None:
    training = pd.read_csv("ex04/Training_data.csv")
    validation = pd.read_csv("ex04/Validation_data.csv")

    X_train, X_test, y_train, y_test = preprocess_data(training, validation)

    cls = LogisticRegression(random_state=42)

    model = cls.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    print(f"f1_score: {f1 * 100}%")

    df = pd.read_csv("Train_knight.csv")
    features = ["Push", "Lightsaber", "Friendship", "Attunement"]
    X_train = df[features]
    y_train = [0 if x == "Jedi" else 1 for x in df["knight"]]

    model = cls.fit(X_train, y_train)

    test = pd.read_csv("Test_knight.csv")
    X_test = test[features]
    y_pred = model.predict(X_test)

    with open("Voting.txt", "w+") as f:
        for pred in y_pred:
            if pred == 0:
                f.write("Jedi\n")
            else:
                f.write("Sith\n")


if __name__ == "__main__":
    main()
