import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


def main() -> None:
    df = pd.read_csv("Train_knight.csv")

    features = ["Push", "Lightsaber", "Friendship", "Attunement"]
    X_train = df[features]
    y_train = [0 if x == "Jedi" else 1 for x in df["knight"]]

    knn = KNeighborsClassifier(n_neighbors=3)
    model = knn.fit(X_train, y_train)

    test = pd.read_csv("Test_knight.csv")
    X_test = test[features]
    y_pred = model.predict(X_test)

    with open("KNN.txt", "w+") as f:
        for pred in y_pred:
            if pred == 0:
                f.write("Jedi\n")
            else:
                f.write("Sith\n")


if __name__ == "__main__":
    main()
