import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    train = pd.read_csv("Train_knight.csv")
    X = train[["Push", "Lightsaber", "Friendship", "Attunement"]]
    y = [1 if x == "Jedi" else 0 for x in train["knight"]]

    dtc = DecisionTreeClassifier(random_state=42)
    model = dtc.fit(X, y)

    x_test = pd.read_csv("Test_knight.csv")
    y_pred = model.predict(x_test[["Push", "Lightsaber", "Friendship", "Attunement"]])

    with open("preds.txt", "w+") as f:
        for pred in y_pred:
            if pred == 1:
                f.write("Jedi\n")
            else:
                f.write("Sith\n")

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(
        dtc, feature_names=X.columns, class_names=["Jedi", "Sith"], filled=True
    )
    fig.savefig("./tree.png")


if __name__ == "__main__":
    main()
