import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    train = pd.read_csv("Train_knight.csv")
    train["knight"] = [1 if x == "Jedi" else 0 for x in train["knight"]]
    y = train["knight"]
    X = train.drop(
        [
            "knight",
            "Sensitivity",
            "Recovery",
            "Strength",
            "Stims",
            "Midi-chlorien",
            "Agility",
            "Evade",
            "Grasping",
            "Burst",
            "Awareness",
            "Empowered",
            "Slash",
            "Combo",
            "Dexterity",
            "Power",
            "Delay",
            "Pull",
            "Reactivity",
            "Blocking",
            "Repulse",
            "Sprint",
            "Deflection",
            "Mass",
            "Survival",
        ],
        axis=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    dtc = DecisionTreeClassifier(random_state=42)
    model = dtc.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"f1_score: {f1 * 100}%")

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
