import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    dtc = DecisionTreeClassifier(random_state=42)
    model = dtc.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    f1 = f1_score(y_test.values, y_pred, pos_label='Jedi')

    print(f"f1_score: {f1 * 100}%")

    with open("Tree.txt", "w+") as f:
        for pred in y_pred:
            f.write(f"{pred}\n")

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(
        dtc, class_names=["Jedi", "Sith"], filled=True
    )
    fig.savefig("./tree.png")


if __name__ == "__main__":
    main()
