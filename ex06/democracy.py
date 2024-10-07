import sys
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


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

    clf1 = LogisticRegression(random_state=42)
    clf2 = DecisionTreeClassifier(random_state=42)
    clf3 = KNeighborsClassifier(n_neighbors=5)

    labels = ["Logistic regression", "DecisionTree", "KNN"]

    for clf, label in zip([clf1, clf2, clf3], labels):
        scores = model_selection.cross_val_score(
            clf, X_train, y_train, cv=5, scoring="accuracy"
        )

        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    voting_clf_hard = VotingClassifier(
        estimators=[(labels[0], clf1), (labels[1], clf2), (labels[2], clf3)],
        voting="hard",
    )
    voting_clf_hard.fit(X_train, y_train)
    y_pred = voting_clf_hard.predict(X_test)
    f1 = f1_score(y_test.values, y_pred, pos_label='Sith')
    print(f"F1-score: {f1:.4f}")

    with open("Voting.txt", "w+") as f:
        for pred in y_pred:
            f.write(f"{pred}\n")


if __name__ == "__main__":
    main()
