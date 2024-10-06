import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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


def main() -> None:
    training = None
    validation = None
    try:
        training = pd.read_csv("datasets/Training_data.csv")
        validation = pd.read_csv("datasets/Validation_data.csv")
    except Exception:
        print("Please run the script from the root directory")
        exit(1)

    X_train, X_test, y_train, y_test = preprocess_data(training, validation)

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
    # D'abord, on doit entraîner le classifieur avec fit()
    voting_clf_hard.fit(X_train, y_train)
    # Ensuite on peut faire les prédictions sur X_test
    y_pred = voting_clf_hard.predict(X_test)
    # Calculer et afficher le F1-score
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score: {f1:.4f}")

    with open("Voting.txt", "w+") as f:
        for pred in y_pred:
            if pred == 0:
                f.write("Jedi\n")
            else:
                f.write("Sith\n")


if __name__ == "__main__":
    main()
