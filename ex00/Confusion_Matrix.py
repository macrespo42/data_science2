import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


def main() -> None:
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    jedi_total = 0
    sith_total = 0

    with open("./truth.txt") as truth, open("./predictions.txt") as preds:
        for t, p in zip(truth, preds):
            t = t.strip()
            p = p.strip()

            if p == "Jedi":
                jedi_total += 1
            elif p == "Sith":
                sith_total += 1

            if p == "Jedi" and p == t:
                tp += 1
            elif p == "Sith" and p == t:
                tn += 1
            elif p == "Jedi" and p != t:
                fp += 1
            elif p == "Sith" and p != t:
                fn += 1
    confusion_matrix = [
        [tp, fn],
        [fp, tn],
    ]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    j_precision = tp / (tp + fp)
    j_recall = tp / (tp + fn)
    j_f1_score = 2 * (j_precision * j_recall) / (j_precision + j_recall)

    s_precision = tn / (fn + tn)
    s_recall = tn / (tn + fp)
    s_f1_score = 2 * (s_precision * s_recall) / (s_precision + s_recall)

    data = {
        "precision": [j_precision, s_precision],
        "recall": [j_recall, s_recall],
        "f1-score": [j_f1_score, s_f1_score],
        "total": [jedi_total, sith_total],
    }
    df = pd.DataFrame(data=data, index=["Jedi", "Sith"])
    print(df)
    print(f"accuracy                    {accuracy}      {sith_total + jedi_total}")
    print(confusion_matrix)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=np.array(confusion_matrix), display_labels=[0, 1]
    )
    cm_display.plot()
    plt.show()


if __name__ == "__main__":
    main()
