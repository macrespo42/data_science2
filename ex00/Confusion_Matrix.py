import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


def get_confusion_matrix(truth, prediction):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    jedi_total = 0
    sith_total = 0

    try:
        with open(truth) as truth, open(prediction) as preds:
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
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Stats: tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")
    return tp, fp, tn, fn, jedi_total, sith_total


def main() -> None:
    if len(sys.argv) != 3:
        print("provide 2 arguments: truth and predictions files")
        sys.exit(1)

    tp, fp, tn, fn, jedi_total, sith_total = get_confusion_matrix(
        sys.argv[1], sys.argv[2]
    )

    confusion_matrix = [
        [tp, fp],
        [fn, tn],
    ]

    accuracy = (tp + tn) / (tp + tn + fp + fn) if ((tp + tn + fp + fn)) != 0 else 0
    j_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    j_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    j_f1_score = 2 * (j_precision * j_recall) / (j_precision + j_recall) if (j_precision + j_recall) else 0

    s_precision = tn / (fn + tn) if (fn + tn) > 0 else 0
    s_recall = tn / (tn + fp) if (tn + fp) else 0
    s_f1_score = 2 * (s_precision * s_recall) / (s_precision + s_recall) if (s_precision + s_recall) > 0 else 0

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
