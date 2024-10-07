import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main() -> None:
    df = None
    try:
        df = pd.read_csv("Train_knight.csv")
    except Exception:
        print("Please run the script from the root of the project")
        exit(1)

    X = df.drop(
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
            "Hability",
            "Reactivity",
            "Blocking",
            "Repulse",
            "Sprint",
            "Prescience",
            "Deflection",
            "Mass",
            "Survival",
        ],
        axis=1,
    )

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(len(X.columns))
    ]

    print(vif_data)


if __name__ == "__main__":
    main()
