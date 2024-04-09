import pandas as pd
from sklearn.metrics import accuracy_score
import argparse


def calculate_accuracy(true_labels_csv, predicted_labels_csv):
    true_labels = pd.read_csv(true_labels_csv)
    predicted_labels = pd.read_csv(predicted_labels_csv)

    true_labels = true_labels.sort_values(by="Index")
    predicted_labels = predicted_labels.sort_values(by="Index")

    accuracy = accuracy_score(
        true_labels["Hogwarts House"], predicted_labels["Hogwarts House"]
    )
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Calculate accuracy score between two CSV files."
    )
    parser.add_argument("true_labels", type=str, help="CSV file with true labels.")
    parser.add_argument(
        "predicted_labels", type=str, help="CSV file with predicted labels."
    )

    args = parser.parse_args()

    accuracy = calculate_accuracy(args.true_labels, args.predicted_labels)
    print(f"Accuracy Score: {accuracy}")


if __name__ == "__main__":
    main()
