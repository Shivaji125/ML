from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(y_true, y_pred) -> dict:
    """ Calculates and returns a dictionary of key classification mertics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted")
    }

    return metrics