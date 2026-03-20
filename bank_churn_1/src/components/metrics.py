from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true, y_pred, y_prob=None, y_train_true=None, y_train_pred=None
) -> dict:
    """Returns detailed classification metrics for binary classification.

    Includes both binary (positive-class) and weighted metrics,
    designed for imbalanced datasets.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        # Binary (positive-class) metrics — useful for imbalanced data
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
        # Weighted metrics — accounts for class imbalance in averaging
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        # Confusion matrix
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # ROC-AUC (requires probability scores)
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    # Train accuracy for overfitting detection
    if y_train_true is not None and y_train_pred is not None:
        metrics["train_accuracy"] = accuracy_score(y_train_true, y_train_pred)

    return metrics
