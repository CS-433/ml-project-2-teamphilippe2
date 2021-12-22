from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np


def compute_scores(y_true, y_pred, print_values=True):
    """
    Compute and print the following scores :
        - F1 score
        - Recall
        - Precision
        - Accuracy

    Parameters
    ----------
        y_true :
            Ground truth labels
        y_pred :
            Predicted labels
        print_values:
            If we want to print the result of the prections
    Returns 
    -------
        F1 score, Recall, Precision, Accuracy
    """
    y_true = np.array([y_t.flatten() for y_t in y_true], dtype=np.intc).flatten()
    y_pred = np.array([y_p.flatten() for y_p in y_pred], dtype=np.intc).flatten()

    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    if print_values:
        print("Performance on the local test set")
        print(f'\tF1-score : {f1:.4f}')
        print(f'\tRecall : {recall:.4f}')
        print(f'\tPrecision : {precision:.4f}')
        print(f'\tAccuracy : {accuracy:.4f}')

    return f1, recall, precision, accuracy
