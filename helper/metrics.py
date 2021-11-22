from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

def compute_scores(y_true, y_pred):
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
            
        Returns 
        -------
            F1 score, Recall, Precision, Accuracy
    """
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f'F1-score : {f1:.4f}')
    print(f'Recall : {recall:.4f}')
    print(f'Precision : {precision:.4f}')
    print(f'Accuracy : {accuracy:.4f}')
    
    return f1, recall, precision, accuracy
