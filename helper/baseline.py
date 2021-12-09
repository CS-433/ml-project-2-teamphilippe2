from sklearn.model_selection import cross_validate


def cross_validate(model, ):


    # Different coefficients for the L2 regularizer
    Cs = [1, 500, 1e3, 1e4]
    f1_scores = []

    for C in Cs:
        svm_rbf = model(C=C, class_weight='balanced', kernel='rbf', cache_size=1000)
        cv_results = cross_validate(svm_rbf, X_train, y_train, scoring='f1', n_jobs=-1)

        # Compute mean test f1 score over each fold
        f1_scores.append(np.mean(cv_results['test_score']))

        print(f'F1-score for C={C} : {f1_scores[-1]}')