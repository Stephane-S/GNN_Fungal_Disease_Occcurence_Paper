import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, make_scorer
from statistics import mean, stdev

def KnnModel(sub_df, df_squamosa):
    x_train, x_test, y_train, y_test = train_test_split(sub_df, df_squamosa, test_size=None, shuffle=True)

    knn = KNeighborsClassifier()

    param_dist = {
        'n_neighbors': [1, 3, 5, 7],
        'weights': ['uniform'],
        'metric' : ['euclidean'],
        'leaf_size' : [int(x) for x in np.linspace(start=1, stop= 50, num= 10)]
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score),
        'f1': make_scorer(f1_score)
    }

    grid = GridSearchCV(estimator=knn, param_grid=param_dist, n_jobs=-1, cv=5, scoring=scoring, refit='accuracy', error_score='raise')
    grid.fit(x_train, y_train)

    mean_acc = mean(grid.cv_results_['mean_test_accuracy'])*100
    std_acc = stdev(grid.cv_results_['mean_test_accuracy'])*100
    mean_roc_auc = mean(grid.cv_results_['mean_test_roc_auc'])
    std_roc_auc = stdev(grid.cv_results_['mean_test_roc_auc'])
    mean_f1 = mean(grid.cv_results_['mean_test_f1'])
    std_f1 = stdev(grid.cv_results_['mean_test_f1'])

    return mean_acc, std_acc, mean_roc_auc, std_roc_auc, mean_f1, std_f1