from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, make_scorer
from statistics import mean, stdev


def RandomForestModel(sub_df, df_squamosa):
    x_train, x_test, y_train, y_test = train_test_split(sub_df, df_squamosa, test_size=None, shuffle=True)

    rf = RandomForestClassifier()

    param_dist = {
        'n_estimators': [10, 20, 30, 40, 50],
        'criterion': ['gini'],
        'max_features' : ['log2', 'sqrt'],
        'max_depth' : [1,2,3,4,5],
        'min_samples_split' : [2, 5],
        'min_samples_leaf' : [ 2, 5],
        'bootstrap': [True, False]
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score),
        'f1': make_scorer(f1_score)
    }

    grid = GridSearchCV(estimator=rf, param_grid=param_dist, n_jobs=-1, cv=5, scoring=scoring, refit='accuracy')
    grid.fit(x_train, y_train)

    mean_acc = mean(grid.cv_results_['mean_test_accuracy'])*100
    std_acc = stdev(grid.cv_results_['mean_test_accuracy'])*100
    mean_roc_auc = mean(grid.cv_results_['mean_test_roc_auc'])
    std_roc_auc = stdev(grid.cv_results_['mean_test_roc_auc'])
    mean_f1 = mean(grid.cv_results_['mean_test_f1'])
    std_f1 = stdev(grid.cv_results_['mean_test_f1'])

    return mean_acc, std_acc, mean_roc_auc, std_roc_auc, mean_f1, std_f1
