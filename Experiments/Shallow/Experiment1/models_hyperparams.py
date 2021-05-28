from sklearn.svm import SVC
from sklearn.svm import LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB


hyperparameters_per_model = {
    'decision_tree': {
        # max_depths: [5, 6, 7],
        # max_features: [4, 5, 6],
        'max_depth': [2, 3],
        'max_features': ['sqrt', 'log2'],
        'ccp_alpha': [0.1]
    },
    'svc': {
        'C': [1000],
        'gamma': [0.4, 0.6],
        'tol': [0.00001],
        # 'kernel': ['poly', 'rbf', 'linear']
    },
    'linear_svc': {
        'penalty': ['l2'],
        'C': [1000],
    },
    'nu_svc': {
        'nu':  [0.50],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.6],
        'tol': [0.00001],
        'probability': [True]
    },
    'logistic_regression': {
        'penalty': ['l2'],
        'C': [1000]
    },
    'ridge': {
        'alpha': [0.5, 0.75, 1.0],
    },
    'sgd': {
        'alpha': [0.9],
        'penalty': ['l1', 'l2'],
    },
    'extra_trees': {
        # 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
        'max_depth': [2, 3],
        'max_features': ['sqrt', 'log2'],
        'ccp_alpha': [0.2]
        # 'min_samples_split':  [0.005, 0.01, 0.05, 0.10],
    },
    'bernouli': {
        # 'alpha': [1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
        # 'alpha': [0.3, 0.5, 0.5, 0.6]
        # 'alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
        'alpha': [0.50, 0.75, 1.0]
    },
    'gaussian': {
        # 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0
        'var_smoothing':  [1e-9]
    },
    'random_forest': {
        'max_depth': [2, 3],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [0.2, 0.3],
        'ccp_alpha': [0.4]
    },
    'ada_boost': {
        'base_estimator': [RidgeClassifier(0.9)],
        'n_estimators': [1],
        'algorithm': ['SAMME']
    },
    'gradient_boost': {
        'n_estimators': [2, 3],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [2, 3],
        'subsample': [1],
        'ccp_alpha': [0.1]
    },
    'xg_boost': {
        # 'max_depth': [1, 2, 3],
        # 'min_child_weight':  [1, 3, 5],
        # 'gamma': [0.2, 0.3],
        # 'colsample_bytree': [0.3, 0.4],
        # 'objective': ['reg:squarederror']

        'max_depth': [3],
        'min_child_weight':  [3],
        'gamma': [0.2, 0.3, 0.4, 0.5, 0.6],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.6],
        'objective': ['reg:squarederror'],
        'reg_alpha ': [0.9]
    }
}

models_to_test = {
    LogisticRegression: 'logistic_regression',
    ExtraTreesClassifier: 'extra_trees', # not overfitting
    RandomForestClassifier: 'random_forest', # not overfitting
    DecisionTreeClassifier: 'decision_tree', # not overfitting
    GradientBoostingClassifier: 'gradient_boost', # not overfitting
    XGBClassifier: 'xg_boost', # not overfitting
    AdaBoostClassifier: 'ada_boost', # overfitting - tried many many times
    NuSVC: 'nu_svc', # somehow overfitting
}

non_probabilistic_models_to_test = {
    RidgeClassifier: 'ridge', # no predict proba # tried a lot
    BernoulliNB: 'bernouli',
    GaussianNB: 'gaussian',
    SGDClassifier: 'sgd', # no predict proba
    LinearSVC: 'linear_svc', # not proba # not overfitting
    SVC: 'svc',
}