import warnings
import optuna
import numpy as np
from optuna.trial import Trial

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.exceptions import ConvergenceWarning

from catboost import CatBoostClassifier, Pool, cv

from config import RANDOM_STATE

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def tune_logistic_regression(X_train, Y_train, n_trials: int = 50) -> make_pipeline:
    n_classes = len(Y_train.unique())

    def objective(trial: Trial) -> float:
        params = {
            'C': trial.suggest_float('C', 0.001, 1000, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga', 'liblinear']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000)
        }
        
        # Пропускаем несовместимые комбинации параметров
        if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']: return float('-inf')
        if params['penalty'] == 'elasticnet' and params['solver'] != 'saga': return float('-inf')
        if params['solver'] == 'liblinear' and n_classes > 2: return float('-inf')
        
        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)

        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_STATE, **params)
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        try:
            scores = cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        except Exception:
            return float('-inf')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nЛучшие параметры LR: {study.best_params}")
    return make_pipeline(
        StandardScaler(), 
        LogisticRegression(random_state=RANDOM_STATE, **study.best_params)
    )


# ==============================================================================
# 2. SVM
# ==============================================================================
def tune_svm(X_train, Y_train, n_trials: int = 50) -> make_pipeline:
    """
    Подбирает гиперпараметры для SVC.
    """
    def objective(trial: Trial) -> float:
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'probability': True
        }
        
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
            
        if params['kernel'] in ['rbf', 'poly']:
            params['gamma'] = trial.suggest_float('gamma', 1e-4, 1, log=True)

        pipeline = make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE, **params))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        try:
            scores = cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        except Exception:
            return float('-inf')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nЛучшие параметры SVM: {study.best_params}")
    return make_pipeline(
        StandardScaler(), 
        SVC(random_state=RANDOM_STATE, probability=True, **study.best_params)
    )


# ==============================================================================
# 3. ДЕРЕВО РЕШЕНИЙ
# ==============================================================================
def tune_decision_tree(X_train, Y_train, n_trials: int = 50) -> make_pipeline:
    """
    Подбирает гиперпараметры для DecisionTreeClassifier.
    """
    def objective(trial: Trial) -> float:
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }

        pipeline = make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        try:
            scores = cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        except Exception:
            return float('-inf')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nЛучшие параметры Дерева: {study.best_params}")
    return make_pipeline(
        StandardScaler(), 
        DecisionTreeClassifier(random_state=RANDOM_STATE, **study.best_params)
    )


# ==============================================================================
# 4. СЛУЧАЙНЫЙ ЛЕС
# ==============================================================================
def tune_random_forest(X_train, Y_train, n_trials: int = 50) -> make_pipeline:
    """
    Подбирает гиперпараметры для RandomForestClassifier.
    """
    def objective(trial: Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }

        pipeline = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        try:
            scores = cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        except Exception:
            return float('-inf')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nЛучшие параметры RF: {study.best_params}")
    return make_pipeline(
        StandardScaler(), 
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **study.best_params)
    )


# ==============================================================================
# 5. СТЕКИНГ
# ==============================================================================
def tune_stacking(X_train, Y_train, n_trials: int = 30) -> make_pipeline:
    """
    Подбирает гиперпараметры для финального классификатора Stacking.
    Базовые модели фиксированы, как в вашем ноутбуке.
    """
    def objective(trial: Trial) -> float:
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l2']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'max_iter': trial.suggest_int('max_iter', 1000, 3000)
        }

        # Базовые модели (без настройки, как в вашем ноутбуке)
        estimators = [
            ('logreg', make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced'))),
            ('rf', make_pipeline(StandardScaler(), RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=10))),
            ('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE, kernel='rbf')))
        ]

        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=RANDOM_STATE, **params),
            cv=3,
            n_jobs=-1
        )

        pipeline = make_pipeline(StandardScaler(), stacking_clf)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        try:
            scores = cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        except Exception:
            return float('-inf')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nЛучшие параметры Stacking: {study.best_params}")
    
    estimators = [
        ('logreg', make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced'))),
        ('rf', make_pipeline(StandardScaler(), RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=10))),
        ('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE, kernel='rbf')))
    ]

    return make_pipeline(
        StandardScaler(),
        StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=RANDOM_STATE, **study.best_params),
            cv=3,
            n_jobs=-1
        )
    )


# ==============================================================================
# 6. CATBOOST
# ==============================================================================
def tune_catboost(X_train, Y_train, cat_features: list, n_trials: int = 10, timeout: int = 600) -> CatBoostClassifier:
    """
    Подбирает гиперпараметры для CatBoost.
    """
    fixed_params = {
        'iterations': 75,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'loss_function': 'MultiClass'
    }
    
    def objective(trial: Trial) -> float:
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'auto_class_weights': trial.suggest_categorical('auto_class_weights', [None, 'Balanced'])
        }
        
        params.update(fixed_params)
        params['early_stopping_rounds'] = 20
        params['od_type'] = 'Iter'
        params['eval_metric'] = 'Accuracy'
        params['custom_metric'] = ['TotalF1', 'Accuracy']

        try:
            # Обратите внимание: CatBoost ждет Pool с категориальными фичами
            cv_data = cv(
                Pool(X_train, Y_train, cat_features=cat_features),
                params,
                fold_count=3,
                seed=RANDOM_STATE,
                logging_level='Silent',
                stratified=True
            )
            # Берем максимальный F1 из кросс-валидации
            f1_series = cv_data['test-TotalF1-mean'].fillna(0)
            return f1_series.max()
        except Exception:
            return -1.0

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    print(f"\nЛучшие параметры CatBoost: {study.best_params}")
    
    final_params = study.best_params.copy()
    final_params.update(fixed_params)
    
    return CatBoostClassifier(**final_params)