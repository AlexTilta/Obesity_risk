import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    log_loss, 
    roc_auc_score,
    accuracy_score
)

from config import (
    METRIC_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN
)

# Настраиваем стиль графиков
sns.set_theme(style="whitegrid")


def calculate_metrics(y_true, y_pred, y_proba) -> dict:
    
    # Получаем все классы, чтобы отчет был корректным
    labels = np.unique(y_true)
    
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    try:
        logloss = log_loss(y_true, y_proba, labels=labels)
    except Exception:
        logloss = np.nan
        
    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted', labels=labels)
    except Exception:
        roc_auc = np.nan

    return {
        METRIC_COLUMNS["F1"]: report['weighted avg']['f1-score'],
        METRIC_COLUMNS["ACCURACY"]: report['accuracy'],
        METRIC_COLUMNS["LogLoss"]: logloss,
        METRIC_COLUMNS["RECALL"]: report['weighted avg']['recall'],
        METRIC_COLUMNS["PRECISION"]: report['weighted avg']['precision'],
        METRIC_COLUMNS["ROC_AUC"]: roc_auc,
    }


def evaluate_model_cv(
    pipeline, 
    x, 
    y, 
    model_name: str, 
    results_dict: dict, 
    n_splits: int = 5
) -> dict:
    print(f"\n--- Оценка модели: {model_name} (CV, {n_splits} фолдов) ---")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    start_time = time.time()
    
    # Предсказание классов
    y_pred_cv = cross_val_predict(pipeline, x, y, cv=cv, n_jobs=-1)

    try:
        y_proba_cv = cross_val_predict(pipeline, x, y, cv=cv, n_jobs=-1, method='predict_proba')
    except AttributeError:
        print("ВНИМАНИЕ: Модель не поддерживает predict_proba. LogLoss и ROC_AUC будут NaN.")
        y_proba_cv = None
        
    end_time = time.time()
    cv_time = end_time - start_time

    if y_proba_cv is not None:
        metrics = calculate_metrics(y, y_pred_cv, y_proba_cv)
    else:
        report = classification_report(y, y_pred_cv, output_dict=True, zero_division=0)
        metrics = {
            METRIC_COLUMNS["F1"]: report['weighted avg']['f1-score'],
            METRIC_COLUMNS["ACCURACY"]: report['accuracy'],
            METRIC_COLUMNS["LogLoss"]: np.nan,
            METRIC_COLUMNS["RECALL"]: report['weighted avg']['recall'],
            METRIC_COLUMNS["PRECISION"]: report['weighted avg']['precision'],
            METRIC_COLUMNS["ROC_AUC"]: np.nan,
        }

    metrics[METRIC_COLUMNS["TIME"]] = cv_time

    # Сохраняем в общий словарь
    results_dict[model_name] = metrics
    
    # Вывод в консоль
    print(f"Время: {cv_time:.2f} сек.")
    print(f"F1-Score: {metrics[METRIC_COLUMNS['F1']]:.4f}")
    print(f"Accuracy: {metrics[METRIC_COLUMNS['ACCURACY']]:.4f}")
    print(f"LogLoss:  {metrics[METRIC_COLUMNS['LogLoss']]:.4f}")
    
    # Строим матрицу ошибок
    plot_confusion_matrix(y, y_pred_cv, model_name)
    
    return results_dict


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """
    Строит и показывает матрицу ошибок.
    """
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels, 
        yticklabels=labels
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()


def evaluate_on_test(pipeline, x_train, y_train, x_test, y_test, model_name: str) -> dict:
    """
    Обучает модель на тренировочных данных и оценивает на тестовых.
    
    Args:
        pipeline: Модель.
        x_train, y_train: Тренировочные данные.
        x_test, y_test: Тестовые данные.
        model_name: Название модели.
        
    Returns:
        dict: Метрики на тестовой выборке.
    """
    print(f"\n=== Финальное обучение и тест: {model_name} ===")
    
    start_time = time.time()
    pipeline.fit(x_train, y_train)
    fit_time = time.time() - start_time
    print(f"Время обучения: {fit_time:.2f} сек.")
    
    y_pred_test = pipeline.predict(x_test)
    
    try:
        y_proba_test = pipeline.predict_proba(x_test)
        metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)
    except AttributeError:
        report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        metrics = {
            METRIC_COLUMNS["F1"]: report['weighted avg']['f1-score'],
            METRIC_COLUMNS["ACCURACY"]: report['accuracy'],
            METRIC_COLUMNS["LogLoss"]: np.nan,
            METRIC_COLUMNS["RECALL"]: report['weighted avg']['recall'],
            METRIC_COLUMNS["PRECISION"]: report['weighted avg']['precision'],
            METRIC_COLUMNS["ROC_AUC"]: np.nan,
        }
    
    print(f"F1-Score: {metrics[METRIC_COLUMNS['F1']]:.4f}")
    print(f"Accuracy: {metrics[METRIC_COLUMNS['ACCURACY']]:.4f}")
    print(f"LogLoss:  {metrics[METRIC_COLUMNS['LogLoss']]:.4f}")
    
    plot_confusion_matrix(y_test, y_pred_test, f"{model_name} (Test)")
    
    return metrics


def format_results_table(results_dict: dict) -> pd.DataFrame:
    """
    Превращает словарь с результатами в красивую отформатированную таблицу.
    """
    if not results_dict:
        print("Словарь результатов пуст.")
        return pd.DataFrame()
        
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    
    # Сортируем по F1 (если такая колонка есть)
    if METRIC_COLUMNS["F1"] in df.columns:
        df = df.sort_values(by=METRIC_COLUMNS["F1"], ascending=False)
        
    pd.options.display.float_format = '{:.4f}'.format
    
    # Форматируем колонку времени отдельно (чтобы было 2 знака после запятой)
    if METRIC_COLUMNS["TIME"] in df.columns:
        df[METRIC_COLUMNS["TIME"]] = pd.to_numeric(df[METRIC_COLUMNS["TIME"]], errors='coerce')
        df[METRIC_COLUMNS["TIME"]] = df[METRIC_COLUMNS["TIME"]].apply(lambda x: f'{x:.2f}')
        
    return df