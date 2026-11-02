# src/data_loader.py

"""
Модуль для загрузки и предобработки данных.
Содержит функции для чтения CSV, преобразования категориальных признаков
и создания train/test сплитов.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Импортируем все необходимые константы из нашего конфига
from config import (
    RAW_DATA_PATH,
    PREPARED_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    CAEC_CALC_MAP,
    FCVC_MAP,
    NCP_MAP,
    CH2O_MAP,
    FAF_TUE_MAP,
    YES_NO_MAP,
    GENDER_MAP
)


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Загружает сырые данные из CSV-файла.
    
    Args:
        filepath (str): Путь к CSV-файлу. По умолчанию берется из конфига.
        
    Returns:
        pd.DataFrame: Загруженный датафрейм.
    """
    print(f"Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Данные успешно загружены. Размер: {df.shape}")
    return df


def preprocess_data(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет предобработку сырых данных: заменяет строковые категории
    на числовые значения, используя словари из config.py, и приводит типы.
    
    Args:
        df_original (pd.DataFrame): Сырой датафрейм.
        
    Returns:
        pd.DataFrame: Преобразованный датафрейм, готовый к подаче в модели.
    """
    print("Запуск предобработки данных...")
    
    # Работаем с копией, чтобы не изменять исходный датафрейм
    df = df_original.copy()

    # 1. Применяем словари маппинга из config.py
    # Это заменяет длинные цепочки if/else или методы .map()
    df['CAEC'] = df['CAEC'].replace(CAEC_CALC_MAP)
    df['CALC'] = df['CALC'].replace(CAEC_CALC_MAP) # Используем тот же маппинг
    df['FCVC'] = df['FCVC'].replace(FCVC_MAP)
    df['NCP'] = df['NCP'].replace(NCP_MAP)
    df['CH2O'] = df['CH2O'].replace(CH2O_MAP)
    df['FAF'] = df['FAF'].replace(FAF_TUE_MAP)
    df['TUE'] = df['TUE'].replace(FAF_TUE_MAP)

    df['family_history_with_overweight'] = df['family_history_with_overweight'].replace(YES_NO_MAP)
    df['FAVC'] = df['FAVC'].replace(YES_NO_MAP)
    df['SMOKE'] = df['SMOKE'].replace(YES_NO_MAP)
    df['SCC'] = df['SCC'].replace(YES_NO_MAP)

    df['Gender'] = df['Gender'].replace(GENDER_MAP)

    # 2. Кодируем MTRANS с помощью LabelEncoder
    # В ноутбуке вы делали это так же.
    mtrans_encoder = LabelEncoder()
    df['MTRANS'] = mtrans_encoder.fit_transform(df['MTRANS']).astype(float)

    # 3. Приводим типы данных
    # Список колонок, которые должны быть целочисленными (Int64)
    int_cols = ['FCVC', 'CAEC', 'FAF', 'TUE', 'CALC']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype('Int64')

    # Остальные (кроме таргета и int_cols) переводим в float
    exclude_cols = [TARGET_COLUMN] + int_cols
    for col in df.columns:
        if col not in exclude_cols:
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                pass # Если колонка не конвертируется, оставляем как есть

    print("Предобработка завершена.")
    return df


def get_train_test_splits(
    df_original: pd.DataFrame, 
    df_transformed: pd.DataFrame, 
    drop_columns: list,
    test_size: float = 0.25 # По умолчанию 25% на тест, как в sklearn
):
    """
    Создает train и test выборки для двух типов данных:
    1. Для обычных моделей (на преобразованных данных).
    2. Для CatBoost (на оригинальных данных с категориальными фичами).
    
    Args:
        df_original (pd.DataFrame): Сырой датафрейм (для CatBoost).
        df_transformed (pd.DataFrame): Преобразованный датафрейм.
        drop_columns (list): Список колонок для удаления (включая таргет).
        test_size (float): Доля тестовой выборки.
        
    Returns:
        tuple: Кортеж из двух кортежей:
            ((X_train, X_test, Y_train, Y_test), (X_cat_train, X_cat_test, Y_cat_train, Y_cat_test))
    """
    print(f"Создание train/test сплитов (test_size={test_size})...")

    # --- 1. Для стандартных моделей (числовые данные) ---
    X = df_transformed.drop(columns=drop_columns)
    Y = df_transformed[TARGET_COLUMN]
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=test_size, 
        random_state=RANDOM_STATE, 
        stratify=Y # Сохраняем пропорции классов
    )

    # --- 2. Для CatBoost (оригинальные данные) ---
    X_cat = df_original.drop(columns=drop_columns)
    Y_cat = df_original[TARGET_COLUMN]
    
    X_cat_train, X_cat_test, Y_cat_train, Y_cat_test = train_test_split(
        X_cat, Y_cat, 
        test_size=test_size, 
        random_state=RANDOM_STATE
    )

    print("Сплиты готовы.")
    return (X_train, X_test, Y_train, Y_test), (X_cat_train, X_cat_test, Y_cat_train, Y_cat_test)