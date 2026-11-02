import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    
    print(f"Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Данные успешно загружены. Размер: {df.shape}")
    return df


def preprocess_data(df_original: pd.DataFrame) -> pd.DataFrame:
    
    print("Запуск предобработки данных...")
    
    # Работаем с копией, чтобы не изменять исходный датафрейм
    df = df_original.copy()

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


    mtrans_encoder = LabelEncoder()
    df['MTRANS'] = mtrans_encoder.fit_transform(df['MTRANS']).astype(float)


    int_cols = ['FCVC', 'CAEC', 'FAF', 'TUE', 'CALC']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype('Int64')

    exclude_cols = [TARGET_COLUMN] + int_cols
    for col in df.columns:
        if col not in exclude_cols:
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                pass

    print("Предобработка завершена.")
    return df


def get_train_test_splits(
    df_original: pd.DataFrame, 
    df_transformed: pd.DataFrame, 
    drop_columns: list,
    test_size: float = 0.25 # По умолчанию 25% на тест, как в sklearn
):
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