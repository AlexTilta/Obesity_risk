import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = BASE_DIR / "ObesityDataSet.csv"
PREPARED_DATA_PATH = DATA_DIR / "prepared_data.csv"
MODELS_DIR = BASE_DIR / "models"

os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

TARGET_COLUMN = 'NObeyesdad'
ANTHROPOMETRICS = ['Height', 'Weight', 'BMI']

# Конфигурации для экспериментов с признаками
COLUMNS_TO_DROP_HEIGHT_WEIGHT_ONLY = [TARGET_COLUMN, 'BMI']
COLUMNS_TO_DROP_BMI_ONLY = [TARGET_COLUMN, 'Height', 'Weight']
COLUMNS_TO_DROP_ANTHROPOMETRICS_EXCLUDED = [TARGET_COLUMN] + ANTHROPOMETRICS

# Категориальные признаки для CatBoost
CAT_FEATURES = [
    'Gender',
    'family_history_with_overweight',
    'FCVC',
    'FAVC',
    'CAEC',
    'CH2O',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS',
    'NCP',
    'FAF',
    'TUE'
]

# Словари для предобработки категориальных признаков
CAEC_CALC_MAP = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
FCVC_MAP = {'no': 0, 'Sometimes': 1, 'Always': 2}
NCP_MAP = {'1-2': 1.5, '3': 3.0, '>3': 3.5}
CH2O_MAP = {'<1': 0.5, 'Sometimes': 1.0, '1-2': 1.5, '>2': 3.0}
FAF_TUE_MAP = {'0': 0, 'no': 0, '0-2': 1, '1-2': 1, '2-4': 3, '3-5': 4, '4-5': 4, '>5': 5}
YES_NO_MAP = {'yes': 1.0, 'no': 0.0}
GENDER_MAP = {'Male': 1.0, 'Female': 0.0}

MODEL_NAMES = {
    "log_reg": "Логистическая регрессия",
    "svm": "Модель SVM",
    "tree": "Деревья",
    "rf": "Случайный лес",
    "stacking": "Стекинг",
    "cat_boost": "CatBoost"
}

METRIC_COLUMNS = {
    "F1": "F1-Score",
    "ACCURACY": "Accuracy",
    "PRECISION": "Precision",
    "RECALL": "Recall",
    "ROC_AUC": "ROC_AUC",
    "LogLoss": "LogLoss",
    "TIME": "Время CV (сек)"
}