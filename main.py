# main.py

"""
Главный скрипт проекта.
Демонстрирует полный пайплайн:
1. Загрузка данных.
2. Предобработка.
3. Обучение модели (с подбором гиперпараметров).
4. Оценка на кросс-валидации.
5. Финальное тестирование.
6. Сохранение модели.
7. Демонстрация инференса (предсказания для нового пациента).
"""

import os
import joblib
import pandas as pd

from config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    COLUMNS_TO_DROP_BMI_ONLY,
    COLUMNS_TO_DROP_ANTHROPOMETRICS_EXCLUDED,
    MODEL_NAMES
)
from data_loader import load_raw_data, preprocess_data, get_train_test_splits
from tuning import tune_random_forest
from models import evaluate_model_cv, evaluate_on_test, format_results_table
from predictor import ObesityPredictor


def main():
    print("=" * 60)
    print("🚀 ЗАПУСК ML-ПАЙПЛАЙНА ПРОЕКТА 'Obesity Risk'")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. ЗАГРУЗКА ДАННЫХ
    # -------------------------------------------------------------------------
    print("\n[1/7] Загрузка данных...")
    raw_df = load_raw_data(RAW_DATA_PATH)
    
    if raw_df is None or raw_df.empty:
        print("❌ Ошибка: данные не загружены. Проверьте путь к файлу.")
        return

    # -------------------------------------------------------------------------
    # 2. ПРЕДОБРАБОТКА
    # -------------------------------------------------------------------------
    print("\n[2/7] Предобработка данных...")
    transformed_df = preprocess_data(raw_df)

    # -------------------------------------------------------------------------
    # 3. СОЗДАНИЕ СПЛИТОВ (например, режим 'bmi')
    # -------------------------------------------------------------------------
    print("\n[3/7] Создание train/test сплитов (режим: с BMI)...")
    (X_train, X_test, y_train, y_test), _ = get_train_test_splits(
        raw_df, 
        transformed_df, 
        drop_columns=COLUMNS_TO_DROP_BMI_ONLY
    )

    # -------------------------------------------------------------------------
    # 4. ОБУЧЕНИЕ МОДЕЛИ (с подбором гиперпараметров)
    # -------------------------------------------------------------------------
    print("\n[4/7] Подбор гиперпараметров для Random Forest...")
    # Для демонстрации используем малое количество trials (n_trials=5),
    # чтобы скрипт выполнялся быстро. В реальности ставьте 50-100.
    model = tune_random_forest(X_train, y_train, n_trials=5)
    print("✅ Модель обучена.")

    # -------------------------------------------------------------------------
    # 5. ОЦЕНКА НА КРОСС-ВАЛИДАЦИИ
    # -------------------------------------------------------------------------
    print("\n[5/7] Оценка модели с помощью кросс-валидации...")
    results_cv = {}
    results_cv = evaluate_model_cv(
        model, 
        X_train, 
        y_train, 
        MODEL_NAMES["rf"], 
        results_cv
    )
    
    # Вывод красивой таблицы с метриками
    print("\n📊 Результаты кросс-валидации:")
    display(format_results_table(results_cv))

    # -------------------------------------------------------------------------
    # 6. ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ И СОХРАНЕНИЕ
    # -------------------------------------------------------------------------
    print("\n[6/7] Финальное обучение и тестирование...")
    test_metrics = evaluate_on_test(
        model, 
        X_train, y_train, 
        X_test, y_test, 
        MODEL_NAMES["rf"]
    )

    # Сохранение модели для дальнейшего использования в API
    model_filename = f"random_forest_bmi.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_path)
    print(f"\n💾 Модель сохранена по пути: {model_path}")

    # -------------------------------------------------------------------------
    # 7. ДЕМОНСТРАЦИЯ ИНФЕРЕНСА (PREDICTOR)
    # -------------------------------------------------------------------------
    print("\n[7/7] Демонстрация работы класса ObesityPredictor (инференс)...")
    
    # Проверяем, что модель сохранилась и предиктор может её загрузить
    try:
        predictor = ObesityPredictor(model_name="random_forest", mode="bmi")
    except FileNotFoundError as e:
        print(f"❌ Не удалось загрузить модель для предиктора: {e}")
        return

    # Пример данных нового пациента (сырые данные, как в анкете)
    new_patient = {
        'Gender': 'Male',
        'Age': 25.0,
        'Height': 1.75,
        'Weight': 85.0,
        'family_history_with_overweight': 'yes',
        'FAVC': 'yes',
        'FCVC': 'Always',
        'NCP': '3',
        'CAEC': 'Sometimes',
        'SMOKE': 'no',
        'CH2O': '2.0',
        'SCC': 'no',
        'FAF': '1',
        'TUE': '1',
        'CALC': 'Sometimes',
        'MTRANS': 'Public_Transportation'
    }

    print("\n👤 Данные нового пациента:")
    for key, value in new_patient.items():
        print(f"   {key}: {value}")

    # Получаем предсказание
    prediction_result = predictor.predict(new_patient)
    
    print("\n🩺 РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ:")
    print(f"   Предсказанный класс: {prediction_result['prediction']}")
    print("   Вероятности по классам:")
    for cls, prob in prediction_result['probabilities'].items():
        print(f"      - {cls}: {prob:.4f}")

    print("\n" + "=" * 60)
    print("✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    print("=" * 60)


if __name__ == "__main__":
    main()