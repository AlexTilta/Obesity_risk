import os
import joblib

from config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    COLUMNS_TO_DROP_BMI_ONLY,
    MODEL_NAMES
)
from data_loader import load_raw_data, preprocess_data, get_train_test_splits
from tuning import tune_random_forest
from models import evaluate_model_cv, evaluate_on_test, format_results_table
from predictor import ObesityPredictor


def main():
    raw_df = load_raw_data(RAW_DATA_PATH)
    if raw_df is None or raw_df.empty:
        print("Ошибка: данные не загружены.")
        return

    transformed_df = preprocess_data(raw_df)

    (X_train, X_test, y_train, y_test), _ = get_train_test_splits(
        raw_df,
        transformed_df,
        drop_columns=COLUMNS_TO_DROP_BMI_ONLY
    )

    model = tune_random_forest(X_train, y_train, n_trials=5)
    print("Модель обучена.")

    results_cv = evaluate_model_cv(
        model,
        X_train,
        y_train,
        MODEL_NAMES["rf"],
        {}
    )
    print(format_results_table(results_cv))

    evaluate_on_test(
        model,
        X_train, y_train,
        X_test, y_test,
        MODEL_NAMES["rf"]
    )

    model_path = os.path.join(MODELS_DIR, "random_forest_bmi.joblib")
    joblib.dump(model, model_path)
    print(f"\nМодель сохранена: {model_path}")

    try:
        predictor = ObesityPredictor(model_name="random_forest", mode="bmi")
    except FileNotFoundError as e:
        print(f"Не удалось загрузить модель: {e}")
        return

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

    prediction_result = predictor.predict(new_patient)

    print(f"\nПредсказанный класс: {prediction_result['prediction']}")
    for cls, prob in prediction_result['probabilities'].items():
        print(f"  {cls}: {prob:.4f}")


if __name__ == "__main__":
    main()