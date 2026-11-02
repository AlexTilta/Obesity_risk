# src/predictor.py

"""
Модуль для инференса (предсказания) на новых данных.
Содержит класс Predictor, который инкапсулирует логику предобработки 
входных данных и вызова обученной модели.
"""

import os
import pandas as pd
import numpy as np
import joblib

from src.config import (
    MODELS_DIR,
    TARGET_COLUMN,
    COLUMNS_TO_DROP_BMI_ONLY,
    COLUMNS_TO_DROP_ANTHROPOMETRICS_EXCLUDED
)
from src.data_loader import preprocess_data


class ObesityPredictor:
    """
    Класс-обертка для инференса модели предсказания ожирения.
    Поддерживает два режима работы:
      - 'bmi': использует индекс массы тела (не требует роста и веса).
      - 'no_anthro': не использует антропометрию (только привычки).
    """

    def __init__(self, model_name: str = "random_forest", mode: str = "bmi"):
        """
        Инициализация предиктора.
        
        Args:
            model_name: Имя файла модели без расширения (например, 'random_forest').
            mode: Режим работы ('bmi' или 'no_anthro').
        """
        self.mode = mode
        self.model = None
        self.model_path = os.path.join(MODELS_DIR, f"{model_name}_{mode}.joblib")
        
        self._load_model()

    def _load_model(self):
        """Загружает модель с диска."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Модель не найдена по пути: {self.model_path}. "
                "Сначала обучите и сохраните модель с помощью main.py."
            )
        print(f"Загрузка модели из {self.model_path}...")
        self.model = joblib.load(self.model_path)

    def _prepare_input(self, input_data: dict) -> pd.DataFrame:
        """
        Преобразует словарь с признаками в DataFrame и применяет предобработку.
        """
        # 1. Создаем DataFrame из одного словаря (одного пациента)
        df_raw = pd.DataFrame([input_data])
        
        # 2. Применяем ту же предобработку, что и при обучении
        df_transformed = preprocess_data(df_raw)
        
        # 3. Удаляем колонки в зависимости от режима (как при обучении)
        if self.mode == 'bmi':
            df_transformed = df_transformed.drop(
                columns=COLUMNS_TO_DROP_BMI_ONLY, 
                errors='ignore'
            )
        elif self.mode == 'no_anthro':
            df_transformed = df_transformed.drop(
                columns=COLUMNS_TO_DROP_ANTHROPOMETRICS_EXCLUDED, 
                errors='ignore'
            )
            
        return df_transformed

    def predict(self, input_data: dict) -> dict:
        """
        Выполняет предсказание для одного пациента.
        
        Args:
            input_data: Словарь с сырыми признаками (например, {'Gender': 'Male', 'Age': 25.0, ...}).
            
        Returns:
            dict: Предсказанный класс и вероятности для каждого класса.
        """
        if self.model is None:
            raise ValueError("Модель не загружена.")

        # Подготавливаем данные
        df_prepared = self._prepare_input(input_data)

        # Делаем предсказание
        prediction = self.model.predict(df_prepared)[0]
        
        # Получаем вероятности (если модель поддерживает)
        try:
            probabilities = self.model.predict_proba(df_prepared)[0]
            classes = self.model.classes_
            proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        except AttributeError:
            proba_dict = {}

        return {
            "prediction": str(prediction),
            "probabilities": proba_dict
        }

    def predict_batch(self, input_data_list: list[dict]) -> list[dict]:
        """
        Выполняет предсказание для списка пациентов (пакетный режим).
        
        Args:
            input_data_list: Список словарей с признаками.
            
        Returns:
            list[dict]: Список результатов предсказаний.
        """
        if self.model is None:
            raise ValueError("Модель не загружена.")
            
        # Создаем DataFrame из списка словарей
        df_raw = pd.DataFrame(input_data_list)
        
        # Предобработка
        df_transformed = preprocess_data(df_raw)
        
        # Удаление колонок
        if self.mode == 'bmi':
            df_transformed = df_transformed.drop(columns=COLUMNS_TO_DROP_BMI_ONLY, errors='ignore')
        elif self.mode == 'no_anthro':
            df_transformed = df_transformed.drop(columns=COLUMNS_TO_DROP_ANTHROPOMETRICS_EXCLUDED, errors='ignore')
            
        # Предсказание
        predictions = self.model.predict(df_transformed)
        
        # Вероятности
        try:
            probabilities = self.model.predict_proba(df_transformed)
            classes = self.model.classes_
        except AttributeError:
            probabilities = None
            classes = []

        results = []
        for i, pred in enumerate(predictions):
            res = {"prediction": str(pred), "probabilities": {}}
            if probabilities is not None:
                res["probabilities"] = {str(cls): float(prob) for cls, prob in zip(classes, probabilities[i])}
            results.append(res)
            
        return results