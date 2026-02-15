import csv
import numpy as np
import os
import pandas as pd
import seaborn as sb

from catboost import CatBoostClassifier
from math import log2
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer, mean_squared_error, \
    root_mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


if __name__ == '__main__':
    VOWELS = set("aeiou")  # множество гласных букв

    # for dirname, _, filenames in os.walk('/kaggle/input'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))

    def shannon_entropy(s):
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * log2(p) for p in probs)

    def digit_ratio(name):
        length = len(name)
        digits = sum(c.isdigit() for c in name)
        return digits / max(1, length)

    def vowel_ratio(name):
        letters = sum(c.isalpha() for c in name)
        vowels = sum(c in VOWELS for c in name)
        return vowels / max(1, letters)


    def extract_features(domain):
        name = domain.split('.')[0]

        length = len(name)  # длина
        digits = sum(c.isdigit() for c in name)  # количество цифр
        has_digits_flag = digits > 0  # есть ли цифры
        letters = sum(c.isalpha() for c in name)  # количество букв
        entropy = shannon_entropy(name) if length > 0 else 0  # энтропия
        has_dash_flag = "-" in name  # есть ли дефис
        dash_count = name.count("-")  # количество дефисов
        digit_ratio_value = digit_ratio(name)  # доля цифр
        dot_count = domain.count(".")  # количество точек
        vowel_ratio_value = vowel_ratio(name)  # доля гласных
        uniqueness = len(set(name)) / len(name) if len(name) > 0 else 0, # Коэффициент уникальности букв

        features = [
            length,
            digits,
            int(has_digits_flag),
            letters,
            entropy,
            int(has_dash_flag),
            dash_count,
            digit_ratio_value,
            dot_count,
            vowel_ratio_value,
            uniqueness
        ]

        return features

    train = pd.read_csv("datasets/dga_train.csv") #pd.read_csv("/kaggle/input/dga-domain-detection-challenge-i/train.csv")
    test =  pd.read_csv("datasets/dga_test.csv") # pd.read_csv("/kaggle/input/dga-domain-detection-challenge-i/test.csv")

    X_train = np.array([
        extract_features(str(d))
        for d in tqdm(train["domain"], desc="Extracting train features")  # tqdm оборачивает список и создает progress bar
    ])
    y_train = train["label"].values

    X_test = np.array([
        extract_features(str(d))
        for d in tqdm(test["domain"], desc="Extracting test features")
    ])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')


    """  Обучаем ФИНАЛЬНУЮ МОДЕЛЬ с найденными параметрами """
    final_model = CatBoostClassifier(
        depth=10,
        l2_leaf_reg=3,
        learning_rate=0.15,
        bootstrap_type='Bayesian',
        n_estimators=3000,  # теперь можно и 2000-5000 для точности
        loss_function='Logloss',  # стандарт для классификации. или F1
        random_seed=42,  # правильное число
        task_type="GPU", # "CPU" / Если есть видеокарта NVIDIA, поставить "GPU" - будет в 10 раз быстрее
        devices='0',  # Индекс моей видеокарты
        thread_count=-1, # Использует все ядра без копирования данных в памяти
        verbose=100  # будет писать лог каждые 100 деревьев, чтобы не было скучно
    )

    final_model.fit(X_train, y_train) # Обучаем ОДИН РАЗ на всех тренировочных данных

    """ ПРЕДСКАЗЫВАЕМ на рабочей выборке """
    thmax = 0.16
    test_probs = final_model.predict_proba(X_test)[:, 1]  # Получаем вероятности (самый точный инструмент)
    final_classes = test_probs > thmax  # Применяем найденный порог

    """ Сохраняем результат """
    test["label"] = final_classes.astype(int) # Сохраняем результат в DataFrame
    test[["id", "label"]].to_csv("submission_no_prepare.csv", index=False)
    test[["domain", "label"]].to_csv("predict_CatBoost.csv", index=False)  # [optional] датасет с предсказанными значениями


    """
    
    Лучшие параметры: {'bootstrap_type': 'Bayesian', 'depth': 10, 'l2_leaf_reg': 3, 'learning_rate': 0.15}
    Лучший F2-score: 0.768978689832473
    Оптимальный порог: 0.16000000000000003 (F2 на train: 0.8686)
    
    
    """

