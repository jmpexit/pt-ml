import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from math import log2
from pathlib import Path
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer, mean_squared_error, \
    root_mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

""" Продолжение задачи DGA. Оптимизируем модель """

if __name__ == '__main__':
    data = pd.read_csv("datasets/dga.csv")
    data = data.sample(900_000, random_state=42) # TODO увеличить дата сампл
    print(len(data))

    # Feature engineering
    VOWELS = set("aeiou")  # множество гласных букв

    def shannon_entropy(s):
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * log2(p) for p in probs)

    def digit_ratio(name):
        length = len(name)
        digits = sum(c.isdigit() for c in name)
        return digits / max(1, length)

    def letter_count(name):
        return sum(c.isalpha() for c in name)

    def has_digits(name):
        return int(any(c.isdigit() for c in name))


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
        ]

        return features

    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    # stratify - гарантирует, что процент вредоносных доменов в train и test будет ОДИНАКОВЫМ

    X_train = np.array([
        extract_features(str(d))
        for d in tqdm(train["domain"], desc="Extracting train features")  # прогресс-бар
    ])
    y_train = train["label"].values

    X_test = np.array([
        extract_features(str(d))
        for d in tqdm(test["domain"], desc="Extracting test features")
    ])
    y_test = test["label"].values

    # Применяем алгоритм градиентного бустинга на решающих деревьях (Categorical Boosting)
    cb = CatBoostClassifier(n_estimators=200, random_seed=42) # TODO n_estimators 500...2000
                # n_estimators - количество деревьев, которые построит алгоритм.
                # Каждое следующее дерево пытается исправить ошибки предыдущих (обычно используют от 500 до 2000)

    """ Используем f2_scorer"""
    f2_scorer = make_scorer(fbeta_score, beta=2)
    params = {
        'depth': [4, 6, 8],  # Глубина деревьев (чем выше, тем сложнее зависимости)
        'learning_rate': [0.05, 0.1],  # Скорость обучения
        'l2_leaf_reg': [1, 3, 5]  # Регуляризация (чтобы не переобучиться)
    }

    gs = GridSearchCV(     #  Настраиваем поиск
        estimator=cb,  # наш CatBoost
        param_grid=params,
        scoring=f2_scorer,  # Оптимизируем F2
        cv=3,  # Кросс-валидация на 3 фолда
        verbose=3,
        n_jobs=-1 # задействует все ядра процессора
    )

    gs.fit(X_train, y_train)     # Запускаем поиск лучших параметров (обучаем)

    best_cb = gs.best_estimator_  # Это наша лучшая модель
    print(f"Лучшие параметры: {gs.best_params_}")
    print(f"Лучший F2-score: {gs.best_score_}")
    """"""

    # Подбор порога
    fbmax = -1
    thmax = -1

    cb_train_probs = best_cb.predict_proba(X_train)[:,1]  # просим модель выдать вероятность от 0 до 1, а именно [:, 1]-
    # вероятности «положительного» класса (вредоносных доменов)

    # Поиск лучшего порога
    for th in tqdm(np.arange(0.01, 1.01, 0.05)):
        classes = cb_train_probs > th
        fbeta = fbeta_score(y_train, classes, beta=2) # вычисляем f_beta скор  # beta=0.5
        if fbeta > fbmax:
          fbmax = fbeta
          thmax = th

    print(f"Оптимальный порог: {thmax} (F2 на train: {fbmax:.4f})")

    # применяем найденный "идеальный" порог к тестовым данным
    cb_test_probs = best_cb.predict_proba(X_test)[:, 1]
    final_classes = cb_test_probs > thmax

    final_f2 = fbeta_score(y_test, final_classes, beta=2)  # вычисляем f_beta скор # beta=0.5
    print(f"Итоговый F2-score на тесте: {final_f2:.4f}")


    """
    Обучаем ФИНАЛЬНУЮ МОДЕЛЬ с найденными параметрами
    Лучшие параметры: {'depth': 8, 'l2_leaf_reg': 1, 'learning_rate': 0.1}
    Лучший F2-score: 0.756593477852197
    Оптимальный порог: 0.16000000000000003 (F2 на train: 0.8645)
    Итоговый F2-score на тесте: 0.8625
    
    """
    best_params = gs.best_params_.copy()
    best_params.pop('n_estimators', None)
    best_params.pop('iterations', None)

    # Создаем модель СРАЗУ с лучшими параметрами
    # Здесь мы объединяем найденные гиперпараметры и 2000 деревьев
    final_model = CatBoostClassifier(
        **best_params,
        n_estimators=2000,  # теперь можно и 2000 для точности
        loss_function='Logloss',  # стандарт для классификации
        random_seed=42,  # правильное число
        verbose=100  # будет писать лог каждые 100 деревьев, чтобы не было скучно
    )

    # Обучаем ОДИН РАЗ на всех тренировочных данных
    final_model.fit(X_train, y_train)

    # Применяем найденный порог (0.16)

    test_probs = final_model.predict_proba(X_test)[:, 1]
    final_classes = test_probs > thmax

    # Проверяем итоговый результат
    final_f2 = fbeta_score(y_test, final_classes, beta=2)
    print(f"Финальный F2 на тесте с порогом {thmax}: {final_f2:.4f}")






    """ Примечания
    Двойная оптимизация: Сначала GridSearchCV находит лучшую "форму" модели (глубину деревьев и т.д.), используя f2_scorer.
    Смена приоритетов: Везде в функциях fbeta_score мы заменили beta=0.5 на beta=2.
    Порог: Скорее всего, thmax теперь будет значительно ниже 0.5 (например, 0.15), так как модель будет стараться 
    "зацепить" как можно больше подозрительных доменов.

    Если данных очень много, GridSearchCV может работать долго. В таком случае можно заменить его на RandomizedSearchCV
    
    Поставьте n_estimators=100 или 200 в cb_base.
Найдите лучшие depth и learning_rate.
И только потом, когда у вас будут идеальные параметры, обучите финальную модель best_cb.fit(...) один раз с n_estimators=2000.
    """
