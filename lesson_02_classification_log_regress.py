import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from math import log2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


"""

### Обнаружение DGA-доменов

Эта задача посвящена выявлению алгоритмически сгенерированных доменов второго уровня (SLD) в доменных именах. 
Ваша задача — построить модель машинного обучения, способную отличать легитимные домены от SLD, 
сгенерированных с помощью DGA.

Цель — обнаружение алгоритмически сгенерированных доменов второго уровня (SLD), которые часто используются вредоносным 
ПО для обхода механизмов обнаружения. У нас есть датасет доменных имен с метками: DGA (1) или легитимный (0). 
Обратите внимание, что в некоторых примерах предоставляется только SLD без домена верхнего уровня (TLD).

Поскольку ложноположительные срабатывания (когда легитимный домен ошибочно классифицируется как DGA) могут приводить 
к серьезным проблемам, минимизация ложноположительных ошибок важнее, чем минимизация ложноотрицательных.

тренировочный датасет:
wget -O dga.csv "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/f2Z4w-xHheenKg"
"""


if __name__ == '__main__':
    # Посмотрим на данные
    data = pd.read_csv("datasets/dga.csv")
    print(data.head())
    print(data.sample(10))

    # # Посмотрим на баланс классов
    print(data['label'].value_counts(normalize=True))
    print(len(data))

    data = data.sample(800_000, random_state=42)
    print(len(data))

    # # Feature engineering
    # # Самая важная честь - придумать признаки, на которых мы будем решать задачу

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

    # Разобъем данные на train и test
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    print(len(train), len(test))

    #
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

    # # Классификатор - логистическая ренрессия
    # Перед применением модели масштабируем признаки
    model_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression( # используем простую модель логрега
            max_iter=100,
            random_state=0
        )
    )

    # обучаем модель
    model_lr.fit(X_train, y_train)

    # Оценим качество получившейся модели
    y_test_pred = model_lr.predict(X_test)  # предсказываем на валидационной выборке

    fbeta = fbeta_score(y_test, y_test_pred, beta=0.5)  # вычисляем f_beta скор
    print(fbeta)

    # посмотрим и на остальные показатели, для этого есть функция classification_report
    #   classification_report выводит значения **precision**, **recall**, **F1-score** и **support** для каждого класса:
    #     - **0** — нормальные домены
    #     - **1** — DGA-домены
    # Также выводятся:
    # - **accuracy** — общая доля правильных предсказаний
    # - **macro avg** — среднее арифметическое метрик по классам
    # - **weighted avg** — среднее метрик, взвешенное по количеству объектов каждого класса

    print(classification_report(y_test, y_test_pred, digits=4))

    # ## Улучшаем решение
    f2_scorer = make_scorer(fbeta_score, beta=2) # beta=2 - приоритет Полноте (Recall). нам важнее найти как можно
    # больше вредоносных доменов, даже если будет чуть больше ложных тревог

    params = {'logisticregression__class_weight': [None, 'balanced'], # None: Относиться к обоим классам как к равным;
              # 'balanced': Уделить больше внимания редкому классу
              'logisticregression__C': [0.5, 1., 2.]} # C - сила регуляризации. Меньшие значения (0.5) делают модель
        # «проще», чтобы избежать переобучения, а большие (2.0) позволяют модели сильнее подстраиваться под данные.
        # Двойное подчеркивание нужно, чтобы GridSearchCV понял, что параметр C относится именно к лог.регрессии внутри пайплайна


    gs = GridSearchCV(model_lr, params, verbose=3, scoring=f2_scorer) # по очереди подставляет в модель все возможные
    # комбинации параметров из params (в данном случае комбинаций 2*3=6)
    # verbose=3: Заставляет программу подробно писать в консоль, что она делает (какую комбинацию проверяет).
    # scoring=f2_scorer: Говорит выбирать ту комбинацию, которая показала лучший результат по метрике с beta=2
    gs.fit(X_train, y_train) # обучаем
    # После завершения объект gs будет содержать лучшую версию модели из всех возможных комбинаций.

    print(gs.best_score_, gs.best_estimator_, gs.best_params_)

    # Улучшаем дальше
    cb = CatBoostClassifier(n_estimators=100)
    cb.fit(X_train, y_train) # обучаем модель

    cb_pred = cb.predict(X_test)
    fbeta = fbeta_score(y_test, cb_pred, beta=0.5)  # вычисляем f_beta скор
    print(fbeta)

    # Подбор порога
    fbmax = -1
    thmax = -1

    cb_train_probs = cb.predict_proba(X_train)[:,1]

    for th in tqdm(np.arange(0.01, 1.01, 0.05)):
        classes = cb_train_probs > th

        fbeta = fbeta_score(y_train, classes, beta=0.5) # вычисляем f_beta скор
        if fbeta > fbmax:
          fbmax = fbeta
          thmax = th

    print(fbmax, thmax)

    cb_test_probs = cb.predict_proba(X_test)[:, 1]
    classes = cb_test_probs > thmax

    fbeta = fbeta_score(y_test, classes, beta=0.5)  # вычисляем f_beta скор
    print(fbeta)