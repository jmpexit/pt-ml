import csv
import numpy as np
import os
import pandas as pd
import seaborn as sb

from catboost import CatBoostClassifier
from math import log2
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer, mean_squared_error, \
    root_mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import math
from collections import Counter


if __name__ == '__main__':
    # Список подозрительных доменных зон (TLD)
    SUSPICIOUS_TLDS = {'.xyz', '.top', '.pw', '.icu', '.club', '.info', '.biz', '.loan'}


    def get_entropy(text):
        """Вычисляет энтропию Шеннона (степень хаотичности строки)."""
        if not text:
            return 0
        counts = Counter(text)
        probs = [count / len(text) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def digit_ratio(name):
        length = len(name)
        digits = sum(c.isdigit() for c in name)
        return digits / max(1, length)


    def extract_features(domain):
        domain = str(domain).lower().strip()

        # Разделяем домен и TLD (например, 'google.com' -> 'google', '.com')
        parts = domain.rsplit('.', 1)
        name = parts[0]
        tld = '.' + parts[1] if len(parts) > 1 else ''

        # Считаем базовые показатели
        domain_len = len(domain)
        name_len = len(name)
        letters = sum(c.isalpha() for c in name)

        # 1. Цифры и спецсимволы
        digit_count = sum(c.isdigit() for c in name)
        hyphen_count = name.count('-')
        double_hyphen_count = name.count('--') # for punycode
        dot_count = domain.count(".")
        digit_ratio_value = digit_ratio(name)  # доля цифр


        # 2. признаки "читаемости"
        vowels = "aeiou"
        vowel_count = sum(c in vowels for c in name)
        # Согласные (только буквы, не цифры)
        consonant_count = sum(c.isalpha() and c not in vowels for c in name)

        # Соотношение (защита от деления на ноль)
        vowel_ratio = vowel_count / name_len if name_len > 0 else 0

        consonant_ratio = consonant_count / name_len if name_len > 0 else 0

        # Добавим разницу: в DGA согласных обычно НАМНОГО больше, чем гласных
        diff_vowels_consonants = (consonant_count - vowel_count) / name_len if name_len > 0 else 0
        vow_cons_dep = diff_vowels_consonants * consonant_ratio

        # 3. Энтропия (САМЫЙ ВАЖНЫЙ ПРИЗНАК для DGA)
        entropy = get_entropy(name)
        ent_name_len_dep = entropy * name_len
        ent_name_dom_dep = entropy * domain_len
        ent_name_let_dep = entropy * letters

        # 4. Уникальность символов
        unique_chars_ratio = len(set(name)) / name_len if name_len > 0 else 0

        # 5. Подозрительный TLD (1 если зона из списка, иначе 0)
        is_suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0

        # 6. Наличие последовательных цифр (часто бывает в DGA)
        has_digit_seq = 1 if any(name[i:i + 2].isdigit() for i in range(len(name) - 1)) else 0
        digit_dep = digit_count * has_digit_seq
        digit_dep2 = digit_ratio_value * digit_count

        return [
            domain_len,
            name_len,
            letters,
            digit_count,
            vowel_ratio,
            consonant_ratio,
            diff_vowels_consonants,
            entropy,
            unique_chars_ratio,
            is_suspicious_tld,
            has_digit_seq,
            hyphen_count,
            dot_count,
            digit_ratio_value,
            double_hyphen_count,
            vow_cons_dep,
            ent_name_len_dep,
            ent_name_dom_dep,
            ent_name_let_dep,
            digit_dep,
            digit_dep2
        ]

    feature_names  = [
    "domain_len",
    "name_len",
    "letters",
    "digit_count",
    "vowel_ratio",
    "consonant_ratio",
    "diff_vowels_consonants",
    "entropy",
    "unique_chars_ratio",
    "is_suspicious_tld",
    "has_digit_seq",
    "hyphen_count",
    "dot_count",
    "digit_ratio_value",
    "double_hyphen_count",
    "vow_cons_dep",
    "ent_name_len_dep",
    "ent_name_dom_dep",
    "ent_name_let_dep",
    "digit_dep",
    "digit_dep2"
    ]

    train = pd.read_csv("datasets/dga_train.csv")
    test = pd.read_csv("datasets/dga_test.csv")

    train = train.sample(800_000, random_state=42)

    # train = pd.read_csv("/kaggle/input/dga-domain-detection-challenge-i/train.csv")
    # test = pd.read_csv("/kaggle/input/dga-domain-detection-challenge-i/test.csv")

    X_train = np.array([
        extract_features(str(d))
        for d in tqdm(train["domain"], desc="Extracting train features")
    ])
    y_train = train["label"].values

    X_test = np.array([
        extract_features(str(d))
        for d in tqdm(test["domain"], desc="Extracting test features")
    ])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=0,
        verbose=100
    )

    # model = make_pipeline( # Technically, Random Forest (like CatBoost) is not sensitive to the scale of data
    #     StandardScaler(),
    #     RandomForestClassifier(
    #         n_estimators=1000,
    #         random_state=0
    #     )
    # )

    model.fit(X_train, y_train)
    importances = model.feature_importances_
    print("Важность признаков:", model.feature_importances_)

    # y_test_pred = model.predict(X_test)  # предсказываем на валидационной выборке
    #
    # test["label"] = y_test_pred.astype(int)
    # test[["id", "label"]].to_csv("submission_RandomForest.csv", index=False)


    """
    Важность признаков:
        "domain_len",
    "name_len",
    "letters",
    "digit_count",
    "vowel_ratio",
    "consonant_ratio",
    "diff_vowels_consonants",
    "entropy",
    "unique_chars_ratio",
    "is_suspicious_tld",
    "has_digit_seq",
    "hyphen_count",
    "dot_count",
    "digit_ratio_value",
    "double_hyphen_count",
    "vow_cons_dep",
    "ent_name_len_dep",
    "ent_name_dom_dep",
    "ent_name_let_dep",
    "digit_dep",
    "digit_dep2"
    
    [0.05102867 0.07186947 0.04580813 0.01835111 0.10909196 0.07831181
 0.12076449 0.07343849 0.02575232 0.00894956 0.00322539 0.05058474
 0.03442163 0.01287467 0.00079844 0.0777936  0.07220107 0.05725997
 0.06579355 0.00648774 0.01519317]

    
    """


    """
    Sample
    """
    # data = pd.read_csv("datasets/dga_train.csv")
    # data = data.sample(900_000, random_state=42)
    # train, test = train_test_split(data, test_size=0.2, random_state=42)
    #
    # X_train = np.array([
    #     extract_features(str(d))
    #     for d in tqdm(train["domain"], desc="Extracting train features")
    # ])
    # y_train = train["label"].values
    #
    # X_test = np.array([
    #     extract_features(str(d))
    #     for d in tqdm(test["domain"], desc="Extracting test features")
    # ])
    # y_test = test["label"].values
    #
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    #
    # model = RandomForestClassifier(
    #     n_estimators=100,
    #     n_jobs=-1,
    #     random_state=0,
    #     verbose=100
    # )
    #
    # model.fit(X_train, y_train)
    # importances = model.feature_importances_
    # print("Важность признаков:", model.feature_importances_)
    #
    # y_test_pred = model.predict(X_test) #.astype(int)  # предсказываем на валидационной выборке
    # fbeta = fbeta_score(y_test, y_test_pred, beta=0.5)  # вычисляем f_beta скор
    # print(fbeta)
    #
    # test['id'] = test.index
    # test["label"] = y_test_pred.astype(int)
    # test[["id", "label"]].to_csv("submission_RandomForest_sample.csv", index=False)
