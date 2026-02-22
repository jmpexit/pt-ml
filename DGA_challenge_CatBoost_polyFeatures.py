import gc
import re
import joblib
import numpy as np
import os
import pandas as pd
import seaborn as sb

from catboost import CatBoostClassifier, Pool
from math import log2
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer, mean_squared_error, \
    root_mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

import math
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures, StandardScaler



if __name__ == '__main__':

    # Список подозрительных доменных зон (TLD)
    SUSPICIOUS_TLDS = {
        '.xyz', '.top', '.pw', '.icu', '.club', '.info', '.biz', '.loan',
        '.zip', '.mov', '.site', '.online', '.website', '.su', '.work',
        '.today', '.bid', '.monster', '.quest', '.beauty', '.best', '.click'
    }

    def get_consecutive_counts(name):
        """Returns the max length of consecutive vowels and consonants."""
        if not name:
            return 0, 0
        vowels = "aeiou"

        # Max consecutive vowels
        v_matches = re.findall(r'[aeiouy]+', name)
        max_v = len(max(v_matches, key=len)) if v_matches else 0

        # Max consecutive consonants (letters that aren't vowels)
        c_matches = re.findall(r'[bcdfghjklmnpqrstvwxz]+', name)
        max_c = len(max(c_matches, key=len)) if c_matches else 0

        return max_v, max_c


    def get_digit_transitions(name):
        """Counts how many times the string switches from letter to digit or vice versa."""
        # Example: 'abc123de' -> switches at 'c1' and '3d'. Transitions = 2.
        transitions = 0
        for i in range(len(name) - 1):
            if (name[i].isdigit() and name[i + 1].isalpha()) or \
                    (name[i].isalpha() and name[i + 1].isdigit()):
                transitions += 1
        return transitions


    def has_meaningful_word(name):
        """
        Checks if the name contains common English words.
        On 17M rows, using a full dictionary is slow.
        Use a small set of very common 3-4 letter words instead.
        """
        common_words = {'com', 'net', 'web', 'mail', 'shop', 'blog', 'news', 'test', 'info', 'link'}
        # Check if any common word is a substring
        for word in common_words:
            if word in name:
                return 1
        return 0

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
        parts = domain.rsplit('.', 1)
        name = parts[0]

        max_v, max_c = get_consecutive_counts(name)
        digit_trans = get_digit_transitions(name)
        has_word = has_meaningful_word(name)

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
            # vow_cons_dep,
            # ent_name_len_dep,
            # ent_name_dom_dep,
            # ent_name_let_dep,
            # digit_dep,
            # digit_dep2,
            max_v,           # High in DGA if vowels clump
            max_c,           # VERY High in DGA (e.g., 'zrtpql')
            digit_trans,     # High in DGA (e.g., 'abc12def3')
            has_word
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
  #  "unique_chars_ratio",
   # "is_suspicious_tld",
  #  "has_digit_seq",
    "hyphen_count",
    "dot_count",
   # "digit_ratio_value",
  #  "double_hyphen_count",
    "vow_cons_dep",
    "ent_name_len_dep",
    "ent_name_dom_dep",
    "ent_name_let_dep"
  #  "digit_dep",
  #  "digit_dep2"
    ]

    """
    # DGA data + PolynomialFeatures
    # """
    train = pd.read_csv("datasets/dga_train.csv")
    test = pd.read_csv("datasets/dga_test.csv")
    train = train.sample(5_000_000, random_state=42)
    train = train.drop_duplicates()

    X_train = np.array([
      extract_features(str(d))
      for d in tqdm(train["domain"], desc="Extracting train features")
    ])
    y_train = train["label"].values

    X_test = np.array([
      extract_features(str(d))
      for d in tqdm(test["domain"], desc="Extracting test features")
    ])

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train).astype('float32')

    # Освобождаем память от старого X
    del X_train
    gc.collect()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly).astype('float32')

    del X_train_poly
    gc.collect()

    X_test_poly = poly.transform(X_test).astype('float32')
    X_test_scaled = scaler.transform(X_test_poly).astype('float32')

    del X_test
    del X_test_poly
    gc.collect()

    print(f"Итоговая матрица: {X_train_scaled.shape}, размер в памяти: {X_train_scaled.nbytes / 1024 ** 3:.2f} ГБ")

    cb = CatBoostClassifier(
      iterations=5000,
      depth=10,  # Глубина 10 с полиномами может быть слишком тяжелой для VRAM
      learning_rate=0.01,
      l2_leaf_reg=5,
      task_type="GPU",
      devices='0',
      bootstrap_type='Bayesian',
      random_seed=42,
      random_strength=2,
      bagging_temperature=1,      # Adds diversity to trees
      od_type='Iter',            # Overfit detector.  tells CatBoost to monitor the error
      od_wait=200,  # If the score doesn't improve for N iterations in a row, stop training
      verbose=100
    )

    cb.fit(X_train_scaled, y_train)  # обучаем модель

    # Сохраненяем модель + инструменты трансформации
    cb.save_model("dga_poly_model.cbm")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(poly, "poly.pkl")

    cb_train_probs = cb.predict_proba(X_train_scaled)[:, 1]
    fbmax = -1
    thmax = -1

    for th in tqdm(np.arange(0.01, 1.01, 0.05), desc="Поиск порога"):
      current_classes = cb_train_probs > th
      current_fbeta = fbeta_score(y_train, current_classes, beta=0.5)  # вычисляем f_beta скор

      if current_fbeta > fbmax:
          fbmax = current_fbeta
          thmax = th

    print(f"Лучший F0.5 на Train: {fbmax:.4f} при пороге: {thmax}")

    # if X_test_scaled.shape[1] != cb.feature_count_:
    #     raise ValueError(f"Ошибка! У модели {cb.feature_count_} признаков, а у теста {X_test_scaled.shape[1]}")

    cb_test_probs = cb.predict_proba(X_test_scaled)[:, 1] # Получаем вероятности для теста
    y_pred_classes = (cb_test_probs > thmax).astype(int) # Применяем найденный порог

    test["label"] = y_pred_classes.astype(int)  # Сохраняем результат в DataFrame
    test[["id", "label"]].to_csv("submission_CatBoost_poly_drop_doubles.csv", index=False)

    """
    Samples + PolynomialFeatures
    """
  #   data = pd.read_csv("datasets/dga_train.csv")
  # #  data = data.sample(2000000, random_state=42)
  #   train, test = train_test_split(data, test_size=0.2, random_state=42)
  #
  #   X_train = np.array([
  #       extract_features(str(d))
  #       for d in tqdm(train["domain"], desc="Extracting train features")
  #   ])
  #   y_train = train["label"].values
  #
  #   X_test = np.array([
  #       extract_features(str(d))
  #       for d in tqdm(test["domain"], desc="Extracting test features")
  #   ])
  #
  #   y_test = test["label"].values
  #
  #   poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
  #   X_train_poly = poly.fit_transform(X_train).astype('float32')
  #
  #   # Освобождаем память от старого X
  #   del X_train
  #   gc.collect()
  #
  #   scaler = StandardScaler()
  #   X_train_scaled = scaler.fit_transform(X_train_poly).astype('float32')
  #
  #   del X_train_poly
  #   gc.collect()
  #
  #   X_test_poly = poly.transform(X_test).astype('float32')
  #   X_test_scaled = scaler.transform(X_test_poly).astype('float32')
  #
  #   del X_test
  #   del X_test_poly
  #   gc.collect()
  #
  #   print(f"Итоговая матрица: {X_train_scaled.shape}, размер в памяти: {X_train_scaled.nbytes / 1024 ** 3:.2f} ГБ")
  #
  #   cb = CatBoostClassifier(
  #       iterations=5000,
  #       depth=10,  # Глубина 10 с полиномами может быть слишком тяжелой для VRAM
  #       learning_rate=0.03,
  #       l2_leaf_reg=5,
  #       task_type="GPU",
  #       devices='0',
  #       bootstrap_type='Bayesian',
  #       random_seed=42,
  #       random_strength=2,
  #       bagging_temperature=1,      # Adds diversity to trees
  #       od_type='Iter',            # Overfit detector.  tells CatBoost to monitor the error
  #       od_wait=200,  # If the score doesn't improve for N iterations in a row, stop training
  #       verbose=100
  #   )
  #
  #   cb.fit(X_train_scaled, y_train)  # обучаем модель
  #
  #   # Сохраненяем модель + инструменты трансформации
  #   cb.save_model("dga_poly_model.cbm")
  #   joblib.dump(scaler, "scaler.pkl")
  #   joblib.dump(poly, "poly.pkl")
  #
  #   cb_train_probs = cb.predict_proba(X_train_scaled)[:, 1]
  #   fbmax = -1
  #   thmax = -1
  #
  #   for th in tqdm(np.arange(0.01, 1.01, 0.05), desc="Поиск порога"):
  #       current_classes = cb_train_probs > th
  #       current_fbeta = fbeta_score(y_train, current_classes, beta=0.5)  # вычисляем f_beta скор
  #
  #       if current_fbeta > fbmax:
  #           fbmax = current_fbeta
  #           thmax = th
  #
  #   print(f"Лучший F0.5 на Train: {fbmax:.4f} при пороге: {thmax}")
  #
  #   # if X_test_scaled.shape[1] != cb.feature_count_:
  #   #     raise ValueError(f"Ошибка! У модели {cb.feature_count_} признаков, а у теста {X_test_scaled.shape[1]}")
  #
  #   cb_test_probs = cb.predict_proba(X_test_scaled)[:, 1] # Получаем вероятности для теста
  #   y_pred_classes = (cb_test_probs > thmax).astype(int) # Применяем найденный порог
  #
  #   fbeta_test = fbeta_score(y_test, y_pred_classes, beta=0.5)  # вычисляем f_beta скор
  #   print(f"F0.5 на Test: {fbeta_test:.4f}")
  #
  #   test['id'] = test.index
  #   test["label"] = y_pred_classes.astype(int)  # Сохраняем результат в DataFrame
  #   test[["id", "label"]].to_csv("submission_CatBoost_train_poly.csv", index=False)
