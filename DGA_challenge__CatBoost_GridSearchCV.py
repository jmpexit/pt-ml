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

import math
from collections import Counter


"""
https://www.kaggle.com/competitions/dga-domain-detection-challenge-i/overview

Detecting DGA Domains

Welcome to the DGA Domain Detection competition! This competition focuses on detecting algorithmically generated 
second-level domains (SLDs) within domain names. Your task is to build a machine learning model that can distinguish 
between legitimate and DGA-generated SLDs.

Goal
The goal of this competition is to detect algorithmically generated second-level domains (SLDs), which are often used 
by malware to evade detection. Participants will be provided with a dataset of domain names labeled as DGA (1) 
or legitimate (0). Note that in some samples, only the SLD is provided, without a top-level domain (TLD). 
Your task is to build a model that predicts the correct label for each domain in the test set.
Since false positives (predicting a legitimate domain as DGA) can cause serious issues, it is more important to 
minimize false positives than false negatives.

Who Can Participate
This competition is designed for students in Using machine learning algorithms to solve cybersecurity problems 
course— but anyone is welcome to participate and test their skills. Prior experience with classification tasks and 
Python programming will be helpful, but the problem is approachable with basic ML knowledge.

Files
train.csv – the training set containing domain names and labels.
test.csv – the test set containing domain names without labels (to be predicted).
sample_submission.csv – a sample submission file in the correct format, typically with columns example_id and label.

Columns
domain – the domain name to classify. Examples: 0000ad264572a083d3863cc42d97037b.co.cc, example.com.
label – the target variable for classification:
    1 – DGA (algorithmically generated) domain
    0 – legitimate domain
    
Task Description
In the Domain Name System (DNS), every website or network service is identified by a domain name — a human-readable 
address that maps to an IP address. A typical domain consists of several hierarchical parts, separated by dots:
    Top-Level Domain (TLD): the rightmost part (e.g., .com, .org, .net)
    Second-Level Domain (SLD): the main identifying part, located before the TLD (e.g., example in example.com)
    Subdomains: optional prefixes that further divide a domain (e.g., mail.example.com)
When malware communicates with its command-and-control (C2) servers, it often needs to hide these connections 
from defenders. To do this, many malicious programs use Domain Generation Algorithms (DGAs) — algorithms that 
automatically create large numbers of pseudo-random domain names every day. By registering only a few of these domains, 
attackers can ensure their malware always finds an active C2 domain, while defenders struggle 
to block them all in advance.

How DGAs work
A DGA typically uses various system or time-based parameters to generate domain strings, such as:
    Current date or timestamp
    A hardcoded secret seed
    Mathematical operations (hashing, bitwise shifts, modulo, etc.)
    Character sets or wordlists

The result is a domain that looks random, but is deterministically generated from the algorithm.
For example: Legitimate - google.com, wikipedia.org, news.bbc.co.uk - Readable, often based on real words, 
brand-related, meaningful DGA-generated - xj3k9sd.info, qertuvap.net, aolqjxcz.biz - Random-looking, 
lacks semantic meaning, may vary daily or hourly
Some samples in this competition include only the SLD part (e.g., xj3k9sd) without specifying a TLD. 
This is because DGAs usually generate second-level domains, while the final registration can happen under any TLD.

What makes a good DGA algorithm
A well-designed DGA should:
    Produce unique domains over time (to avoid reusing blocked ones)
    Be hard to predict without knowing the seed or algorithm
    Generate syntactically valid domain names (only allowed characters)
    Maintain operational resilience, ensuring malware can always reach its C2 server

Your task in this competition is to detect whether a given domain (or SLD) was generated algorithmically (DGA) 
or is a legitimate one. This requires understanding linguistic patterns, randomness, and statistical properties 
of domain strings.

Submission File
For each ID in the test set, you must predict a probability for the label variable. 
The file should contain a header and have the following format:
    id,label 0,1 1,1 2,1 3,1 4,0 5,1 etc.

"""


if __name__ == '__main__':
    # VOWELS = set("aeiou")  # множество гласных букв
    #
    # def shannon_entropy(s):
    #     probs = [s.count(c) / len(s) for c in set(s)]
    #     return -sum(p * log2(p) for p in probs)
    #
    # def digit_ratio(name):
    #     length = len(name)
    #     digits = sum(c.isdigit() for c in name)
    #     return digits / max(1, length)
    #
    # def vowel_ratio(name):
    #     letters = sum(c.isalpha() for c in name)
    #     vowels = sum(c in VOWELS for c in name)
    #     return vowels / max(1, letters)
    #
    # def extract_features(domain):
    #     name = domain.split('.')[0]
    #
    #     length = len(name)  # длина
    #     digits = sum(c.isdigit() for c in name)  # количество цифр
    #     has_digits_flag = digits > 0  # есть ли цифры
    #     letters = sum(c.isalpha() for c in name)  # количество букв
    #     entropy = shannon_entropy(name) if length > 0 else 0  # энтропия
    #     has_dash_flag = "-" in name  # есть ли дефис
    #     dash_count = name.count("-")  # количество дефисов
    #     digit_ratio_value = digit_ratio(name)  # доля цифр
    #     dot_count = domain.count(".")  # количество точек
    #     vowel_ratio_value = vowel_ratio(name)  # доля гласных
    #     uniqueness = len(set(name)) / len(name) if len(name) > 0 else 0, # Коэффициент уникальности букв
    #
    #     features = [
    #         length,
    #         digits,
    #         int(has_digits_flag),
    #         letters,
    #         entropy,
    #         int(has_dash_flag),
    #         dash_count,
    #         digit_ratio_value,
    #         dot_count,
    #         vowel_ratio_value,
    #         uniqueness
    #     ]
    #
    #     return features

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
        double_hyphen_count = name.count('--')
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

        # 3. Энтропия (САМЫЙ ВАЖНЫЙ ПРИЗНАК для DGA)
        entropy = get_entropy(name)

        # 4. Уникальность символов
        unique_chars_ratio = len(set(name)) / name_len if name_len > 0 else 0

        # 5. Подозрительный TLD (1 если зона из списка, иначе 0)
        is_suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0

        # 6. Наличие последовательных цифр (часто бывает в DGA)
        has_digit_seq = 1 if any(name[i:i + 2].isdigit() for i in range(len(name) - 1)) else 0

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
            double_hyphen_count
        ]

    train = pd.read_csv("datasets/dga_train.csv") #pd.read_csv("/kaggle/input/dga-domain-detection-challenge-i/train.csv")
    test =  pd.read_csv("datasets/dga_test.csv") # pd.read_csv("/kaggle/input/dga-domain-detection-challenge-i/test.csv")
    train = train.sample(800_000, random_state=42)

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

    #y_test = test["label"].values # в данных чаленжа нет values в тестовой выборке, поэтому скипаем

    # ### Вариант 1 обучаем модель логистической регрессии (КЛАССИФИКАЦИЯ!)
    #
    # model_lr = make_pipeline(
    #     StandardScaler(),  # масштабируем признаки
    #     LogisticRegression(max_iter=1000, random_state=0)
    # )
    # model_lr.fit(X_train, y_train) # fit последовательно запускает обучение для каждого звена:
    # # Для StandardScaler: заставляет вычислить среднее и отклонение признаков, определяет разброс значений в данных
    # # Для LogisticRegression: заставляет подобрать веса
    #
    # test["label"] = model_lr.predict(X_test).astype(int)  # предсказываем на валидационной выборке и добавляем столбец
    # test[["id", "label"]].to_csv("submission_log_regress.csv", index=False)
    # y_predict = test[["domain", "label"]].to_csv("predict_log_regress.csv", index=False) # [optional] датасет с предсказанными значениями

    ### Вариант 2 - алгоритм градиентного бустинга

    cb = CatBoostClassifier(
        n_estimators=300, # TODO n_estimators для первичного алгоритма достаточно 100-200
        task_type="GPU",
        devices='0',
        random_seed=42,
        verbose=100
    )
    # n_estimators - количество деревьев, которые построит алгоритм.
    # Каждое следующее дерево пытается исправить ошибки предыдущих (обычно используют от 500 до 2000)

    """ Используем f2_scorer"""
    f2_scorer = make_scorer(fbeta_score, beta=2)
    params = {
        'depth': [8, 10, 12],  # Глубина деревьев (чем выше, тем сложнее зависимости)
        'learning_rate': [ 0.1, 0.15],  # Скорость обучения
        'bootstrap_type': ['Bayesian', 'Bernoulli'],  # Как модель выбирает подмножества данных
        'l2_leaf_reg': [3, 5, 7],  # Регуляризация (чтобы не переобучиться)
        'random_strength': [1, 2]  # Добавляет случайности, чтобы не "зазубривать" трейн
    }
    # params = {
    #     'depth': [7, 8, 10],
    #     'learning_rate': [0.1, 0.15],
    #     'bootstrap_type': ['Bayesian', 'Bernoulli'],
    #     'l2_leaf_reg': [3, 5, 7]
    # }

    gs = GridSearchCV(  # Настраиваем поиск
        estimator=cb,  # наш CatBoost
        param_grid=params,
        scoring=f2_scorer,  # Оптимизируем F2
        cv=3,  # Кросс-валидация на 3 фолда
        verbose=3
    )

    gs.fit(X_train, y_train)  # Запускаем поиск лучших параметров, обучаем

    best_cb = gs.best_estimator_  # Это наша лучшая модель
    print(f"Лучшие параметры: {gs.best_params_}")
    print(f"Лучший F2-score: {gs.best_score_}")
    """"""

    # Подбор порога
    fbmax = -1
    thmax = -1

    cb_train_probs = best_cb.predict_proba(X_train)[:, 1]  # просим модель выдать вероятность от 0 до 1, а именно [:, 1]-
    # вероятности «положительного» класса (вредоносных доменов)

    # Поиск лучшего порога
    for th in tqdm(np.arange(0.01, 1.01, 0.05)):
        classes = cb_train_probs > th
        fbeta = fbeta_score(y_train, classes, beta=2)  # вычисляем f_beta скор  # beta=0.5
        if fbeta > fbmax:
            fbmax = fbeta
            thmax = th

    print(f"Оптимальный порог: {thmax} (F2 на train: {fbmax:.4f})")

    # применяем найденный "идеальный" порог к тестовым данным
    cb_test_probs = best_cb.predict_proba(X_test)[:, 1]
  #  final_classes = cb_test_probs > thmax

    # final_f2 = fbeta_score(y_test, final_classes, beta=2)  # вычисляем f_beta скор. в данных чаленжа нет values в тестовой выборке, поэтому скипаем
    # print(f"Итоговый F2-score на тесте: {final_f2:.4f}")

    """  Обучаем ФИНАЛЬНУЮ МОДЕЛЬ с найденными параметрами """
    best_params = gs.best_params_.copy()
    best_params.pop('n_estimators', None)

    # Создаем модель СРАЗУ с лучшими параметрами. Здесь мы объединяем найденные гиперпараметры и 2000 деревьев
    final_model = CatBoostClassifier(
        **best_params,
        n_estimators=3000,  # теперь можно и 2000-5000 для точности
        loss_function='Logloss',  # стандарт для классификации. или F1
        random_seed=42,  # правильное число
        task_type="GPU", # "CPU" / Если есть видеокарта NVIDIA, поставить "GPU" - будет в 10 раз быстрее
        devices='0',  # Индекс моей видеокарты
        thread_count=-1, # Использует все ядра без копирования данных в памяти
        verbose=100,  # будет писать лог каждые 100 деревьев, чтобы не было скучно
        # depth=12,
        # l2_leaf_reg=3,
        learning_rate=0.15,
        bootstrap_type='Bayesian',
        random_strength=2
    )



    # Обучаем ОДИН РАЗ на всех тренировочных данных
    final_model.fit(X_train, y_train)

    """ ПРЕДСКАЗЫВАЕМ на рабочей выборке """
    test_probs = final_model.predict_proba(X_test)[:, 1]  # Получаем вероятности (самый точный инструмент)
    final_pred_classes = test_probs > thmax  # Применяем найденный порог

    # """ Проверяем метрику  """  # в данных чаленжа нет values в тестовой выборке, поэтому скипаем
    # final_f2 = fbeta_score(y_test, final_pred_classes, beta=2)
    # print(f"Финальный F2 на тесте с порогом {thmax}: {final_f2:.4f}")

    """ Сохраняем результат """
    test["label"] = final_pred_classes.astype(int) # Сохраняем результат в DataFrame
    test[["id", "label"]].to_csv("submission_CatBoost_GridSearch_sample.csv", index=False)
    test[["domain", "label"]].to_csv("predict_CatBoost.csv", index=False)  # [optional] датасет с предсказанными значениями


    # Fitting 3 folds for each of 18 candidates, totalling 54 fits

    """
    Лучшие параметры: {'depth': 8, 'l2_leaf_reg': 5, 'learning_rate': 0.1}
    Лучший F2-score: 0.7511554857315512
    Оптимальный порог: 0.16000000000000003 (F2 на train: 0.8615)
    
    sample
    Лучшие параметры: {'bootstrap_type': 'Bayesian', 'depth': 12, 'l2_leaf_reg': 3, 'learning_rate': 0.15, 'random_strength': 2}
Лучший F2-score: 0.7697926314895059
Оптимальный порог: 0.16000000000000003 (F2 на train: 0.8702)
    """

    """
    Поскольку комбинаций стало больше (
     фита), GridSearchCV может занять время. Чтобы не ждать вечно:
    Используй RandomizedSearchCV: Вместо перебора всех комбинаций, он проверит, например, 20 случайных. Это в 7 раз быстрее, а результат часто такой же.
    python
    from sklearn.model_selection import RandomizedSearchCV
    
    rs = RandomizedSearchCV(cb_base, params, n_iter=20, scoring=f2_scorer, cv=3)
    rs.fit(X_train, y_train)
    """

    # # Разбиваем 7.5 млн строк на 10 частей по ~750к каждая
    # X_test_batches = np.array_split(X_test, 10)
    # test_probs = []
    #
    # for batch in tqdm(X_test_batches, desc="Predicting batches"):
    #     probs = final_model.predict_proba(batch)[:, 1]
    #     test_probs.append(probs)
    #
    # # Собираем всё в один массив
    # test_probs = np.concatenate(test_probs)
    #
    # # Применяем порог (thmax)
    # final_labels = (test_probs > thmax).astype(int)
