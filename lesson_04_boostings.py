"""
Градиентный бустинг
Catboost vs. LightGBM vs. XGBoost
Сравнение моделей на искусственном примере
CatBoost для решения задачи
Интерпретация модели
Блендинг и стекинг

В прошлый раз мы посмотрели простую версию градиентного бустинга из scikit-learn, придуманную в 1999 году Фридманом.
Прогресс не стоит на месте, и на сегодняшний день есть три популярные библиотеки с разными имплементациями градиентного
бустинга, которые на практике показывают лушие результаты:

XGBoost. Появилась в 2014 году, статья автора вышла в 2016. После выхода быстро набрала популярность и оставалась
стандартом до конца 2016 года. Об особенностях данной библиотеки рассказывалось на лекции.
CatBoost от компании Яндекс с релизом в 2017 году. Алгоритм можно запускать с дефолтными гиперпараметрами, потому что
он является менее чувствительным к выбору их конкретных значений. Отлично умеет работать с категориальным признаками,
при этом автоматически обрабатывая полученные на вход непредобработанные фичи.
LightGBM. Релиз в один год с Catboost, библиотека от Microsoft. Отличается очень быстрым построением композиции.
Например, при построении узла дерева, вместо перебора по всем значениям признака, производится перебор значений
гистограммы этого признака. Таким образом, вместо  O(N)  требуется  O (m), где  m  - число бинов гистограммы.
В отличие от других библиотек, строит деревья в глубину, при этом на каждом шаге строит вершину, дающую наибольшее
уменьшение функционала.

        "|Критерий|Catboost|Lightgbm|Xgboost|\n",
        "|--|--|--|--|\n",
        "|Год релиза|2014|2017|2017|\n",
        "|Построение деревьев|симметрично по уровням|в глубину|асимметрично по уровням до максимальной глубины с прунингом|\n",
        "|Параметры контроля переобучения|learning_rate, depth, l2-leaf-reg (аналога min_child_weigth нет) |learning_rate, max_depth, num_leaves, min_data_in_leaf|learning_rate (eta), min_child_weigth, max_depth|\n",
        "|Контроль скорости обучения|rsm, iterations|feature_fraction, bagging_fraction, num_iterations|n_estimator, colsample_bytree, subsample|\n",
        "|Параметры категориальных фичей|cat_features, one_hot_max_size|categorical_feature|не доступно|\n",
        "|Бинаризация признаков|сетка выбирается заранее|-|перебор всех границ, выбор сетки на каждой итерации|\n",
        "|Скор сплита|Похожесть векторов градиентов |-| Смотрим на изменение функции ошибки|\n",
        "|Bootstrap|Можно перевзвешивать и менять интенсивность |-|-|\n",
        "|Рандомизация скора сплита|+ |-|-|\n"

Основные параметры
objective – функция ошибки для настройки композиция
learning_rate / eta – скорость обучения
n_estimators / num_iterations – число итераций градиентного бустинга

Настройка сложности деревьев
max_depth – максимальная глубина
max_leaves / num_leaves – максимальное число вершин в дереве
gamma / min_gain_to_split – порог на уменьшение функции ошибки при расщеплении в дереве
min_data_in_leaf – минимальное число объектов в листе
min_sum_hessian_in_leaf – минимальная сумма весов объектов в листе, минимальное число объектов, при котором делается
расщепление
lambda – коэффициент регуляризации (L2)
subsample / bagging_fraction – какую часть объектов обучения использовать для построения одного дерева
colsample_bytree / feature_fraction – какую часть признаков использовать для построения одного дерева

Начать настройку можно с самых главных параметров: learning_rate и n_estimators. Один из них фиксируем, оставшийся из
этих двух параметров подбираем (например, подбираем скорость обучения при n_estimators=100). Следующий параметр по
важности - max_depth, так как мы хотим неглубокие деревья (в Catboost и LightGBM) для снижения переобучения.

Техническое отступление

Данные библиотеки необходимо сначала устанавливать (можно через pip / conda или brew, если Вы работаете на MAC OS).
Чтобы у Вас точно вопроизводился ноутбук и не было проблем из-за несовпадающих версий библиотек, рекомендуется через
python создавать виртуальную среду. Подробнее см. файлики техническое_отступление.md и requirements.txt.

"""


import warnings
import catboost
import lightgbm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = (8, 5)

"""
Catboost и другие бустинги для искусственного примера
В алгоритме сделаны улучшения и выбор разных опций для борьбы с переобучением, подсчету сркднего таргета на отложенной 
выборке, подсчету статистик по категориальным фичам, бинаризацией фичей, рандомизации скора сплита, разные типы 
бутсрапирования.

Cначала зафиксируем все гиперпараметры со значениями по умолчанию, кроме количества деревьев в композиции - 
n_estimators.
"""

def plot_surface(X, y, clf):
    h = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Добавим на график сами наблюдения
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=2,
                           flip_y=0.05, class_sep=0.8, random_state=241)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=241)

catboost = CatBoostClassifier(n_estimators=300, logging_level='Silent')
catboost.fit(X_train, y_train)
plot_surface(X_test, y_test, catboost)

print(roc_auc_score(y_test, catboost.predict_proba(X_test)[:, 1]))

"""GradientBoost"""
n_trees = [1, 5, 10, 100, 200, 300, 400, 500, 600, 700, 1000, 2000]
quals_train = []
quals_test = []
for n in n_trees:
    boost = GradientBoostingClassifier(n_estimators=n)
    boost.fit(X_train, y_train)
    q_train = roc_auc_score(y_train, boost.predict_proba(X_train)[:, 1])
    q_test = roc_auc_score(y_test, boost.predict_proba(X_test)[:, 1])
    quals_train.append(q_train)
    quals_test.append(q_test)

plt.plot(n_trees, quals_train, marker='o', label='train')
plt.plot(n_trees, quals_test, marker='o', label='test')
plt.xlabel('Number of trees')
plt.ylabel('AUC-ROC')
plt.legend()
plt.show()

"""CatBoost"""
n_trees = [1, 5, 10, 100, 200, 300, 400, 500, 600, 700, 1000, 2000]
quals_train = []
quals_test = []
for n in n_trees:
    catboost = CatBoostClassifier(iterations=n, logging_level='Silent')
    catboost.fit(X_train, y_train)
    q_train = roc_auc_score(y_train, catboost.predict_proba(X_train)[:, 1])
    q_test = roc_auc_score(y_test, catboost.predict_proba(X_test)[:, 1])
    quals_train.append(q_train)
    quals_test.append(q_test)

plt.plot(n_trees, quals_train, marker='o', label='train')
plt.plot(n_trees, quals_test, marker='o', label='test')
plt.xlabel('Number of trees')
plt.ylabel('AUC-ROC')
plt.legend()
plt.show()

"""
Xgboost
Базовый алгоритм приближает направление, посчитанное с учетом второй производной функции потерь
Функционал регуляризуется – добавляются штрафы за количество листьев и за норму коэффициентов
При построении дерева используется критерий информативности, зависящий от оптимального вектора сдвига
Критерий останова при обучении дерева также зависит от оптимального сдвига

Ссылка на источник https://github.com/esokolov/ml-course-hse/blob/master/2021-fall/lecture-notes/lecture11-ensembles.pdf
"""
xgb = XGBClassifier(n_estimators=300, verbosity=0)

xgb.fit(X_train, y_train)
plot_surface(X_test, y_test, xgb)

print(roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))

n_trees = [1, 5, 10, 100, 200, 300, 400, 500, 600, 700, 1000, 2000]
quals_train = []
quals_test = []
for n in n_trees:
    xgboost = XGBClassifier(n_estimators=n, verbosity=0)
    xgboost.fit(X_train, y_train)
    q_train = roc_auc_score(y_train, xgboost.predict_proba(X_train)[:, 1])
    q_test = roc_auc_score(y_test, xgboost.predict_proba(X_test)[:, 1])
    quals_train.append(q_train)
    quals_test.append(q_test)

plt.figure(figsize=(8, 5))
plt.plot(n_trees, quals_train, marker='.', label='train')
plt.plot(n_trees, quals_test, marker='.', label='test')
plt.xlabel('Number of trees')
plt.ylabel('AUC-ROC')
plt.legend()

plt.show()

#Видно, что переобучились - качество на тесте только падает.

"""LightGBM"""
lightgbm = LGBMClassifier(n_estimators=300)
lightgbm.fit(X_train, y_train)
plot_surface(X_test, y_test, lightgbm)

print(roc_auc_score(y_test, lightgbm.predict_proba(X_test)[:, 1]))

n_trees = [1, 5, 10, 100, 200, 300, 400, 500, 600, 700, 1000, 2000]
quals_train = []
quals_test = []
for n in n_trees:
    lightgbm = LGBMClassifier(n_estimators=n)
    lightgbm.fit(X_train, y_train)
    q_train = roc_auc_score(y_train, lightgbm.predict_proba(X_train)[:, 1])
    q_test = roc_auc_score(y_test, lightgbm.predict_proba(X_test)[:, 1])
    quals_train.append(q_train)
    quals_test.append(q_test)

plt.figure(figsize=(8, 5))
plt.plot(n_trees, quals_train, marker='o', label='train')
plt.plot(n_trees, quals_test, marker='o', label='test')
plt.xlabel('Number of trees')
plt.ylabel('AUC-ROC')
plt.legend()

plt.show()

#В целом, у LightGBM получилась та же проблема с переобучением, как у Xgboost.
# Нужно дальше подбирать гиперпараметры для этих двух.

"""
Попробуем взять фиксированное количество деревьев n_estimators, но будем менять их максимальную глубину max_depth. 
У этих алгоритмов разное время обучения, поэтому возьмем какой-то небольшой диапазон глубины и сравним все три 
модели - Catboost, LightGBM, Xgboost.
"""

def plot_model_diff_depths(model=LGBMClassifier, depth_range=list(range(1, 5)), n_trees=10):
    roc_auc_train = []
    roc_auc_test = []
    for i in depth_range:
        clf = model(n_estimators=n_trees, max_depth=i)
        if type(clf) == type(CatBoostClassifier()):
            clf = CatBoostClassifier(n_estimators=n_trees, max_depth=i, logging_level="Silent")
        clf.fit(X_train, y_train)
        q_train = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        q_test = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        roc_auc_train.append(q_train)
        roc_auc_test.append(q_test)

    plt.figure(figsize=(7, 5))
    plt.plot(depth_range, roc_auc_train, marker='o', label='train')
    plt.plot(depth_range, roc_auc_test, marker='o', label='test')
    plt.title(f'{model}')
    plt.xlabel('Depth')
    plt.ylabel('AUC-ROC')
    plt.legend()

    plt.show()

    plot_model_diff_depths(model=LGBMClassifier, depth_range=list(range(1, 16, 2)), n_trees=100)
    plot_model_diff_depths(model=CatBoostClassifier, depth_range=list(range(1, 16, 2)), n_trees=100)
    plot_model_diff_depths(model=XGBClassifier, depth_range=list(range(1, 16, 2)), n_trees=100)

    #Когда мы обучили лучшие версии моделей, можно их сохранить и использовать для получения предсказаний, например,
    # на новом батче данных.

# Сохранить
lightgbm.booster_.save_model('lightgbm.txt')
catboost.save_model('catboost.cbm', format='cbm')
xgboost.save_model('xgboost.json')

# Загрузить
lightgbm = LGBMClassifier(model_file='mode.txt')
catboost = catboost.load_model('catboost.cbm')
xgboost = xgboost.load_model('xgboost.json')


"""## Catboost для решения задачи + интерпретация признаков"""
data = pd.read_csv("https://raw.githubusercontent.com/Murcha1990/ML_AI24/refs/heads/main/Lesson9_ClassificationBase/bike_buyers_clean.csv")

print(data.head())

X = data.drop(['ID','Purchased Bike'], axis=1)
y = data['Purchased Bike']

X.head()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=42)

model1 = CatBoostClassifier(cat_features = [0, 1, 4, 5, 6, 8, 9])  # one_hot_max_size=10)
model1.fit(Xtrain, ytrain)

pred = model1.predict(Xtest)

accuracy_score(ytest, pred)

params = {'max_depth' : np.arange(10,20,3),
          'learning_rate' : [0.01, 0.05, 0.1],
          'one_hot_max_size' : [10, 50, 100]}

gs = GridSearchCV(CatBoostClassifier(n_estimators=100, cat_features = [0, 1, 4, 5, 6, 8, 9], verbose=0), params, cv=3, scoring='accuracy', verbose=2)

gs.fit(Xtrain, ytrain)

gs.best_score_, gs.best_params_