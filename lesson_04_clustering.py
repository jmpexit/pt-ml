"""
На сегодняшнем семинаре будем говорить про кластеризацию и про визуализацию. Также поговорим про метрики
качества кластеризации.

Из методов кластеризации рассмотрим:
k-means
DBSCAN
"""

from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pylab as pl
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from sklearn.cluster import MiniBatchKMeans
from scipy.stats import randint


"""# Кластеризация
Сгенерируем точки из трех кластеров"""

X = np.zeros((150, 2))

np.random.seed(seed=42)
X[:50, 0] = np.random.normal(loc=0.0, scale=.3, size=50)
X[:50, 1] = np.random.normal(loc=0.0, scale=.3, size=50)

X[50:100, 0] = np.random.normal(loc=2.0, scale=.5, size=50)
X[50:100, 1] = np.random.normal(loc=-1.0, scale=.2, size=50)

X[100:150, 0] = np.random.normal(loc=-1.0, scale=.2, size=50)
X[100:150, 1] = np.random.normal(loc=2.0, scale=.5, size=50)

pl.figure(figsize=(12,8))
pl.scatter(X[...,0], X[...,1], s=50, cmap='viridis')
pl.xlabel('x')
pl.ylabel('y')

#Применим kmeans

kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(X)

print(kmeans.labels_) #список из номеров кластеров для каждого объекта обучающей выборки

pl.figure(figsize=(12,8))
pl.scatter(X[:,0], X[:,1], c=kmeans.labels_, s=50, cmap='viridis')
pl.xlabel('x')
pl.ylabel('y')

"""Что произойдет, если подобрать неверное число кластеров?"""

pl.figure(figsize= (15,8))
for n_c in range(2,8):
    kmeans = KMeans(n_clusters = n_c)
    kmeans = kmeans.fit(X)
    clusters = kmeans.predict(X)
    pl.subplot(2,3,n_c - 1)
    pl.scatter(X[:,0], X[:,1], c = clusters)
    pl.title('n_clusters = {}'.format(n_c))
    print('n=', n_c, 'score:', silhouette_score(X, clusters))

pl.show()

"""
## DBSCAN
(Density-based spatial clustering of applications with noise)

Это алгоритм, основанный на плотности — если дан набор объектов в некотором пространстве, алгоритм группирует вместе 
объекты, которые расположены близко и помечает как выбросы объекты, которые находятся в областях с малой плотностью 
(ближайшие соседи которых лежат далеко).

Алгоритм имеет два основных гиперпараметра:

eps — радиус рассматриваемой окрестности
min_samples — число соседей в окрестности

Для выполнения кластеризации DBSCAN точки делятся на основные точки, достижимые по плотности точки и выпадающие 
следующим образом:
- Точка p является основной точкой, если по меньшей мере min_samples точек находятся на расстоянии, не превосходящем eps 
от неё. Говорят, что эти точки достижимы прямо из p.
- Точка q прямо достижима из p, если точка q находится на расстоянии, не большем eps, от точки p, и p — основная точка. 
Точка q достижима из p, если имеется путь p1,…,pn где p1=p и pn=q , а каждая точка pi+1 достижима прямо из pi 
(все точки на пути должны быть основными, за исключением q).

Все точки, не достижимые из основных точек, считаются выбросами.

Теперь, если p является основной точкой, то она формирует кластер вместе со всеми точками (основными или неосновными), 
достижимыми из этой точки. Каждый кластер содержит по меньшей мере одну основную точку. Неосновные точки могут быть 
частью кластера, но они формируют его «край», поскольку не могут быть использованы для достижения других точек.
"""

"""
Рассмотрим диаграмму, параметр min_samples=4.

Точка A и другие красные точки являются основными точками, поскольку область с радиусом eps , окружающая эти точки, 
содержит по меньшей мере 4 точки (включая саму точку). Поскольку все они достижимы друг из друга, точки образуют 
один кластер. Точки B и C основными не являются, но достижимы из A (через другие основные точки), и также принадлежат 
кластеру. Точка N является точкой шума, она не является ни основной точкой, ни доступной прямо.

Автор https://commons.wikimedia.org/wiki/User:Chire
https://creativecommons.org/licenses/by-sa/3.0/

Визуализация работы DBSCAN https://www.google.com/url?q=https%3A%2F%2Fwww.naftaliharris.com%2Fblog%2Fvisualizing-dbscan-clustering%2F
"""

#Посмотрим на результаты кластеризации при разном выборе параметров eps и min_samples.


pl.figure(figsize= (15,23))
i = 1
for samples in [2, 4, 8]:
    for e in [0.1, 0.2, 0.5, 1, 2]:
        dbscan = DBSCAN(eps=e, min_samples=samples)
        clusters = dbscan.fit_predict(X)
        pl.subplot(6, 3, i)
        pl.scatter(X[:,0], X[:,1], c = clusters)
        pl.title('eps = {}, min_samples = {}'.format(e, samples))
        try:
            print('eps=',e,'n=',samples,'score:',silhouette_score(X, clusters))
        except ValueError:
            print('eps=',e,'n=',samples,'score:',-1)
        i += 1
    i+=1

pl.show()


"""Пример: кластеризация игроков NBA"""

nba = pd.read_csv("https://raw.githubusercontent.com/CaioBrighenti/nba-models/master/season-stats-totals/2019.csv")
nba.head()

print(nba.columns)

# Применим kmeans с 5ю кластерами только к числовым столбцам (объявим алгоритм и обучим его с помощью метода fit)

numeric_cols = nba._get_numeric_data().dropna(axis=1)

kmeans = KMeans(n_clusters=5, random_state=3)
kmeans.fit(numeric_cols)

"""Визуализируем кластеры

Посмотрим, какое смысловое значение несут кластеры.
Визуализируем точки в осях nba['pts'] (total points) и nba['ast'] (total assistances) и раскрасим их в цвета кластеров.
Визуализируем точки в осях nba['age'] (age) и nba['mp'] (minutes played) и раскрасим их в цвета кластеров.
Подпишем оси.

"""

pl.figure(figsize=(12,8))

pl.scatter(nba['PTS'], nba['AST'], c=kmeans.labels_, s=50, cmap='viridis')
pl.xlabel('points')
pl.ylabel('assistances')

pl.scatter(nba['Age'], nba['MP'], c=kmeans.labels_, s=50, cmap='viridis')
pl.xlabel('age')
pl.ylabel('minutes played')


"""### Инициализация центроидов

Метод `sklearn.KMeans` содержит параметры `n_init` (число запусков из различных начальных приближений) и `init`. 
Есть три способа инициализации центроидов:
- `k-means++` – "умная" инициализация центроидов для ускорения сходимости.
- `random` – случайная инициализация центроидов.
- `ndarray` – заданная инициализация центроидов.

"""


"""## Сжатие изображений с K-means"""

img = mpimg.imread('datasets/duck.jpg')[..., 1]
plt.figure(figsize = (15, 9))
plt.axis('off')
plt.imshow(img, cmap='gray')

X = img.reshape((-1, 1)) #вытягиваем картинку в вектор
k_means = MiniBatchKMeans(n_clusters=15)
k_means.fit(X)
values = k_means.cluster_centers_ # усредненный цвет
labels = k_means.labels_

img_compressed = values[labels].reshape(img.shape) #возвращаем к размерности исходной картинки

plt.figure(figsize = (15, 9))
plt.axis('off')
plt.imshow(img_compressed, cmap = 'gray')


"""
Задание:
Возьмите любую фотографию (можно работать с той же) и подберите минимальное число кластеров, которое визуально не 
ухудшает качество изображения.
"""