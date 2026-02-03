import csv
import numpy as np
import pandas as pd
import seaborn as sb

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

learn_file='datasets/diamonds_learn.csv'
predict_file='datasets/diamonds_act_small.csv'


def read_file_generator(file):
    with open(file, 'r', newline='') as file:
        for row in file:
            yield row

def count_rows(file):
    with open(file, 'r', encoding='utf-8') as file:
        count = sum(1 for line in file) - 1

    return count

def count_columns(file):
    with open(file, 'r', newline='') as file:
        reader = csv.reader(file)
        count = len(next(reader))

    return count

if __name__ == '__main__':
    RANDOM_STATE = 42

    """
    ### Задание 1 ###
          Примечания:
           - Goood заменен на Good
           - строки с пустыми или нулевыми значениями удалены, т.к., имхо, нерепрезентавивы
    """

    lf = pd.read_csv(learn_file)
    pf = pd.read_csv(predict_file)
    print(lf.head())

    print('Выборка для обучения: ', count_rows(learn_file), ' строк ', count_columns(learn_file), ' столбцов')

    X = lf[['carat', 'depth', 'table', #TODO 'x', 'y', 'z',
            'cut_Fair', 'cut_Good', 'cut_Ideal', 'cut_Prem', 'cut_V_good',
            'color_D', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J',
            'clarity_I1', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2'
        ]]
    y = lf['price']

    # Для графика зависимости цены от каратов
    carat_values = lf['carat']
    price_values = lf['price']

    plt.scatter(carat_values, price_values)
    plt.xlabel('Carat')
    plt.ylabel('Price')
    plt.title('Carat vs. Price')
    # plt.show()

    """
    ### Задание 2 ###
    """

    corr = X.corr()
    sb.heatmap(corr, cmap="Blues", annot=True)
    # plt.show()

    """
    ### Задание 3 ###
    """

    X2 = lf[['carat', 'depth', 'table',
            'cut_Fair', 'cut_Good', 'cut_Ideal', 'cut_Prem', 'cut_V_good',
            'color_D', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J',
            'clarity_I1', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2',
             'price'
        ]]

    corr = X2.corr()
    sb.heatmap(corr, cmap="Blues", annot=True)
    #plt.show()

    # Примечание: наиборльшая корреляция - с каратами

    """
    ### Задание 4 ### Обучите линейную регрессию с параметрами по умолчанию на тренировочных данных 
    и сделайте предсказание на тестовых данных. 
    
    Примечания:
      - в train файлике небольшой набор данных, т.к. очень много времени заняла их подготовка
    """

    Xtest = pf[['carat', 'depth', 'table', #TODO 'x', 'y', 'z',
            'cut_Fair', 'cut_Good', 'cut_Ideal', 'cut_Prem', 'cut_V_good',
            'color_D', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J',
            'clarity_I1', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2'
        ]]
    ytest = pf['price']

    print('Выборка для прогноза: ', count_rows(predict_file), ' строк ', count_columns(predict_file), ' столбцов')

    # Для графика зависимости цены от каратов
    carat_values = lf['carat']
    price_values = lf['price']

    model = LinearRegression() # модель с параметрами по умолчанию
    model.fit(X, y)  # обучаем
    ypred = model.predict(Xtest)  # предсказываеь

    results = pd.DataFrame({'Actual': ytest, 'Predicted': ypred})
    print(results)

    plt.figure(figsize=(8, 6))
    sb.scatterplot(x=ytest, y=ypred)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red', lw=2)  # Линия идеального предсказания
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    #plt.show()

    """
    ### Задание 5 ### Вычислите значение метрик MSE и RMSE на тестовых данных. Ответ округлите до десятых.
    """
    # Метрики
    print(f"R2 Score: {r2_score(ytest, ypred):.2f}")
    print(f"MSE: {mean_squared_error(ytest, ypred):.2f}")
    print(f"RMSE: {root_mean_squared_error(ytest, ypred):.2f}")

    # R2 Score: -27.00
    # MSE: 837610.95
    # RMSE: 915.21