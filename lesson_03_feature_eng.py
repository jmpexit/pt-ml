import pandas as pd

from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from ydata_profiling import ProfileReport


""" Продолжение задачи DGA. Оптимизируем модель """

if __name__ == '__main__':
    data = pd.read_csv('datasets/pokemon.csv')
    profile_report = ProfileReport(data, title='Pandas Profiling Report')

    profile_report.to_file('Pokemon_report.html')

    """ Работа с признаками """

    print(data.sample(10))

    # fillna and drop useless cols

    print(data.isnull().sum())
    data['Type 2'] = data['Type 2'].fillna('No 2nd type')

    data.drop(columns=['#', 'Name'], inplace=True)

    X = data.drop(columns='Legendary')
    y = data['Legendary'].astype('int')

    y.value_counts(normalize=True)

    # define cat_cols

    cat_cols = ['Type 1', 'Type 2']

    default_pipeline = Pipeline([
        ('cat_encoder_', LeaveOneOutEncoder(cols=cat_cols)),
        ('scaler_', StandardScaler()),
        ('model_', LogisticRegression())]
    )

    cv_res1 = cross_validate(default_pipeline,
                             X,
                             y,
                             cv=5,
                             scoring='f1',
                             n_jobs=-1,
                             return_train_score=True
                             )

    print(cv_res1['test_score'].mean())

    """ # Make pipeline more complicated """
    pipe_dif = Pipeline([
        ('cat_encoder_', LeaveOneOutEncoder(cols=cat_cols)),
        ('poly_featurizer_', PolynomialFeatures(degree=4)),
        ('scaler_', StandardScaler()),
        ('model_', LogisticRegression())]
    )
    cv_res2 = cross_validate(pipe_dif,
                             X,
                             y,
                             cv=5,
                             scoring='f1',
                             n_jobs=-1,
                             return_train_score=True
                             )

    print(cv_res2, cv_res2['train_score'].mean(), cv_res2['test_score'].mean())

    """ # Introduce feature selectors """
    data_tr = pipe_dif[:-1]

    X_tr = data_tr.fit_transform(X, y)
    print(f'data shape after transformation is {X_tr.shape}')

    # 1k признаков - многовато, добавим в пайплайн селектор

    """ 
    Фильтрационные методы
    
    Суть таких методов в том, чтобы для каждого признака посчитать некоторую метрику "связи" с целевым признаком. 
    И в результате оставить топ-K признаков согласно выбранной метрике.
    Самые популярные такие метрики:
     - статистика хи-квадрат
     - метрика mutual information
     """

    # k_best = 30

    pipe = Pipeline([
        ('cat_encoder_', LeaveOneOutEncoder(cols=cat_cols)),
        ('poly_featurizer_', PolynomialFeatures(degree=4)),
        ('scaler_', StandardScaler()),
        ('selector_', SelectKBest(score_func=mutual_info_classif, k=100)),
        ('model_', LogisticRegression())]
    )

    cv_res = cross_validate(pipe, X, y, cv=5, scoring='f1', return_train_score=True)
    print(cv_res)

    # k best нужно подбирать
    print(cv_res['test_score'].mean(), cv_res['train_score'].mean())

    """ Жадный метод отбора """
    k_best = 50
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=k_best, step=30)
    print(X_tr.shape)

    res = rfe.fit_transform(X_tr, y)
    print(res.shape, res)

    pipe_rfe = Pipeline([
        ('cat_encoder_', LeaveOneOutEncoder(cols=cat_cols)),
        ('poly_featurizer_', PolynomialFeatures(degree=4)),
        ('scaler_', StandardScaler()),
        ('selector_', RFE(LogisticRegression(max_iter=1000),
                          n_features_to_select=20,
                          step=30
                          )),
        ('model_', LogisticRegression())]
    )

    cv_res3 = cross_validate(pipe_rfe, X, y, cv=5, scoring='f1', return_train_score=True)
    print(cv_res3, cv_res3['test_score'].mean(), cv_res3['train_score'].mean())

    """ С помощью модели """

    sel = SelectFromModel(LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear'), threshold=0.1)

    # пример

    res = sel.fit_transform(X_tr, y)
    print(res.shape, res)

    pipe_lasso =  Pipeline([
        ('cat_encoder_', LeaveOneOutEncoder(cols=cat_cols)),
        ('poly_featurizer_', PolynomialFeatures(degree=4)),
        ('scaler_', StandardScaler()),
        ('selector_', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'), threshold=1e-5)),
        ('model_', LogisticRegression())]
    )

    cv_res4 = cross_validate(pipe_lasso, X, y, cv=5, scoring='f1', return_train_score=True)
    print(cv_res4, cv_res4['test_score'].mean())



