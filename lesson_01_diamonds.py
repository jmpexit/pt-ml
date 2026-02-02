import csv
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

test_file='datasets/diamonds_good.csv'

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
    # X = Data['data']
    # y = Data['target']


      # ### Задание 1 ###
      # Примечания:
      #  - Goood заменен на Good
      #  - строки с пустыми или нулевыми значениями удалены, т.к., имхо, нерепрезентавивы



    print(count_columns(test_file), count_rows(test_file))

