"""
Полносвязные и сверточные нейронные сети
План семинара
- Учимся работать с картинками средствами Python, NumPy и PyTorch
- Применяем константные свёртки к изображениям
- Сравниваем работу полносвязных и свёрточных сетей на датасете MNIST

https://colab.research.google.com/github/Murcha1990/ML_Course_PT/blob/main/Lecture5_IntroDL_CV/PT_CNN.ipynb
"""

import warnings

import matplotlib.pyplot as pyplot
import seaborn as sns

import numpy as np
from matplotlib import cm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

"""1. Разбираемся что такое картинка"""

# при работе в колабе, скачайте изображение этой командой
#wget https://raw.githubusercontent.com/hushchyn-mikhail/CourseraDL/main/cnn/screencast_1/butterfly.jpg

from PIL import Image

img = Image.open("datasets/butterfly.jpg")
print(f"Image format: {img.format}; shape: {img.size}; color scheme: {img.mode}")
img.show()

"""
## Матричное представление изображения

На самом деле каждая картинка это набор пикселей. Если мы попросим питон показать нам картинку, он покажет матрицу 
из чисел.  Каждому пикселю в этой матрице соответствует число. Это число сообщает нам о том, насколько этот пиксель 
яркий. Яркость можно измерять в разных шкалах. В нашем случае она измеряется по шкале от 0 до 1.

Цветное изображение состоит из 3 числовых матриц или трехмерного тензора. Каждая матрица соответствует одному из 3 
базовых цветов: красному, зеленому и синему. Такой формат хранения картинки называется [RGB-форматом]
(https://www.wikiwand.com/ru/RGB)
"""

# преобразуем изображение в массив
img_matrix = np.array(img)

# (высота, ширина, число каналов)
print(f"Image array shape: {img_matrix.shape}")

pyplot.imshow(img_matrix)

# посмотрим на все каналы изображения отдельно
pyplot.imshow(img_matrix[:, :, 0], cmap=cm.Reds)
pyplot.show()

pyplot.imshow(img_matrix[:, :, 1], cmap=cm.Greens)
pyplot.show()

pyplot.imshow(img_matrix[:, :, 2], cmap=cm.Blues)
pyplot.show()

"""
Все действия по редактированию картинки сводятся к математике. Например, чтобы осветлить картинку, нужно прибавить 
к каждому пикселю какое-то число. Часто такие математические действия над картинками записывают в виде операции 
свёртки. Свёртка принимает на вход одну картинку, а на выход отдаёт новую, переработанную.
"""

"""
# 2. Пробуем применить свёртки к картинке

**Необязательное задание:** один из семенирстов собрал [коллекцию ручных задачек на свёртки.]
(https://fulyankin.github.io/deep_learning_masha_book/problem_set_05_conv/problem_01.html) 
Чтобы лучше почувствовать, как работают разные части свёрточных сеток, можно попробовать порешать эти задачки. 
К каждой из них на страничке есть решение.
"""

"""### Класс torch.nn.Conv2d"""

import torch.nn as nn

nn.Conv2d

"""
В **PyTorch** свёрточный слой представлен в модуле `torch.nn` классом 
[`Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) с параметрами:
- `in_channels`: количество входных каналов
- `out_channels`: количество выходных каналов
- `kernel_size`: размер ядра
- `stride`: шаг
- `padding`: паддинг
- `padding_mode`: режим паддинга  (`'zeros'`, `'reflect'` и др.)
- `dilation`: растяжение

#### `kernel_size`
**Размер ядра**. `int`, если ядро квадратное и кортеж из двух чисел, если ядро прямоугольное.
Задает размер фильтра, с которым производится свёртка изображения.

**`kernel_size=3`**
см. datasets/no_padding_no_strides.gif

Эта и следующие анимации взяты [здесь](https://github.com/vdumoulin/conv_arithmetic).

#### `stride`
**Шаг**. Задает шаг, в пикселях, на который сдвигается фильтр. `int`, если по горизонтали и вертикали сдвигается 
на одно и то же число. Кортеж из двух чисел, если сдвиги разные.

**`stride=2`**
см. datasets/no_padding_strides.gif

#### `padding`
**Паддинг**. Количество пикселей, которыми дополняется изображение. Аналогично шагу и размеру ядра, может быть, 
как `int`, так и кортежем из двух чисел.

**`padding=1`**
см. datasets/same_padding_no_strides.gif
"""


"""### Класс MaxPool2d
В **PyTorch** уменьшает размерность, сохраняя наиболее «выразительные» признаки (максимумы), представлен в модуле 
`torch.nn` классом [`MaxPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) с параметрами:
- `kernel_size`: размер ядра
- `stride`: шаг
- `padding`: паддинг (по умолчанию 0)
- `padding_mode`: режим паддинга  (`'zeros'`, `'reflect'` и др.)
- `dilation`: растяжение
см. datasets/Screenshot-from-2017-08-15-17-04-02.png
"""

nn.MaxPool2d

"""### Класс Flatten
В **PyTorch** операция выпрямления (преобразования многомерного тензора в вектор) представлена в модуле torch.nn 
классом [`Flatten`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Flatten.html).
см. datasets/Flatten.png
"""

import torch
from torchsummary import summary
#nn.Flatten

# Например, батч из 2 картинок 28x28 с 3 каналами
input_tensor = torch.randn(2, 3, 28, 28)
flatten = nn.Flatten()
output = flatten(input_tensor)

print(f"До: {input_tensor.shape}") # torch.Size([2, 3, 28, 28])
print(f"После: {output.shape}")    # torch.Size([2, 2352]) (3*28*28 = 2352)

model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.Flatten(), nn.Linear(16*26*26, 10))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the active device

summary(model, (3, 28, 28))

"""## Свёртка изображений

Чем может быть полезна свертка при работе с изображениями? Свертки детектируют **паттерны на картинках** – цвета и сочетания цветов, небольшие объекты. Обычно значения свертки являются обучаемыми параметрами нейрости. Однако существуют "готовые" свертки, настроенные на определенные паттерны.

Например, оператор Собеля (свертка с определенными параметрами) используется для детекции границ на изображении. Применим этот оператор. Для этого пока не будем пользоваться классом `torch.nn.Conv2d`, а возьмём соответствующую функцию из модуля `torch.nn.functional`.

"""
