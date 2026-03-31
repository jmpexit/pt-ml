import matplotlib.pyplot as pyplot
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings
from PIL import Image
from matplotlib import cm
from torch import nn
from torch.nn.functional import conv2d
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms as T
from torchvision.datasets import MNIST
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.get_device_name(0)}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[✓] Using device: {device}")

"""Обучаем полносвязную нейросеть"""

#Подгрузим данные
transform = torchvision.transforms.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,)),
])

# train_set = MNIST('.MNIST', transform=transform, train=True, download=True)
# val_set = MNIST('.MNIST', transform=transform, train=False, download=True)
# img, target = val_set[0]
# pyplot.imshow(img.squeeze(), cmap='gray')
# pyplot.title(f"Метка: {target}")
# pyplot.show()

train_set = MNIST('datasets/lingvo/train/train_set', transform=transform, train=True, download=True)
val_set = MNIST('datasets/lingvo/test', transform=transform, train=False, download=True)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size     = 3,
    shuffle        = False,
    num_workers    = 0
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size    = 3,
    shuffle       = False,
    num_workers   = 0
)

#Подготовим функцию для отрисовки процесса обучения.
from IPython.display import clear_output
def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    clear_output()
    fig, axs = pyplot.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    pyplot.show()

#Опишем функции для обучения и валидации модели
def training_epoch(model, optimizer, criterion, train_loader, tqdm_desc):
    """Одна эпоха обучения"""
    train_loss, train_accuracy = 0.0, 0.0
    model.train()
    for images, labels in tqdm(train_loader, desc=tqdm_desc):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.shape[0]
        train_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy

@torch.no_grad()
def validation_epoch(model, criterion, val_loader, tqdm_desc):
    """Прогнозы на валидации"""
    val_loss, val_accuracy = 0.0, 0.0
    model.eval()
    for images, labels in tqdm(val_loader, desc=tqdm_desc):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        val_loss += loss.item() * images.shape[0]
        val_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)
    return val_loss, val_accuracy

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, scheduler=None):
    """Обучение модели"""
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss, val_accuracy = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # функция для смены lr по расписанию
        if scheduler is not None:
            scheduler.step()

        plot_losses(train_losses, val_losses, train_accuracies, val_accuracies)

    # печатаем метрики
    print(f"Epoch: {epoch}, loss: {np.mean(val_loss)}, accuracy: {np.mean(val_accuracy)}")

"""Соберем полносвязную сеть из 3 слоёв с 32, 16, и 10 нейронами. На всех промежуточных слоях используем `ReLU`
в качестве функции активации. Последний слой должен возвращать логиты (вероятности) для дальнейшей классификации"""

class FcNet(nn.Module):
    def __init__(self, input_shape, hide_neurons=32, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                                # превращаем картинку 28х28 в вектор размером 784
            nn.Linear(input_shape, hide_neurons),        # линейный слой, преобразующий вектор размера 784 в вектор размера 64
            nn.ReLU(),                                   # нелинейность
            nn.Linear(hide_neurons, hide_neurons // 2),  # линейный слой, преобразующий вектор размера 784 в вектор размера 32
            nn.ReLU(),                                   # нелинейность
            nn.Linear(hide_neurons//2, num_classes),     # линейный слой, преобразующий вектор размера 784 в вектор размера 10
        )

    def forward(self, x):
        return self.model(x)

IMG_SIZE = 28
NUM_EPOCH = 7

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_fc = FcNet(IMG_SIZE**2).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model_fc.parameters(), lr=0.01, momentum=0)

train(model_fc, optimizer, criterion, train_loader, val_loader, NUM_EPOCH)

# #Попробуем построить предсказания на валидационной выборке и посмотрим, на примеры картинок, где модель ошиблась.
# # Иногда бывает полезно отсматривать конкретные примеры ошибок, чтобы подумать как можно улучшить модель.
# def predict(model, val_loader):
#     X, y, logit = [], [], []
#     model.eval()
#     for batch_num, (x_batch, y_batch) in enumerate(val_loader):
#         X.extend(x_batch)
#         y.extend(y_batch)
#         x_batch = x_batch.to(device)
#
#         with torch.no_grad():
#             logits = model(x_batch)
#
#         logit.extend(torch.max(logits, 1)[1].cpu().numpy())
#     return np.array(X), np.array(y), np.array(logit)
#
# X_test, y_test, y_pred = predict(model_fc, val_loader)
#
# errors = y_test != y_pred
#
# X_err = X_test[errors]
# y_err = y_test[errors]
# y_pred_err = y_pred[errors]
#
# cols = 6
# rows = 2
# fig = pyplot.figure(figsize=(3 * cols - 1, 4 * rows - 1))
# for i in range(cols):
#     for j in range(rows):
#         random_index = np.random.randint(0, len(y_err))
#         ax = fig.add_subplot(rows, cols, i * rows + j + 1)
#         ax.grid('off')
#         ax.axis('off')
#         ax.imshow(np.transpose(X_err[random_index], (1, 2, 0)), cmap='gray')
#         ax.set_title('real_class: {} \n  predict class: {}'.format(y_err[random_index], y_pred_err[random_index]))
# pyplot.show()
#
# """# Обучаем свёрточную нейронную сеть
# По аналогии с предыдущим примером обучим свёрточную нейронную сеть. В части `encoder` зададим следующие слои:
# - Свёрточный слой с 2 ядрами размером 3
# - Функция активации ReLU
# - Уменьшить картинку в 2 раза (по каждому измерению)
# - Свёрточный слой с 4 ядрами размером 3
# - Функция активации ReLU
# - Уменьшить картинку в 2 раза (по каждому измерению)
# - Свёрточный слой с 8 ядрами размером 3
# - Функция активации ReLU
# - Уменьшить картинку в 2 раза (по каждому измерению)
#
# В части `classifier` зададим следующие слои:
# - Полносвязный слой с 32 нейронами (аккуратно осознайте сколько нейронов будет на входе и почему)
# - Функция активации ReLU
# - Выходной слой с 10 нейронами
# """
# class ConvNet(nn.Module):
#     def __init__(self, image_channels=1):
#         super().__init__()
#         self.encoder = nn.Sequential(  # 28 x 28 x 1
#             nn.Conv2d(in_channels=image_channels, out_channels=4,
#                       kernel_size=3, padding='same'),  # 28 x 28 x 4
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # 14 x 14 x 4
#
#             nn.Conv2d(in_channels=4, out_channels=8,
#                       kernel_size=3, padding='same'),  # 14 x 14 x 8
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # 7 x 7 x 8
#
#             nn.Conv2d(in_channels=8, out_channels=16,
#                       kernel_size=3, padding='same'),  # 7 x 7 x 16
#             nn.ReLU(),
#             nn.MaxPool2d(2)  # 3 x 3 x 16 = 144
#         )
#
#         self.head = nn.Sequential(
#             nn.Linear(in_features=144, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=10)
#         )
#
#     def forward(self, x):
#
#         # x: B x 1 x 28 x 28
#         out = self.encoder(x)   # out: B x 392
#         out = nn.Flatten()(out) # out: B x 128
#         out = self.head(out)    # out: B x 10
#         return out
#
#     def get_embedding(self, x):
#         out = self.encoder(x)
#         return nn.Flatten()(out)
#
# NUM_EPOCH = 10
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# model_cnn = ConvNet().to(device)
# criterion = nn.CrossEntropyLoss()
#
# optimizer = torch.optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0)
#
# #Посмотрим на ошибки и увидим, что объекты, где наша модель их допускает, более сложные,
# # чем в случае с полносвязной сетью.
# X_test, y_test, y_pred = predict(model_cnn, val_loader)
#
# errors = y_test != y_pred
#
# X_err = X_test[errors]
# y_err = y_test[errors]
# y_pred_err = y_pred[errors]
#
# cols = 6
# rows = 2
# fig = pyplot.figure(figsize=(3 * cols - 1, 4 * rows - 1))
# for i in range(cols):
#     for j in range(rows):
#         random_index = np.random.randint(0, len(y_err))
#         ax = fig.add_subplot(rows, cols, i * rows + j + 1)
#         ax.grid('off')
#         ax.axis('off')
#         ax.imshow(np.transpose(X_err[random_index], (1, 2, 0)), cmap='gray')
#         ax.set_title('real_class: {} \n  predict class: {}'.format(y_err[random_index], y_pred_err[random_index]))
# pyplot.show()

