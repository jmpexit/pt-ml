import os
import re
import torch
import easyocr  # <--- Замена Paddle
from spellchecker import SpellChecker
from tqdm.auto import tqdm

# --- ВАШИ ПУТИ (СОХРАНЕНЫ) ---
BASE_PATH = r"C:\Users\Julie\PycharmProjects\pt-ml\datasets\lingvo"
train_path = r"C:\Users\Julie\PycharmProjects\pt-ml\datasets\lingvo\train\train_set"
test_path = r"C:\Users\Julie\PycharmProjects\pt-ml\datasets\lingvo\test"
OUTPUT_FILE = os.path.join(BASE_PATH, "submission_final.csv")


class BatchEnglishExtractor:
    def __init__(self):
        print("--- Инициализация ---")

        # 1. Используем PyTorch для проверки GPU (так как он у вас точно работает)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print(f"[✓] GPU обнаружена: {torch.cuda.get_device_name(0)}")
        else:
            print("[x] GPU не найдена. Работаю на CPU.")

        # 2. Инициализация EasyOCR (вместо PaddleOCR)
        # verbose=False убирает лишний шум в консоли
        self.reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)

        # 3. Инициализация словаря
        self.spell = SpellChecker()

    def _is_valid_english(self, text):
        """Фильтр 'Двойной замок': Unicode + Словарь"""
        clean_text = re.sub(r'[^\w\s]', '', text).strip()

        # Пропускаем мусор и короткие слова
        if not clean_text or clean_text.isdigit() or len(clean_text) < 2:
            return False

        # Шаг Б: Unicode-ловушка (Кириллица)
        if re.search(r'[\u0400-\u04ff]', text):
            return False

        # Шаг В: Словарная проверка
        words_to_check = [clean_text, clean_text.lower()]
        # known() возвращает список найденных слов. Если список не пуст -> True
        return bool(self.spell.known(words_to_check))

    def process_folder(self, folder_path, output_file):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        if not os.path.exists(folder_path):
            print(f"ОШИБКА: Папка {folder_path} не найдена.")
            return

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        print(f"Найдено изображений: {len(files)}")

        with open(output_file, "w", encoding="utf-8") as f_out:
            # Используем tqdm для красивой полоски прогресса
            for filename in tqdm(files, desc="Обработка"):
                img_path = os.path.join(folder_path, filename)

                try:
                    # EasyOCR возвращает список кортежей: (bbox, text, confidence)
                    result = self.reader.readtext(img_path)

                    if not result:
                        continue

                    # Сбор данных
                    page_words = []
                    for (bbox, text, prob) in result:
                        # EasyOCR bbox формат: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        # Берем y1 (верхний левый угол) для сортировки
                        y_coord = bbox[0][1]
                        page_words.append({"text": text, "y": y_coord})

                    # Сортировка сверху вниз (как в вашем оригинале)
                    page_words.sort(key=lambda x: x["y"])

                    # Фильтрация и запись
                    for item in page_words:
                        word = item["text"]
                        if self._is_valid_english(word):
                            f_out.write(word + "\n")

                except Exception as e:
                    print(f"\nОшибка в файле {filename}: {e}")

        print(f"\nГотово! Все слова сохранены в: {output_file}")


"""Очистка и сбор статистики"""
from collections import Counter
import pandas as pd

def analyze_frequency(input_file, top_n=10):
    """Считает частотность и выводит статистику в консоль."""
    with open(input_file, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file if line.strip()]

    # Считаем частотность
    word_counts = Counter(words)

    # Сортируем по популярности (от самых частых к редким)
    common_words = word_counts.most_common()

    print(f"--- Статистика файла: {input_file} ---")
    print(f"Всего слов (с повторами): {len(words)}")
    print(f"Уникальных слов: {len(word_counts)}")
    print(f"\nТоп-{top_n} частых слов:")

    for word, count in common_words[:top_n]:
        print(f"{word}: {count}")

    return common_words


def remove_top_noise(input_file, noise_count=7):
    """Находит топ N самых частых слов и удаляет их из файла полностью."""
    with open(input_file, 'r', encoding='utf-8') as file:
        all_words = [line.strip() for line in file if line.strip()]

    # Находим, какие именно слова являются шумом
    word_counts = Counter(all_words)
    noise_words = set(word for word, count in word_counts.most_common(noise_count))
    # Когда делаем if word not in noise_words, Python в случае со списком каждый раз пробегает по нему.
    # В случае с set (множеством) поиск происходит мгновенно. На 600 словах разницы нет, но на 10 000+
    # это станет заметно.

    print(f"Удаляем шум: {', '.join(noise_words)}")

    # Оставляем только те слова, которых нет в списке шума
    filtered_words = [word for word in all_words if word not in noise_words]

    with open(input_file, 'w', encoding='utf-8') as f:
        for word in filtered_words:
            f.write(f"{word}\n")

    print(f"Файл {input_file} очищен от шума")
    return filtered_words # Опционально

def remove_duplicates(input_file):
    """Читает файл и возвращает список уникальных слов в порядке их появления."""
    with open(input_file, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file if line.strip()]

    # Использование dict.fromkeys сохраняет порядок первого появления слова
    unique_words = list(dict.fromkeys(words))

    with open(input_file, 'w', encoding='utf-8') as f:
        for word in unique_words:
            f.write(f"{word}\n")

    print(f"Файл {input_file} успешно обновлен.")

    return unique_words # Опционально

# --- ЗАПУСК ---
if __name__ == "__main__":
    # """ Распознаем слова """
    # # Проверка существования папки перед запуском
    # if not os.path.exists(test_path):
    #     try:
    #         os.makedirs(test_path)
    #         print(f"Папка '{test_path}' создана. Пожалуйста, положите туда картинки.")
    #     except OSError:
    #         print(f"Не удалось создать папку {test_path}. Проверьте пути.")
    # # Формируем файл со словами
    # else:
    #     extractor = BatchEnglishExtractor()
    #     extractor.process_folder(test_path, OUTPUT_FILE)
    #
    # """ Подготовка таблицы уникальных слов """
    # remove_top_noise(OUTPUT_FILE, noise_count=7)
    # remove_duplicates(OUTPUT_FILE)
    # analyze_frequency(OUTPUT_FILE, top_n=20)
    pass

"""
Technically, "I" (the AI you are talking to now) am a large-scale model hosted on Google's servers. You cannot 
"download" or "load" me directly into your local Python script like a file from Hugging Face because my full 
architecture is too massive for a home PC.
However, you have three ways to get that same "deep dive" style of translation in your project:
1. The "Big Model" API (Recommended for Quality)
Using an API (OpenAI, Anthropic, or Google Gemini) is the best way to achieve the same level of detail, including 
historical context and examples.
How: Use the openai or google-generativeai library instead of transformers.
Why: These models can understand prompts such as: "Translate this word, give 3 usage examples, and explain the 
etymology or historical context."
"""