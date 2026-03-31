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
                    # --- ГЛАВНОЕ ИЗМЕНЕНИЕ ---
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


# --- ЗАПУСК ---
if __name__ == "__main__":
    # Проверка существования папки перед запуском
    if not os.path.exists(test_path):
        try:
            os.makedirs(test_path)
            print(f"Папка '{test_path}' создана. Пожалуйста, положите туда картинки.")
        except OSError:
            print(f"Не удалось создать папку {test_path}. Проверьте пути.")
    else:
        extractor = BatchEnglishExtractor()
        extractor.process_folder(test_path, OUTPUT_FILE)



###для очистки и сбора статистики:
# from collections import Counter
# import pandas as pd
#
# def clean_and_analyze(input_file):
#     # Читаем все слова из вашего файла
#     with open(input_file, 'r', encoding='utf-8') as f:
#         words = [line.strip() for line in f if line.strip()]
#
#     # Считаем частотность
#     word_counts = Counter(words)
#
#     # Сортируем по популярности (от самых частых к редким)
#     common_words = word_counts.most_common()
#
#     print(f"Всего слов распознано: {len(words)}")
#     print(f"Уникальных слов: {len(word_counts)}")
#     print("\nТоп-10 самых частых слов:")
#     for word, count in common_words[:10]:
#         print(f"{word}: {count}")
#
#     # Сохраняем чистый список (без дублей) для GPT
#     unique_words = [word for word, count in common_words]
#
#     return unique_words, common_words
#
# # Пример использования:
# # unique_list, full_stats = clean_and_analyze(OUTPUT_FILE)
