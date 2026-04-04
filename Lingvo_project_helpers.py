import os
from huggingface_hub import snapshot_download
from envs import HF_TOKEN


""" Скачиваем модель """
# Убираем предупреждения
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"--- Начинаю загрузку модели {model_name} ---")
print("Это займет время. Если прогресс замрет — не выключай, он прогрузится.")

try:
    # snapshot_download скачивает всю папку целиком со всеми конфигами. мощный инструмент в библиотеке, он сам разберется, какие файлы нужны.
    path = snapshot_download(
        repo_id=model_name,
        token=HF_TOKEN,
        repo_type="model",
        resume_download=True, # Позволяет докачивать при обрывах
        max_workers=1         # Качаем в один поток, чтобы не вешать сеть. часто загрузка виснет, потому что Windows пытается открыть 8 соединений сразу, и антивирус/брандмауэр это блокирует. В один поток медленнее, но стабильнее.
    )
    print(f"\n[✓] Готово! Модель сохранена в: {path}")
except Exception as e:
    print(f"\n[!] Ошибка при загрузке: {e}")
