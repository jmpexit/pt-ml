"""
# Работа с большими языковыми моделями (LLM)

pip install transformers datasets evaluate -q

Сейчас все основные SoTA-решения являются Closed-Source, и доступны только через веб-интерфейс, или через API с
жесткими ограничениями. Удобно, если нужно прогнать несколько вопросов - попробуйте сами.


- OpenAI API (через VPN) - [openai.com/api](https://openai.com/api/)
- Chatbot Arena (удобный способ попробовать топовые LLMки, но с очень строгими ограничениями) -
[chat.lmsys.org](https://chat.lmsys.org)
- YandexGPT Lite/Pro (поддерживает дообучение) -
[console.yandex.cloud](https://console.yandex.cloud/folders/b1g4lgsfdsvocob346tv/foundation-models/overview)
- GigaChat API (без дообучения) - [developers.sber.ru](https://developers.sber.ru/docs/ru/gigachat/api/overview)

Но в демо-режиме особо не разгонишься, и ничего не автоматизируешь. Для масштабного применения придется платить за
доступ к API. Как быть, если хотим классные модели, но бесплатно?

### Open-Source модели
К счастью, тут спасают модели с открытым исходным кодом. Удобнее всего их искать на
[HuggingFace](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) или на уже упомянутом
[ChatBot Arena](https://chat.lmsys.org) - во вкладке LeaderBoard искать модели с открытыми лицензиями.

Примеры:
- [ruGPT-3.5](https://huggingface.co/ai-forever/ruGPT-3.5-13B)
- [GigaChat-20B-A3B-instruct](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (требует HF token)
- [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) (требует HF token)

Будем работать со следующими моделями:
-   [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
-   [OpenChat-3.5](https://huggingface.co/openchat/openchat-3.5-0106)
-   [gpt2](https://huggingface.co/openai-community/gpt2)

### Загрузка модели
Давайте попробуем подгрузить и использовать такую модель. Не забудьте поставить GPU в среде выполнения.

### Генерация
Вспомним, что под капотом это все еще просто генеративная модель, которая предсказывает вероятности следующих токенов.
Что делать с этими вероятностями дальше - можно определить с помощью стратегии генерации.

| Стратегия | Описание | Плюсы и минусы |
| --- | --- | --- |
| Greedy Search | Выбирает слово с наивысшей вероятностью как следующее слово в последовательности. | Плюсы: Простота и скорость.<br> Минусы: Может привести к повторяющемуся и несвязному тексту. |
| Семплинг с температурой | Добавляет случайность в выбор слова. Большая температура приводит к большей случайности. | Плюсы: Позволяет исследовать и получать разнообразный результат.<br> Минусы: Высокие температуры могут привести к бессмысленным результатам. |
| Семплинг по ядру (Top-p семплинг) | Выбирает следующее слово из усеченного словаря, "ядра" слов, которые имеют суммарную вероятность, превышающую предустановленный порог (p). | Плюсы: Обеспечивает баланс между разнообразием и качеством.<br> Минусы: Настройка оптимального 'p' может быть затруднительна. |
| Beam Search | Исследует множество гипотез (последовательностей слов) на каждом шаге и сохраняет 'k' наиболее вероятных, где 'k' - ширина луча. | Плюсы: Дает более надежные результаты, чем жадный поиск.<br> Минусы: Может страдать от нехватки разнообразия и приводить к общим ответам. |
| Top-k семплинг | Случайным образом выбирает следующее слово из 'k' слов с самыми высокими вероятностями. | Плюсы: Вводит случайность, увеличивая разнообразие результатов.<br> Минусы: Случайный выбор иногда может привести к менее связному тексту. |
| Нормализация длины | Предотвращает предпочтение модели более коротких последовательностей за счет деления логарифмированных вероятностей на длину последовательности, возведенную в некоторую степень. | Плюсы: Делает более длинные и потенциально более информативные последовательности более вероятными.<br> Минусы: Настройка фактора нормализации может быть сложной. |
| Стохастический Beam Search | Вводит случайность в процесс выбора 'k' гипотез в поиске пучком. | Плюсы: Увеличивает разнообразие в сгенерированном тексте.<br> Минусы: Баланс между разнообразием и качеством может быть сложно управлять. |
| Декодирование с минимальным риском Байеса (MBR) | Выбирает гипотезу (из многих), которая минимизирует ожидаемую потерю для функции потерь. | Плюсы: Оптимизирует результат в соответствии с определенной функцией потерь.<br> Минусы: Вычислительно более сложно и требует хорошо подобранную функциию потерь. |

Референсы:
- [Документация `AutoModelForCausalLM.generate()`](https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationMixin.generate)
- [Документация `AutoTokenizer.decode()`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.decode)
- [Статья о стратегиях генерации на Huggingface](https://huggingface.co/docs/transformers/generation_strategies)
"""

import torch
import transformers

from envs import OPENROUTER_TOKEN, HF_TOKEN

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.get_device_name(0)}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[✓] Using device: {device}")

#huggingface.co/settings/tokens

# model_name = 'openchat/openchat-3.5-0106'
model_name = "Qwen/Qwen1.5-1.8B"

#tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, device_map=device)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    device_map=device
)

tokenizer.pad_token_id = tokenizer.eos_token_id

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_state_dict=True,
    token=HF_TOKEN
)

prompt = 'The first known Martian language'
batch = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(model.device) #to(device)
print("Input batch (encoded):", batch)

"""Greedy Search
Always picks the #1 most likely word. It’s logical but can be repetitive.
"""
print('\nUsing Greedy Search...')
output_tokens = model.generate(**batch, max_new_tokens=256, do_sample=False) # do_sample=False - turns off "randomness",
                                            #max_new_tokens=64: - tells to write at most 64 words/tokens
                                            #**batch: feeds the encoded version of your prompt
#print("\nOutput:", tokenizer.decode(output_tokens[0].cpu()))
print("Output:", tokenizer.decode(output_tokens[0].cpu(), skip_special_tokens=True))
# Note: Even though your model is running on your NVIDIA GPU, the tokenizer.decode() function and the standard
# Python print() function work best with data that is in your computer's system RAM (CPU).

"""# Стратегии генерации ответов"""

"""## Temperature (datasets/temper.png, datasets/temper_ex.png)
Adds "randomness." Higher temperature = more creative/risky; Lower = more focused.
Семплинг с температурой
"""
print('\nUsing sampling with temperature...')
output_tokens = model.generate(**batch, max_new_tokens=256, do_sample=True, temperature=11.7)
print("Output:", tokenizer.decode(output_tokens[0].cpu(), skip_special_tokens=True))

"""Top-K семплинг
Limits the AI to only the top top_k=N most likely next words, preventing it from picking completely nonsensical words.
"""
print('\nUsing Top-K sampling...')
output_tokens = model.generate(**batch, max_new_tokens=64, do_sample=True, temperature=4.0, top_k=50)
print("Output:", tokenizer.decode(output_tokens[0].cpu(), skip_special_tokens=True))

"""Beam Search (datasets/beam.png)
Explores multiple "paths" of sentences at once and picks the one that makes the most sense overall.
"""
print('\nUsing Beam Search...')
output_tokens = model.generate(**batch, max_new_tokens=256, do_sample=False, num_beams=5)
print("Output:", tokenizer.decode(output_tokens[0].cpu(), skip_special_tokens=True))

"""#####################"""
"""### Создание промпта
Изначально модели заточены на генерацию. Чтобы общаться с ними в привычном режиме диалога, промпт нужно отформатировать.
Правильный формат обычно указан в документации модели, но для некоторых моделей его можно восстановить с помощью 
метода apply_chat_template
"""

""" Manual Prompting: manually appending "User:" and "Assistant:" strings """
prompt = "What is your favourite english word?"
prompt += "Well, I'm not that fluent in english. But one of the most good-sounding for me is 'eloquent', I reckon it's beautiful"
prompt += "Do you have any examples with it?"

print('\nUsing prompts...')

batch = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(model.device)

output_tokens = model.generate(
    **batch,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=256
)

print("Output:", tokenizer.decode(output_tokens[0], skip_special_tokens=True))

"""### Chat template
apply_chat_template: takes a list of messages (Role/Content) and automatically formats them with the specific tags 
(like <|im_start|>) that the Qwen model was trained to understand.
"""
print('\nUsing Chat template...')
messages = [
    {"role": "user", "content": "What is your favourite english word?"},
    {"role": "assistant", "content": "Well, I'm not that fluent in english. But one of the most good-sounding for me is 'eloquent', I reckon it's beautiful"},
    {"role": "user", "content": "Do you have any examples with it?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
inputs = inputs.to(model.device)

output_tokens = model.generate(**inputs, do_sample=True, temperature=0.7, max_new_tokens=256)

print("Output:", tokenizer.decode(output_tokens[0], skip_special_tokens=True))

"""### Chain-of-Thought Reasoning
Для оптимальных промптов модели необходимо давать не только примеры ответов, но и снабжать эти примеры детально 
описанным процессом того, как прийти к этому результату, и при генерации требовать от модели того же.
"""
""" Chain-of-Thought (CoT) Reasoning
This is a "prompt engineering" trick. By providing examples where the AI shows its Rationale (step-by-step math), 
you "teach" the model to think before it gives the final answer. 
This significantly improves performance on logic and math problems.
"""

print('\nUsing Chain-of-Thought Reasoning...')
prompt = """
GPT4 Correct User:
Question: The original retail price of an appliance was 60 percent more than its wholesale cost. 
If the appliance was actually sold for 20 percent less than the original retail price, then it was sold for what 
percent more than its wholesale cost?
Answer Choices: (A) 20% (B) 28% (C) 36% (D) 40% (E) 42% <|end_of_turn|>
GPT4 Correct Assistant:
Rationale: wholesale cost = 100;\noriginal price = 100*1.6 = 160;\nactual price = 160*0.8 = 128.\nAnswer: B.
Correct Answer: B <|end_of_turn|>

GPT4 Correct User:
Question: A grocer makes a 25% profit on the selling price for each bag of flour it sells. If he sells each bag for 
$100 and makes $3,000 in profit, how many bags did he sell?
Answer Choices: (A) 12 (B) 16 (C) 24 (D) 30 (E) 40 <|end_of_turn|>
GPT4 Correct Assistant:
Rationale: Profit on one bag: 100*1.25= 125\nNumber of bags sold = 3000/125 = 24\nAnswer is C.
Correct Answer: C <|end_of_turn|>


GPT4 Correct User:
Question: 20 marbles were pulled out of a bag of only white marbles, painted black, and then put back in. Then, 
another 20 marbles were pulled out, of which 1 was black, after which they were all returned to the bag. If the 
percentage of black marbles pulled out the second time represents their percentage in the bag, how many marbles 
in total Q does the bag currently hold?
Answer Choices: (A) 40 (B) 200 (C) 380 (D) 400 (E) 3200
GPT4 Correct Assistant:
Rationale: We know that there are 20 black marbles in the bag and this number represent 1/20 th of the number of all 
marbles in the bag, thus there are total Q of 20*20=400 marbles.\nAnswer: D.
Correct Answer: D <|end_of_turn|>


GPT4 Correct User: Question: Janice bikes at 10 miles per hour, while Jennie bikes at 20. How long until they have 
collectively biked 1 mile?
Answer Choices: (A) 1 minute (B) 2 minutes (C) 3 minutes (D) 4 minutes (E) 5 minutes
GPT4 Correct Assistant:
Rationale:
""".strip()

inputs =  tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(model.device)
output_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, max_new_tokens=512)
print("Output:", tokenizer.decode(output_tokens[0].cpu(), skip_special_tokens=True))

"""Memory Management"""
import gc

del model
del tokenizer
torch.cuda.empty_cache()
gc.collect()
