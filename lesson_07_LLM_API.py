from envs import OPENROUTER_TOKEN
"""
# Доступ к LLM по API и базовый инференс

# Шаг 1: Получаем доступ к LLM по API и учимся с ней общаться

## Будем работать с моделями (LLM) через API OpenRouter
OPENROUTER_TOKEN возьмем с сайта openrouter в личном кабинете.

pip install langchain langchain-openai langchain_community
"""

#Подключимся к модели DeepSeek
#OPENROUTER_TOKEN = "" # My test key
# сгенерировать на https://openrouter.ai/

from langchain_openai import ChatOpenAI

deepseek_llm = ChatOpenAI(
    api_key=OPENROUTER_TOKEN,
    base_url="https://openrouter.ai/api/v1",
    model = "stepfun/step-3.5-flash:free"
#    model="deepseek/deepseek-r1-0528:free"
#     default_headers={
#         "HTTP-Referer": "http://localhost:3000",  # Your site URL
#         "X-Title": "Local Test Script"  # Your app name
#     }
)

deepseek_response = deepseek_llm.invoke("Привет! Как дела?").content
print(deepseek_response)

# #Пообщаемся с моделью и выясним, есть ли у нее память?
question1 = "Как корректно перевести фразу с английского 'lead did not pan out'?" #question1 = "Сколько лап у паука?"
additional_question = "А на французский? (с транскрипцией)"
additional_question_2 = "А какие есть аналоги?"

# print(question1)
# print(deepseek_llm.invoke(question1).content)
# print(additional_question)
# print(deepseek_llm.invoke(additional_question).content)

#Оформим диалог красиво
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

question1_message = HumanMessage(content=question1)
additional_question_message = HumanMessage(content=additional_question)
additional_question_message_2 = HumanMessage(content=additional_question_2)

# answer1_message = AIMessage(content=deepseek_llm.invoke([question1_message]).content)
# answer2_message = AIMessage(content=deepseek_llm.invoke([additional_question_message]).content)
# answer3_message = AIMessage(content=deepseek_llm.invoke([additional_question_message_2]).content)

# question1_message.pretty_print()
# answer1_message.pretty_print()
# additional_question_message.pretty_print()
# answer2_message.pretty_print()
# additional_question_message_2.pretty_print()
# answer3_message.pretty_print()

#Памяти - нет. Организуем ее самостоятельно!

dialogue = []
dialogue.append(question1_message)
dialogue.append(AIMessage(content=deepseek_llm.invoke(dialogue).content))
dialogue.append(additional_question_message)
dialogue.append(AIMessage(content=deepseek_llm.invoke(dialogue).content))
dialogue.append(additional_question_message_2)
dialogue.append(AIMessage(content=deepseek_llm.invoke(dialogue).content))

for phrase in dialogue:
    phrase.pretty_print()