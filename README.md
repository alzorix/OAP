#  OAP (Telegram Sentiment Analyzer)

Инструмент для сбора сообщений из Telegram-каналов, классификации их эмоциональной окраски с помощью нейросети и визуализации результатов.

##  Структура проекта

```
project/
├── main.py                        # Точка входа — меню выбора модуля
├── data/
│   ├── telegram_messages.csv           # Сырые данные после парсинга 
│   └── telegram_messages_preprocessed.csv  # Данные с emotion_score после AI-обработки
│
├── NewsFetcher/
│   ├── main.py                    # Парсер Telegram-каналов (Telethon)
│   └── config.py                  # api_id, api_hash и прочие настройки Telegram
│
├── DataPreprocessing/
│   ├── main.py                    # Запуск AI-классификации эмоций
│   ├── AI_tools.py                # Модель, батчевая классификация, вспомогательные утилиты
│   └── config.py                  # Путь к CSV с сырыми данными
│
└── DataAnalyze/
    ├── main.py                    # Построение графиков
    └── config.py                  # Путь к CSV с предобработанными данными
```


## Пайплайн обработки данных

```
 [Telegram API]
      │
      ▼
 NewsFetcher         ← Собирает сообщения из всех каналов аккаунта за заданный период
      │
      ▼
telegram_messages.csv
      │
      ▼
DataPreprocessing    ← Классифицирует эмоции, добавляет колонку emotion_score
      │
      ▼
telegram_messages_preprocessed.csv
      │
      ▼
DataAnalyze          ← Строит 8 графиков: тональность по времени, активность, тренды
```


## Модули

### 1\. `NewsFetcher` — Парсер Telegram

Асинхронный парсер на базе [Telethon](https://docs.telethon.dev/). Собирает сообщения из всех каналов (не групп), на которые подписан аккаунт, за указанный период и сохраняет их в CSV.

**Ключевые параметры** (`NewsFetcher/main.py`):

|Параметр|По умолчанию|Описание|
|-|-|-|
|`START_DATE`|`2021-01-01`|Начало периода сбора|
|`END_DATE`|`2026-03-31`|Конец периода сбора|
|`CSV_FILE`|`telegram_messages.csv`|Имя выходного файла|
|`DELAY_BETWEEN_CHANNELS`|`2.0 с`|Пауза между каналами (защита от FloodWait)|
|`MESSAGES_PER_BATCH`|`50`|Размер пачки перед паузой|
|`QUEUE_MAX_SIZE`|`5000`|Максимальный размер очереди в памяти|

**Архитектура**: producer/consumer через `asyncio.Queue`.

* `message_generator()` — асинхронный генератор сообщений одного канала
* `parser()` — обходит все каналы, кладёт данные в очередь
* `consumer()` — читает из очереди и пишет в CSV

**Поддержка прокси**: SOCKS5 (`127.0.0.1:2080`) — настраивается в `main.py`.

**Выходной CSV**:

```
date,text
2024-01-15 10:32:00,"Текст сообщения..."
```

### 2\. `DataPreprocessing` — AI-классификация эмоций

Добавляет к каждому сообщению числовую оценку тональности (`emotion_score`) на основе многоязычной модели классификации эмоций.

**Модель**: [`tabularisai/multilingual-emotion-classification`](https://huggingface.co/tabularisai/multilingual-emotion-classification) (HuggingFace Transformers)

**Поддерживаемые эмоции и их вес**:

|Эмоция|Вес|
|-|-|
|`love`|+1.0|
|`joy`|+0.9|
|`gratitude`|+0.6|
|`neutral`|0.0|
|`surprise`|0.0|
|`frustration`|−0.5|
|`contempt`|−0.6|
|`fear`|−0.7|
|`sadness`|−0.7|
|`disgust`|−0.8|
|`anger`|−1.0|

**Обработка длинных текстов**: тексты длиннее 512 символов разбиваются на фрагменты, валентности фрагментов суммируются.

**Декораторы**:

* `@timer` — логирует время выполнения функции
* `@log_call` — логирует количество входных элементов

**Добавляемые колонки**:

```
hour, day, month, weekday, msg\_len, emotion_score
```

### 3\. `DataAnalyze` — Визуализация

Строит 8 графиков на основе предобработанных данных.

|#|График|Описание|
|-|-|-|
|1|Доля эмоций|Столбчатая диаграмма: позитив / нейтрально / негатив|
|2|Тональность по месяцам|Линейный тренд среднего `emotion_score`|
|3|Тональность по часам|В какое время суток пишут позитивнее|
|4|Тональность по дням недели|Средний `emotion_score` по дням Пн–Вс|
|5|Активность по месяцам|Количество постов во времени|
|6|Активность по часам|Когда публикуется больше всего постов|
|7|Активность по дням недели|Распределение постов по дням|
|8|Тренд (скользящее среднее)|По месячной тональности|

Категоризация тональности:

```
emotion_score < −0.33  →  Негатив
−0.33 ≤ score ≤ 0.33   →  Нейтрально
emotion_score > 0.33   →  Позитив
```

## Установка

### 1\. Клонирование и создание окружения

```bash
git clone https://github.com/alzorix/OAP
cd project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
```

### 2\. Установка зависимостей

```bash
pip install telethon socks transformers torch datasets tqdm pandas numpy matplotlib
```

> Для GPU (CUDA): установите PyTorch с поддержкой CUDA согласно \[официальной инструкции](https://pytorch.org/get-started/locally/).

### 3\. Настройка Telegram API

Получите `api_id` и `api_hash` на [my.telegram.org](https://my.telegram.org):

```python
# NewsFetcher/config.py
api_id   = 123456         # ваш api_id
api_hash = "your_hash"    # ваш api_hash
```

##  Запуск

### Интерактивное меню

```bash
python main.py
```

```
=== Выбор модуля ===
1. DataAnalyze (анализ обработанных данных)
2. DataPreprocessing (AI обработка)
3. NewsFetcher (Парсинг новостей)
0. Выход
```

### Запуск конкретного модуля напрямую

```bash
python main.py fetcher      # Запустить парсер
python main.py preprocess   # Запустить AI-классификацию
python main.py analyze      # Запустить визуализацию
```

### Рекомендуемый порядок запуска

```bash
python main.py fetcher      # 1. Собрать данные
python main.py preprocess   # 2. Классифицировать эмоции
python main.py analyze      # 3. Визуализировать результаты
```

## Требования

|Библиотека|Назначение|
|-|-|
|`telethon`|Клиент Telegram MTProto API|
|`PySocks`|SOCKS5 прокси для Telethon|
|`transformers`|HuggingFace pipeline для классификации|
|`torch`|PyTorch — бэкенд для модели|
|`datasets`|Эффективная работа с данными в HuggingFace|
|`tqdm`|Прогресс-бар батчевой обработки|
|`pandas`|Работа с CSV и временными рядами|
|`numpy`|Скользящее среднее|
|`matplotlib`|Построение графиков|

## Замечания

* **Первый запуск `NewsFetcher`** запросит номер телефона и код подтверждения Telegram — сессия сохраняется в файл `parser\parser_session.session` и повторная авторизация не требуется.
* **FloodWait**: при превышении лимитов Telegram парсер автоматически делает паузу на указанное сервером время + 5 секунд.
* **Прокси**: если прокси не нужен, удалите параметр `proxy=...` из вызова `TelegramClient` в `NewsFetcher/main.py`.
* **Память**: батчевая классификация чанкует данные, поэтому работает даже на датасетах в несколько сотен тысяч сообщений.
* Модель загружается с HuggingFace при первом запуске и кешируется локально.
*  *Не забудьте переместить данные из рабочих папок,в папку data*

