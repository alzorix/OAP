from datetime import time
from os import truncate

import torch
from transformers import pipeline
import time
import logging
logger = logging.getLogger(__name__)
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
# Константы
EMOTION_VALENCE = {
    "anger": -1,
    "contempt": -0.6,
    "disgust": -0.8,
    "fear": -0.7,
    "frustration": -0.5,
    "sadness": -0.7,
    "neutral": 0.0,
    "gratitude": 0.6,
    "joy": 0.9,
    "love": 1.0,
    "surprise": 0.0
}

CHUNK_SIZE = 25_000
BATCH_SIZE = None

#Декараторы!!!!
def timer(func):
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[timer] {func.__name__} выполнилась за {elapsed:.3f} с")
        return result
    return wrapper


def log_call(func):
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        first_arg = args[0] if args else None
        count = len(first_arg) if isinstance(first_arg, (list, str)) else "?"
        logger.info(f"[log_call] {func.__name__} вызвана | элементов: {count}")
        return func(*args, **kwargs)
    return wrapper


def setup_device():
    global BATCH_SIZE
    """Настройка устройства (CUDA или CPU)"""
    if torch.cuda.is_available():
        print(f"CUDA доступна. Используется видеокарта: {torch.cuda.get_device_name(0)}")
        print(f"Объём памяти: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} ГБ")
        BATCH_SIZE = 256
        return 0
    else:
        print("CUDA не доступна. Будет использован CPU.")
        BATCH_SIZE= 64
        return -1


def create_classifier(device):
    """Создание классификатора эмоций"""
    return pipeline(
        "text-classification",
        model="tabularisai/multilingual-emotion-classification",
        device=device,
        truncation= False,#Автообрезка текста - отключена в рамках дороботки от 06.05
        max_length=512,
        dtype=torch.float16
    )

# Инициализация
device = setup_device()
classifier = create_classifier(device)

@timer
def classify_emotion(text: str, max_length=512) -> float:
    """
    Классификация эмоциональной окраски текста.

    Args:
        text: Входной текст для анализа
        max_length: Ограничение модели по длине

    Returns:
        float: Совокупная тональность сообщения
    """

    if not text or not isinstance(text, str):
        logger.warning("Exist bad text")
        return 0

    texts = list()
    result = 0.0
    if len(text) > max_length:
        text_gen = (text[i:i + max_length] for i in range(0, len(text), max_length)) # <--- генератор

        texts = list(text_gen)
    else:
        texts.append(text)

    for text in texts:
        count = classifier(text)[0]['label']
        result += EMOTION_VALENCE[count]

    return result


@log_call
def classify_emotion_batch_edition(texts: list[str], batch_size: int = BATCH_SIZE, chunk_size: int = CHUNK_SIZE, max_length=512) -> list[int]:
    """
    Классификация эмоциональной окраски текста группой.

    Длинные тексты разбиваются на части по max_length символов,
    валентности частей суммируются (как в classify_emotion).
    Обработка идёт чанками — Dataset не нужен, MemoryError исключён.

    Args:
        texts: Входные тексты для анализа
        batch_size: Кол-во текстов, обрабатываемых GPU за один проход
        chunk_size: Кол-во sub-текстов за один вызов pipeline
        max_length: Макс. длина одного фрагмента в символах

    Returns:
        Список float: суммарная тональность каждого исходного текста
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    # Разбиваем длинные тексты, запоминаем исходный индекс
    sub_texts: list[str] = []
    sub_to_orig: list[int] = []

    for i, text in enumerate(texts):
        text = text if isinstance(text, str) and text.strip() else " "
        parts = [text[j:j + max_length] for j in range(0, len(text), max_length)]
        sub_texts.extend(parts)
        sub_to_orig.extend([i] * len(parts))

    results = [0.0] * len(texts)

    with tqdm(total=len(sub_texts), desc="Анализ эмоциональной окраски текста") as progress_bar:
        for start in range(0, len(sub_texts), chunk_size):
            chunk = sub_texts[start: start + chunk_size]
            chunk_origs = sub_to_orig[start: start + chunk_size]

            for r, orig_idx in zip(classifier(chunk, batch_size=batch_size), chunk_origs):
                results[orig_idx] += EMOTION_VALENCE.get(r['label'], 0.0)
                progress_bar.update(1)

    return results


# Сравнение скорости
if __name__ == "__main__":
    test_texts = [
        "Я очень счастлив сегодня!",
        "Это ужасно и печально.",
        "Сегодня обычный день.",
        "Обожаю этот проект, всё отлично работает!",
        "Ненавижу баги в коде.",
    ]

    print("Одиночная классификация")
    start = time.time()
    for text in test_texts:
        print(f"  {classify_emotion(text)}  '{text}'")
    single_time = time.time() - start
    print(f"Время: {single_time} с  ({single_time / len(test_texts)} с/текст)\n")

    print("Батчевая классификация")
    start = time.time()
    scores = classify_emotion_batch_edition(test_texts)
    batch_time = time.time() - start
    for text, score in zip(test_texts, scores):
        print(f"  {score}  '{text}'")
    print(f"Время: {batch_time:} с  ({batch_time / len(test_texts)} с/текст)")
    print(f"\nУскорение: x{single_time / batch_time}")