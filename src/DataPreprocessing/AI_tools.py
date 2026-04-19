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
    "contempt": -1,
    "disgust": -1,
    "fear": -1,
    "frustration": -1,
    "sadness": -1,
    "neutral": 0,
    "gratitude": 1,
    "joy": 1,
    "love": 1,
    "surprise": 1
}

BATCH_SIZE = None
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
        truncation= True,#Автообрезка текста
        max_length=512,
        dtype=torch.float16
    )

# Инициализация
device = setup_device()
classifier = create_classifier(device)

def classify_emotion(text: str, max_length=512) -> int:
    """
    Классификация эмоциональной окраски текста.

    Args:
        text: Входной текст для анализа
        max_length: Ограничение модели по длине

    Returns:
        int: -1 (негативный), 0 (нейтральный), 1 (позитивный)
    """
    if not text or not isinstance(text, str):
        logger.warning("Exist bad text")
        return 0

    if len(text) > max_length:
        text = text[:max_length]

    result = classifier(text)[0]['label']
    return EMOTION_VALENCE.get(result, 0)  # По умолчанию возвращаем 0


def classify_emotion_batch_edition(texts: list[str],batch_size: int = BATCH_SIZE) -> list[int]:
    """
        Классификация эмоциональной окраски текста группой

    Args:
        texts: Входные текста для анализа
        batch_size: Количество обрабатываемого текста за один подход

    Returns:
        Список из int: -1 (негативный), 0 (нейтральный), 1 (позитивный)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    clean_texts = [t if isinstance(t, str) and t else " " for t in texts]
    dataset = Dataset.from_dict({"text": clean_texts}) #

    results = classifier(KeyDataset(dataset, "text"), batch_size=batch_size)
    return [EMOTION_VALENCE.get(r['label'], 0) for r in tqdm(results, total=len(clean_texts), desc="Анализ эмоциональной окраски текста")]


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