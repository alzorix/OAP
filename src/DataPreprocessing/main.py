import pandas as pd
import numpy as np
from config import CSV_FILE
from tqdm import tqdm
import AI_tools



df = pd.read_csv(CSV_FILE)
df['date'] = pd.to_datetime(df['date']) #Переводим дату str в дату pandas
df = df.dropna(subset=['text']).reset_index(drop=True)  #Удаляем пустые сообщения т.е. репосты из тг каналов + переделываем индексы от греха подальше
df.drop_duplicates(subset=['text']).reset_index(drop=True)
df['hour']    = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['msg_len'] = df['text'].str.len()


print(df.info())

BATCH_SIZE = AI_tools.BATCH_SIZE # Размер BATCH определяет модуль AI_tools
texts = df['text'].tolist()
print(f"Всего сообщений: {len(texts)}")
all_scores = AI_tools.classify_emotion_batch_edition(texts)
df['emotion_score'] = all_scores

out_path = CSV_FILE.parent / "telegram_messages_preprocessed.csv"
df.to_csv(out_path, index=False)
print(f"\nРезультат сохранён: {out_path}")




