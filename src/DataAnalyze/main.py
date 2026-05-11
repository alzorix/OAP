import matplotlib.pyplot as plt
import pandas as pd
from config import CSV_FILE
import numpy as np

df = pd.read_csv(CSV_FILE)
df["date"] = pd.to_datetime(df["date"])
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

fig, axes = plt.subplots(4, 2, figsize=(18, 20))
fig.suptitle("Анализ тональности постов", fontsize=16, fontweight="bold")

#1. Доля эмоций
bins = [-float('inf'), -0.33, 0.33, float('inf')]
labels = ["Позитив", "Нейтрально", "Негатив"]

df["emotion_cat"] = pd.cut(df["emotion_score"], bins=bins, labels=labels)

counts = df["emotion_cat"].value_counts(normalize=True).reindex(labels, fill_value=0)

ax = axes[0, 0]
counts.plot(kind="bar", color=["green", "gray", "red"], ax=ax)
ax.set_title("Доля эмоций по всем постам")
ax.set_xlabel("Тональность")
ax.set_ylabel("Доля постов")
ax.tick_params(axis="x", rotation=0)

#2. Средняя тональность по месяцам
ax = axes[0, 1]
df.groupby([df["date"].dt.year, df["date"].dt.month])["emotion_score"].mean().plot(ax=ax)
ax.set_title("Тональность по месяцам")
ax.set_xlabel("Год / Месяц")
ax.set_ylabel("Средний emotion_score")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

#3. Средняя тональность по часам
ax = axes[1, 0]
df.groupby("hour")["emotion_score"].mean().plot(kind="bar", ax=ax)
ax.set_title("В какое время суток пишут позитивнее?")
ax.set_xlabel("Час суток")
ax.set_ylabel("Средний emotion_score")
ax.tick_params(axis="x", rotation=0)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

#4. Средняя тональность по дням недели
ax = axes[1, 1]
df.groupby("weekday")["emotion_score"].mean().plot(kind="bar", ax=ax)
ax.set_xticklabels(WEEKDAYS, rotation=0)
ax.set_title("Тональность по дням недели")
ax.set_xlabel("День недели")
ax.set_ylabel("Средний emotion_score")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

#5. Количество постов по месяцам
ax = axes[2, 0]
df.groupby([df["date"].dt.year, df["date"].dt.month]).size().plot(ax=ax)
ax.set_title("Активность: количество постов по месяцам")
ax.set_xlabel("Год / Месяц")
ax.set_ylabel("Количество постов")

#6. Активность по часам суток
ax = axes[2, 1]
df.groupby("hour").size().plot(kind="bar", ax=ax)
ax.set_title("Активность по часам суток")
ax.set_xlabel("Час суток")
ax.set_ylabel("Количество постов")
ax.tick_params(axis="x", rotation=0)

#7. Активность по дням недели
ax = axes[3, 0]
df.groupby("weekday").size().plot(kind="bar", ax=ax)
ax.set_xticklabels(WEEKDAYS, rotation=0)
ax.set_title("Активность по дням недели")
ax.set_xlabel("День недели")
ax.set_ylabel("Количество постов")

# 8. Тренд тональности (скользящее среднее)
ax = axes[3, 1]
monthly_sentiment = (
    df.groupby([df["date"].dt.year, df["date"].dt.month])["emotion_score"]
    .mean().values
)
window = 3
smoothed = np.convolve(monthly_sentiment, np.ones(window) / window, mode="valid") # <--- numpy
ax.plot(monthly_sentiment, alpha=0.4, label="Оригинал")
ax.plot(range(window - 1, len(monthly_sentiment)), smoothed, label=f"MA-{window}", linewidth=2)
ax.axhline(0, color="gray", linestyle="--")
ax.set_title("Тренд тональности (скользящее среднее)")
ax.legend()

plt.tight_layout()
plt.show()