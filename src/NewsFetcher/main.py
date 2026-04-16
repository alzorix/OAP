import asyncio
import csv
import logging
from datetime import datetime, timezone, timedelta
from telethon import TelegramClient
from telethon.tl.types import Channel
from telethon.errors import FloodWaitError
import config
import socks
SESSION_FILE = "parser_session"

# Период парсинга
START_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc)

# Имя выходного файла
CSV_FILE = "telegram_messages.csv"

# ЗАДЕРЖКИ
DELAY_BETWEEN_CHANNELS = 2.0  # Пауза между каналами (сек)
DELAY_PER_MESSAGE_BATCH = 1.0  # Пауза каждые N сообщений
MESSAGES_PER_BATCH = 50  # Размер пачки перед паузой
QUEUE_MAX_SIZE = 5000  # Лимит очереди


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


message_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)


async def parser(client, channels, start_date, end_date):
    """Собирает сообщения и кладёт в очередь"""
    for idx, channel in enumerate(channels, 1):
        try:
            logger.info(f" [{idx}/{len(channels)}] Канал: {channel.title}")
            msg_counter = 0

            async for msg in client.iter_messages(channel, offset_date=end_date + timedelta(seconds=1)):
                if msg.date < start_date:
                    break

                if start_date <= msg.date <= end_date:
                    data = {
                        "date": msg.date.strftime("%Y-%m-%d %H:%M:%S"),
                        "text": msg.text or ""
                    }
                    await message_queue.put(data)
                    msg_counter += 1

                    # Периодическая пауза
                    if msg_counter % MESSAGES_PER_BATCH == 0:
                        logger.debug(f" Пачка из {MESSAGES_PER_BATCH} msg. Пауза {DELAY_PER_MESSAGE_BATCH}с...")
                        await asyncio.sleep(DELAY_PER_MESSAGE_BATCH)

            # Пауза между каналами
            await asyncio.sleep(DELAY_BETWEEN_CHANNELS)

        except FloodWaitError as e:
            wait_time = e.seconds + 5
            logger.warning(f" FloodWait на {channel.title}. Ждём {wait_time} сек...")
            await asyncio.sleep(wait_time)
            continue
        except Exception as e:
            logger.error(f" Ошибка при обработке {channel.title}: {e}")
            continue


async def consumer(csv_writer):
    """Забирает данные из очереди и пишет в CSV"""
    processed = 0
    while True:
        data = await message_queue.get()
        try:
            csv_writer.writerow([
                data["date"],
                data["text"]
            ])
            processed += 1
            if processed % 500 == 0:
                logger.info(f" Записано в CSV: {processed} сообщений...")
        except Exception as e:
            logger.error(f" Ошибка записи в CSV: {e}")
        finally:
            message_queue.task_done()


async def main():
    csv_file = open(CSV_FILE, mode="w", encoding="utf-8-sig", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["date", "text"])

    try:
        async with TelegramClient(SESSION_FILE,api_id=config.api_id,api_hash=config.api_hash,proxy=(socks.SOCKS5, '127.0.0.1', 2080)) as client:
            me = await client.get_me()
            logger.info(f"  Авторизован как: {me.first_name}")

            channels = [
                dialog.entity for dialog in await client.get_dialogs()
                if isinstance(dialog.entity, Channel) and not dialog.entity.megagroup
            ]
            logger.info(f"  Найдено каналов: {len(channels)}")

            consumer_task = asyncio.create_task(consumer(writer))

            await parser(client, channels, START_DATE, END_DATE)

            await message_queue.join()
            consumer_task.cancel()
            logger.info("   Парсинг завершён. Очередь пуста.")

    finally:
        csv_file.close()
        logger.info(f"  Данные сохранены в: {CSV_FILE}")


if __name__ == "__main__":
    asyncio.run(main())