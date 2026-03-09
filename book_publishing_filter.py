# -*- coding: utf-8 -*-

from datetime import datetime, timedelta, timezone
from telethon import TelegramClient, events
from telethon.errors import PersistentTimestampOutdatedError
from telethon.tl.types import UpdateShort
import asyncio, time, re
import config

from services.filters import quick_keyword_pass, fuzzy_keyword
from services.llm import llm_is_relevant
from services.ocr import ocr_photo_to_text, ocr_from_image_url
from messengers.telegram import TelegramMessenger, rate_sleep
from messengers.mastodon import MastodonMessenger
from publisher import Publisher


client = TelegramClient(config.SESSION, config.API_ID, config.API_HASH)
client.add_event_handler(lambda e: None, events.Raw(types=(UpdateShort,)))
SEEN_ALBUMS: set[int] = set()
IN_PROGRESS: set[int] = set()
SEEN_MSG_IDS: set[int] = set()

config.TEMP_DIR.mkdir(exist_ok=True)

tg_messenger = TelegramMessenger(client)
mastodon_messenger = MastodonMessenger()
publisher = Publisher(tg_messenger, mastodon_messenger)


async def get_album(msg):
    if not msg.grouped_id:
        return [msg]

    siblings = []
    async for sib in client.iter_messages(
        msg.chat_id,
        min_id=msg.id - 20,
        max_id=msg.id + 20,
    ):
        if sib.grouped_id == msg.grouped_id:
            siblings.append(sib)

    siblings.append(msg)
    siblings = list({m.id: m for m in siblings}.values())
    siblings.sort(key=lambda m: m.id)
    return siblings


def add_to_seen_albums(grouped_ids):
    if grouped_ids:
        IN_PROGRESS.discard(grouped_ids)
        SEEN_ALBUMS.add(grouped_ids)


@client.on(events.NewMessage(chats=config.CHANNELS))
async def new_msg_handler(event):
    chanal = event.chat.username or event.chat.title or "unknown"
    prefix = f"[@{chanal}] " if chanal else ""
    msg = event.message
    chat_name = event.chat.title
    grouped_ids = msg.grouped_id
    url = re.search(r"https://telegra\.ph/file/\S+\.(jpg|jpeg|png)", msg.text or "")
    text_from_img = ""

    if grouped_ids:
        if grouped_ids in SEEN_ALBUMS or grouped_ids in IN_PROGRESS:
            return
        IN_PROGRESS.add(grouped_ids)
    else:
        if msg.id in SEEN_MSG_IDS:
            return
        SEEN_MSG_IDS.add(msg.id)

    album = await get_album(msg)
    caption_text = next((m.raw_text for m in album if m.raw_text), "") or ""
    try:
        if caption_text and quick_keyword_pass(caption_text):
            print(f"✅ keyword‑pass {prefix}")
            await publisher.publish(msg, album, chat_name)
            add_to_seen_albums(grouped_ids)
            return

        if caption_text:
            print(f"LLM‑caption → {prefix}{caption_text[:80]}…")
            if await llm_is_relevant(caption_text):
                print(f"✅ GPT 'yes' for caption {prefix}")
                await publisher.publish(msg, album, chat_name)
                add_to_seen_albums(grouped_ids)
                return
            print(f"❌ GPT 'no' for caption; go to OCR… {prefix}")

        if url:
            text_from_img = await ocr_from_image_url(url.group(0))
        ocr_targets = [
            m for m in album
            if m.photo or (m.document and m.document.mime_type.startswith("image/"))
        ][:2]

        ocr_texts = []
        for idx, m in enumerate(ocr_targets, 1):
            txt = await ocr_photo_to_text(m, client)
            if txt:
                ocr_texts.append(txt)
            print(f"OCR part {idx}: {txt[:60]}")

        ocr_text = "\n".join(ocr_texts)
        full_ocr_text = ocr_text or text_from_img

        if full_ocr_text:
            if fuzzy_keyword(full_ocr_text):
                print(f"✅ fuzzy‑keyword (новинки) pass {prefix}")
                await publisher.publish(msg, album, chat_name)
                add_to_seen_albums(grouped_ids)
                return

            if await llm_is_relevant(full_ocr_text):
                print(f"✅ GPT 'yes' from OCR or link {prefix}")
                await publisher.publish(msg, album, chat_name)
                add_to_seen_albums(grouped_ids)
                return

        if grouped_ids:
            SEEN_ALBUMS.add(grouped_ids)

        print(f"🚫 filtered {prefix}(caption+OCR = no)")

    finally:
        if grouped_ids:
            IN_PROGRESS.discard(grouped_ids)
            if len(SEEN_ALBUMS) > 200: SEEN_ALBUMS.pop()
        else:
            if len(SEEN_MSG_IDS) > 200: SEEN_MSG_IDS.pop()


async def backfill(hours: int = config.BACKFILL_HOURS):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    print(f"🔍 backfill {hours}h ( {since.isoformat(timespec='seconds')})")

    total = 0
    for chan in config.CHANNELS:
        async for msg in client.iter_messages(chan, offset_date=since, reverse=True):
            txt = msg.raw_text or ""
            if txt and await llm_is_relevant(txt):
                album = await get_album(msg)
                await publisher.publish(msg, album)
                total += 1
                await rate_sleep()

    print(f"Backfill end, sended {total} messages")


async def init_channels():
    for ch in config.CHANNELS:
        try:
            await client.get_input_entity(ch)
        except Exception as e:
            print(f"⚠️  Cannot resolve {ch}: {e}")


def main():
    async def runner():
        await client.connect()
        print(f"🔌 Connected at {datetime.now().isoformat(timespec='seconds')}")

        if not config.RUN_BACKFILL:
            await init_channels()
            for ch in config.CHANNELS:
                try:
                    await client.get_input_entity(ch)
                except Exception as e:
                    print(f"⚠️ can't access to {ch}: {e}")

        if config.RUN_BACKFILL:
            await backfill(hours=config.BACKFILL_HOURS)

        print("▶️ Listening…")
        await client.run_until_disconnected()

    while True:
        try:
            client.loop.run_until_complete(runner())
        except PersistentTimestampOutdatedError as e:
            print(f"⚠️ {e}; reconnecting in 5 s…")
        except Exception as e:
            print(f"‼️ loop crashed: {e}")
            import traceback
            traceback.print_exc()
        try:
            client.disconnect()
        except:
            pass
        time.sleep(5)


if __name__ == "__main__":
    main()
