# -*- coding: utf-8 -*-

from datetime import datetime, timedelta, timezone
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from telethon.errors import PersistentTimestampOutdatedError
from telethon.tl.types import UpdateShort
import requests, asyncio, time, tempfile, pathlib
from functools import lru_cache
import asyncio, os
import openai
import easyocr, cv2, numpy as np
import traceback
import re
import json

with open("keys.json", "r", encoding="utf-8") as f:
    keys = json.load(f)
reader = easyocr.Reader(['uk', 'en'], gpu=False)
API_ID   = keys["api_id"]
API_HASH = keys["api_hash"]
SESSION  = "books_monitor"

CHANNELS = [
    "@vivat_publishing",
    "@ksd_bookclub",
    "@vydavnytstvo",
    "@nebobooks",
    "@molfarpublish",
    "@rm_publisher",
    "@librarius_books",
    "@artbooks_ua",
    "@starlev",
    "@ababahalamaha",
    "@vydavnytstvotak",
    "@nashaidea",
    "@fabulabook",
    "@testing_channel_book",
    "@publishingvarvar",
    "@knygolove",
    "@bohdan_books",
    "@KM_Books",
    "@malopus",
    "@laboratoryua",
    "@zhupanskogo",
    "@Book_Chef",
]

TARGET_CHAT = "@new_book_filter"               
BOT_TOKEN   = keys["bot_token"]
BOT_API     = f"https://api.telegram.org/bot{BOT_TOKEN}"
OPENAI_API_KEY = keys["openai_api_key"]

BACKFILL_HOURS = 1
RUN_BACKFILL = True 
FORWARD_MODE = True 
KEYWORDS = ("передзамов", "у друці")
SEEN_ALBUMS: set[int] = set()
IN_PROGRESS: set[int] = set()

client = TelegramClient(SESSION, API_ID, API_HASH)
client.add_event_handler(lambda e: None, events.Raw(types=(UpdateShort,)))

SYSTEM_PROMPT = (
    # --- Role ---
    "You are a binary classifier that must answer ONLY \"yes\" or \"no\".\n\n"

    # --- Task ---
    "Decide whether the Telegram post below FACTUALLY announces a specific new book "
    "or books in any of the following ways:\n"
    " • preorder (передзамовлення) has just started or is still open;\n"
    " • last day / final hours of an ongoing preorder;\n"
    " • a new book (paper, e‑book, or audiobook) has just gone on sale;\n"
    " • new books (paper, e‑book, or audiobook) will be sale in this mounth;\n"
    " • a concrete forthcoming title is officially announced (анонс) AND preorder / sale info is given.\n\n"

    # --- (YES) ---
    "Treat posts as *YES* if they contain phrases like:\n"
    "  - “Передзамовлення відкрите”, “Старт передзамовлення”, “Уже у передпродажі”;\n"
    "  - “Передзамовлення триває до кінця дня”, “Останній день передзамовлення”;\n"
    "  - “Новинка вже у продажу / у е‑форматі / в аудіо”; “Вийшла з друку”, “Книга вийшла”, “Новинки <назва місяця>“;\n"
    "  - “Анонс книги <Назва>: уже можна купити / доступна до замовлення”.\n"
    "  - If OCR contains the Ukrainian word “новинки” (new releases)\n\n"

    # --- (NO) ---
    "Treat posts as *NO* if they are only:\n"
    "  - future schedules, calendars, vague plans (e.g. “Плани передзамовлень на осінь”);\n"
    "  - events, presentations, conferences, discounts, giveaways, memes, reviews, quotes;\n"
    "  - general marketing without stating that preorder or sale is already open.\n\n"

    # --- Details ---
    "The post text may include OCR output extracted from images — read it as well.\n"
    "If the post is ambiguous or you're uncertain, answer “no”.\n\n"

    # --- Answer form ---
    "Answer strictly “yes” or “no”. No explanations."
)

OPENAI_MODEL = "gpt-4o-mini"

# ----- LLM func

_llm_cache: dict[str, bool] = {}

async def llm_is_relevant(text: str) -> bool:
    if text in _llm_cache:          # ← return correct answer
        return _llm_cache[text]

    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)   # key from env
    try:
        resp = await client.chat.completions.create(
            model = OPENAI_MODEL,
            temperature = 0,
            max_tokens = 1,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f'Post:\n"""\n{text[:4000]}\n"""'}
            ],
        )
        decision = resp.choices[0].message.content.strip().lower().startswith("y")
        _llm_cache[text] = decision        
        
        if len(_llm_cache) > 2048:
            _llm_cache.pop(next(iter(_llm_cache)))
        return decision

    except Exception as e:
        print("‼️ LLM error:", e)
        return False

async def is_relevant(text: str) -> bool:
    return await llm_is_relevant(text)

# ----- Start Telethon and bot part

async def rate_sleep(sec: float = 2.0):
    await asyncio.sleep(sec) # pouse between messages in chat

def bot_api(method: str, **params):
    return requests.post(f"{BOT_API}/{method}", **params, timeout=60)

def fuzzy_keyword(text: str) -> bool:
    low = text.lower()
    patterns = [
        r"новинк[аи]",            
        r"новинк",                
    ]
    return any(re.search(p, low) for p in patterns)

def quick_keyword_pass(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in KEYWORDS)

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

# ----- Start OCR part 

async def ocr_photo_to_text(message) -> str:
    raw = await client.download_media(message, bytes)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    lines = reader.readtext(img, detail=0, paragraph=False)
    return "\n".join(lines).strip()

async def ocr_first_two(msg) -> str:
    images = []
    if msg.photo or (msg.document and msg.document.mime_type.startswith("image/")):
        images.append(msg)
    if msg.grouped_id:
        async for sib in client.iter_messages(msg.chat_id, min_id=msg.id-15, max_id=msg.id+15):
            if sib.grouped_id == msg.grouped_id and sib.id != msg.id:
                if sib.photo or (sib.document and sib.document.mime_type.startswith("image/")):
                    images.append(sib)
            if len(images) >= 2:
                break
    texts = [await ocr_photo_to_text(im) for im in images[:2]]
    return "\n".join(t for t in texts if t)

# ----- End OCR part

async def collect_text(msg) -> str:
    parts = [msg.raw_text or ""]
    if msg.photo or (msg.document and msg.document.mime_type.startswith("image/")):
        parts.append(await ocr_photo_to_text(msg))
    if msg.grouped_id:
        async for sibling in client.get_messages(msg.chat_id,
                                                 ids=[m.id for m in
                                                      await client.get_messages(msg.chat_id, limit=10,
                                                           offset_id=msg.id,
                                                           min_id=msg.id-15)]):
            if sibling.grouped_id == msg.grouped_id and sibling.id != msg.id:
                if sibling.photo or (sibling.document and sibling.document.mime_type.startswith("image/")):
                    parts.append(await ocr_photo_to_text(sibling))
    return "\n".join([p for p in parts if p])

def send_text_via_bot(text: str, link: str):
    payload = dict(chat_id=TARGET_CHAT,
                   parse_mode="HTML",
                   disable_web_page_preview=False,
                   text=f"{text}\n\n<a href=\"{link}\">Джерело</a>")
    attempt = 0
    while attempt < 5:
        r = bot_api("sendMessage", json=payload)
        if r.ok:
            return
        if r.status_code == 429:
            retry = r.json().get("parameters", {}).get("retry_after", 10)
            print(f"429, waiting {retry}s…")
            time.sleep(retry + 1)
            attempt += 1
        else:
            print("‼️ BOT ERROR:", r.text)
            return

def send_photo_via_bot(photo_path: pathlib.Path, caption: str = ""):
    with photo_path.open("rb") as f:
        files = {"photo": f}
        data  = dict(chat_id=TARGET_CHAT, caption=caption, parse_mode="HTML")
        attempt = 0
        while attempt < 5:
            r = bot_api("sendPhoto", data=data, files=files)
            if r.ok:
                return
            if r.status_code == 429:
                retry = r.json().get("parameters", {}).get("retry_after", 10)
                print(f"429(photo), waiting {retry}s…")
                time.sleep(retry + 1)
                attempt += 1
            else:
                print("‼️ BOT PHOTO ERROR:", r.text)
                return

def shorten(text: str, limit: int = 1000) -> str:
    return text if len(text) <= limit else text[:limit] + "…"

async def forward_or_send(msg):
    album_msgs = await get_album(msg)

    if FORWARD_MODE:
        try:
            await client.forward_messages(TARGET_CHAT, album_msgs, msg.chat_id)
            await rate_sleep()
            return
        except FloodWaitError as e:
            print(f"⏳ FloodWait {e.seconds}s, sleep...")
            await asyncio.sleep(e.seconds + 1)
            await client.forward_messages(TARGET_CHAT, album_msgs, msg.chat_id)
            await rate_sleep()
            return

    link = f"https://t.me/{msg.chat.username}/{msg.id}"
    text_cap = shorten(msg.raw_text or "") + f"\n\n<a href=\"{link}\">Джерело</a>"
    first = True
    for m in album_msgs:
        if m.photo:
            with tempfile.TemporaryDirectory() as td:
                p = pathlib.Path(td) / "img.jpg"
                await m.download_media(file=p)
                caption = text_cap if first else ""
                send_photo_via_bot(p, caption=caption)
                first = False
                await rate_sleep()

    if first:
        send_text_via_bot(shorten(msg.raw_text or ""), link)
        await rate_sleep()

async def init_channels():
    await client.connect()                 
    for ch in CHANNELS:
        try:
            await client.get_input_entity(ch)
        except Exception as e:
            print(f"⚠️  Cannot resolve {ch}: {e}")
    await client.disconnect()              

def add_to_seen_albums(grouped_ids):
    if grouped_ids:
        IN_PROGRESS.discard(grouped_ids)
        SEEN_ALBUMS.add(grouped_ids)

@client.on(events.NewMessage(chats=CHANNELS))
async def new_msg_handler(event):
    chanal = event.chat.username or event.chat.title or "unknown"
    prefix = f"[@{chanal}] " if chanal else ""
    msg = event.message
    grouped_ids = msg.grouped_id
    
    if grouped_ids:
        if grouped_ids in SEEN_ALBUMS or grouped_ids in IN_PROGRESS:
            return
        IN_PROGRESS.add(grouped_ids)
    
    album = await get_album(msg)  
    caption_text = next((m.raw_text for m in album if m.raw_text), "") or ""

    try:
        if caption_text and quick_keyword_pass(caption_text):
            print(f"✅ keyword‑pass {prefix}")
            await forward_or_send(msg)
            add_to_seen_albums(grouped_ids)
            return

        if caption_text:
            print(f"LLM‑caption → {prefix}{caption_text[:80]}…")
            if await is_relevant(caption_text):
                print(f"✅ GPT 'yes' for caption {prefix}")
                await forward_or_send(msg)
                add_to_seen_albums(grouped_ids)
                return
            print(f"❌ GPT 'no' for caption; go to OCR… {prefix}")

        ocr_targets = [
            m for m in album
            if m.photo or (m.document and m.document.mime_type.startswith("image/"))
        ][:2]

        ocr_texts = []
        for idx, m in enumerate(ocr_targets, 1):
            txt = await ocr_photo_to_text(m)
            if txt:
                ocr_texts.append(txt)
            print(f"OCR part {idx}: {txt[:60]}")

        ocr_text = "\n".join(ocr_texts)
        if ocr_text:
            print(f"OCR ({len(ocr_text)} symb): {ocr_text[:80]}")

        if fuzzy_keyword(ocr_text):
            print(f"✅ fuzzy‑keyword (новинки) pass {prefix}")
            await forward_or_send(msg)
            add_to_seen_albums(grouped_ids)
            return

        if ocr_text and await is_relevant(ocr_text):
            print(f"✅ GPT 'yes' from OCR {prefix}")
            await forward_or_send(msg)
            add_to_seen_albums(grouped_ids)
            return
        
        if grouped_ids:
            SEEN_ALBUMS.add(grouped_ids)

        print(f"🚫 filtered {prefix}(caption+OCR = no)")

    finally:
        if grouped_ids:
            IN_PROGRESS.discard(grouped_ids)
            if len(SEEN_ALBUMS) > 50: SEEN_ALBUMS.clear()

async def backfill(hours: int = BACKFILL_HOURS):
    
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    print(f"🔍 backfill {hours}h ( {since.isoformat(timespec='seconds')})")

    total = 0
    for chan in CHANNELS:
        async for msg in client.iter_messages(chan, offset_date=since, reverse=True):
            txt = msg.raw_text or ""
            if txt and await is_relevant(txt):
                await forward_or_send(msg)
                total += 1
                await rate_sleep()

    print(f"Backfill end, sended {total} messages")

async def periodic_ping(interval_min: int = 20):
    """Раз на interval_min хв викликає get_me() щоб тримати TCP активним."""
    while True:
        try:
            await client.get_me()
        except Exception as e:
            print("⚠️ ping error:", e)
        await asyncio.sleep(interval_min * 60)

# End Telethon and bot part

def main():
    async def runner():
        await client.connect()
        print(f"🔌 Connected at {datetime.now().isoformat(timespec='seconds')}")

        if not RUN_BACKFILL:
            await init_channels()
            for ch in CHANNELS:
                try:
                    await client.get_input_entity(ch)
                except Exception as e:
                    print(f"⚠️ can't access to {ch}: {e}")

        if RUN_BACKFILL:
            await backfill(hours=BACKFILL_HOURS)

        print("▶️ Listening…")
        #asyncio.create_task(periodic_ping())
        await client.run_until_disconnected()

    while True:
        try:
            client.loop.run_until_complete(runner())
        except PersistentTimestampOutdatedError as e:
            print(f"⚠️ {e}; reconnecting in 5 s…")
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
