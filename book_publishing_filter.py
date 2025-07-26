# -*- coding: utf-8 -*-

from datetime import datetime, timedelta, timezone
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from telethon.errors import PersistentTimestampOutdatedError
from telethon.tl.types import UpdateShort
from mastodon import Mastodon
import requests, asyncio, time, tempfile, pathlib
from functools import lru_cache
import asyncio, os
import openai
import easyocr, cv2, numpy as np
import aiohttp
import tempfile
from PIL import Image
import re
import config

reader = easyocr.Reader(['uk', 'en'], gpu=False)
client = TelegramClient(config.SESSION, config.API_ID, config.API_HASH)
client.add_event_handler(lambda e: None, events.Raw(types=(UpdateShort,)))
processed_grouped_ids = set()

SEEN_ALBUMS: set[int] = set()
IN_PROGRESS: set[int] = set()

# ----- LLM func

_llm_cache: dict[str, bool] = {}

async def llm_is_relevant(text: str) -> bool:
    if text in _llm_cache:          # ← return correct answer
        return _llm_cache[text]

    client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)   # key from env
    try:
        resp = await client.chat.completions.create(
            model = config.OPENAI_MODEL,
            temperature = 0,
            max_tokens = 1,
            messages = [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user",   "content": f'Post:\n"""\n{text[:4000]}\n"""'}
            ],
        )
        decision = resp.choices[0].message.content.strip().lower().startswith("y")
        _llm_cache[text] = decision        
        
        if len(_llm_cache) > 2048:
            _llm_cache.pop(next(iter(_llm_cache)))
        return decision

    except Exception as e:
        print("‼️ LLM error:", e)
        return False

mastodon = Mastodon(
    access_token = config.MASTODON_ACCESS_TOKEN,
    api_base_url = config.MASTODON_API_BASE_URL
)

def post_to_mastodon(text: str, image_path: str = None):
    print("post_to_matodon: ", text)
    if image_path:
        print("image mastodon")
        media = mastodon.media_post(image_path, mime_type="image/jpeg")
        return mastodon.status_post(text, media_ids=[media])
    return mastodon.status_post(text)

# ----- Start Telethon and bot part

async def rate_sleep(sec: float = 2.0):
    await asyncio.sleep(sec) # pouse between messages in chat

def bot_api(method: str, **params):
    return requests.post(f"{config.BOT_API}/{method}", **params, timeout=60)

## Special for specific publisher
def fuzzy_keyword(text: str) -> bool:
    low = text.lower()
    patterns = [
        r"новинк[аи]",            
        r"новинк",
        r"передпродаж",
        r"анонси аудіокнижок"               
    ]
    return any(re.search(p, low) for p in patterns)

def quick_keyword_pass(text: str) -> bool:
    return any(k in re.sub(r'[^\w\s]', '', text.lower()) for k in config.KEYWORDS)

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

# ----- Start OCR part 

async def ocr_photo_to_text(message_or_url) -> str:
    if isinstance(message_or_url, str):
        return await ocr_from_image_url(message_or_url)
    else:
        raw = await client.download_media(message_or_url, bytes)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        lines = reader.readtext(img, detail=0, paragraph=False)
        return "\n".join(lines).strip()
    
async def ocr_from_image_url(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return ""
                raw = await resp.read()

        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        lines = reader.readtext(img, detail=0, paragraph=False)
        return "\n".join(lines).strip()

    except Exception as e:
        print("❌ ocr_from_image_url error:", e)
        return ""

## Special for albums
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

def extract_telegram_image_links(text):
    pattern = r'https://telegra\.ph/file/\S+\.(?:jpg|jpeg|png|webp)'
    return re.findall(pattern, text or "")

def send_text_via_bot(text: str, link: str):
    payload = dict(chat_id=config.TARGET_CHAT,
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
        data  = dict(chat_id=config.TARGET_CHAT, caption=caption, parse_mode="HTML")
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

async def forward_or_send(msg, chat_name: str = None):
    album_msgs = await get_album(msg)
    
    if config.FORWARD_MODE:
        try:
            await client.forward_messages(config.TARGET_CHAT, album_msgs, msg.chat_id)
            await rate_sleep()
        except FloodWaitError as e:
            print(f"⏳ FloodWait {e.seconds}s, sleep...")
            await asyncio.sleep(e.seconds + 1)
            await client.forward_messages(config.TARGET_CHAT, album_msgs, msg.chat_id)
            await rate_sleep()
    else:
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
        
    try:
        print("I'm in Mastodon try: ", msg.raw_text)
        tempImg = None
        if msg.media:
            await msg.download_media("temp.jpg") 
            tempImg = "temp.jpg"
        text = msg.raw_text or "📢 Новий пост "
        if chat_name:
            text += "від " + chat_name
        post_to_mastodon(text, tempImg)
        os.remove("temp.jpg")
    except Exception as e:
        print(f"⚠️ Mastodon post failed: {e}")

async def init_channels():                
    for ch in config.CHANNELS:
        try:
            await client.get_input_entity(ch)
        except Exception as e:
            print(f"⚠️  Cannot resolve {ch}: {e}")     

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
    
    album = await get_album(msg)  
    caption_text = next((m.raw_text for m in album if m.raw_text), "") or ""
    if msg.grouped_id:
        if msg.grouped_id in processed_grouped_ids:
            return  
        processed_grouped_ids.add(msg.grouped_id)
    try:
        if caption_text and quick_keyword_pass(caption_text):
            print(f"✅ keyword‑pass {prefix}")
            await forward_or_send(msg, chat_name)
            add_to_seen_albums(grouped_ids)
            return

        if caption_text:
            print(f"LLM‑caption → {prefix}{caption_text[:80]}…")
            if await llm_is_relevant(caption_text):
                print(f"✅ GPT 'yes' for caption {prefix}")
                await forward_or_send(msg, chat_name)
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
            txt = await ocr_photo_to_text(m)
            if txt:
                ocr_texts.append(txt)
            print(f"OCR part {idx}: {txt[:60]}")

        ocr_text = "\n".join(ocr_texts)
        full_ocr_text = ocr_text or text_from_img
        
        if full_ocr_text:
            if fuzzy_keyword(full_ocr_text):
                print(f"✅ fuzzy‑keyword (новинки) pass {prefix}")
                await forward_or_send(msg, chat_name)
                add_to_seen_albums(grouped_ids)
                return

            if await llm_is_relevant(full_ocr_text):
                print(f"✅ GPT 'yes' from OCR or link {prefix}")
                await forward_or_send(msg, chat_name)
                add_to_seen_albums(grouped_ids)
                return
        
        if grouped_ids:
            SEEN_ALBUMS.add(grouped_ids)

        print(f"🚫 filtered {prefix}(caption+OCR = no)")

    finally:
        if grouped_ids:
            IN_PROGRESS.discard(grouped_ids)
            if len(SEEN_ALBUMS) > 50: SEEN_ALBUMS.clear()

async def backfill(hours: int = config.BACKFILL_HOURS):
    
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    print(f"🔍 backfill {hours}h ( {since.isoformat(timespec='seconds')})")

    total = 0
    for chan in config.CHANNELS:
        async for msg in client.iter_messages(chan, offset_date=since, reverse=True):
            txt = msg.raw_text or ""
            if txt and await llm_is_relevant(txt):
                await forward_or_send(msg)
                total += 1
                await rate_sleep()

    print(f"Backfill end, sended {total} messages")

# End Telethon and bot part

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
