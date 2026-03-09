# -*- coding: utf-8 -*-

from datetime import datetime, timedelta, timezone
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from telethon.errors import PersistentTimestampOutdatedError
from telethon.tl.types import UpdateShort
from telethon.tl.types import MessageEntityTextUrl
from mastodon import Mastodon
import requests, asyncio, time, tempfile, pathlib
import openai
import easyocr, cv2, numpy as np
import aiohttp
import re
import config
import shutil



reader = easyocr.Reader(['uk', 'en'], gpu=False)
client = TelegramClient(config.SESSION, config.API_ID, config.API_HASH)
client.add_event_handler(lambda e: None, events.Raw(types=(UpdateShort,)))
SEEN_ALBUMS: set[int] = set()
IN_PROGRESS: set[int] = set()
SEEN_MSG_IDS: set[int] = set()

config.TEMP_DIR.mkdir(exist_ok=True)
delete_timer_task = None

async def delayed_cleanup(delay=60):
    await asyncio.sleep(delay)
    try:
        shutil.rmtree(config.TEMP_DIR, ignore_errors=True)
        config.TEMP_DIR.mkdir(exist_ok=True)
        print("🧹 Temp files cleaned up.")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

# ----- LLM func

_llm_cache: dict[str, bool] = {}

async def llm_is_relevant(text: str) -> bool:
    if text in _llm_cache:         
        return _llm_cache[text]

    client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
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

# ----- Start Telethon and bot part

async def rate_sleep(sec: float = 2.0):
    await asyncio.sleep(sec)

def bot_api(method: str, **params):
    return requests.post(f"{config.BOT_API}/{method}", **params, timeout=60)

## Special for specific publisher
def fuzzy_keyword(text: str) -> bool:
    low = text.lower()
    patterns = [
        r"новинк[аи]",            
        r"новинк",
        r"передпродаж",
        r"анонси аудіокнижок",
        r"попереднє замовлення",               
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

def reveal_hidden_links(msg) -> str:
    if not msg.raw_text or not msg.entities:
        return msg.raw_text

    result = ""
    last_index = 0

    for entity, txt in msg.get_entities_text():
        if isinstance(entity, MessageEntityTextUrl):
            start = msg.raw_text.find(txt, last_index)
            if start == -1:
                continue 
            end = start + len(txt)
            result += msg.raw_text[last_index:end] + f" ({entity.url})"
            last_index = end
        else:
            result += msg.raw_text[last_index:last_index + len(txt)]
            last_index += len(txt)

    result += msg.raw_text[last_index:]

    return result

async def post_to_mastodon_with_retries(text, image_paths=None, max_retries=5, delay_seconds=10, reply_to_id=None):
    for attempt in range(1, max_retries + 1):
        print(f"Mastodon try #{attempt}")
        try:
            media_ids = []
            if image_paths:
                for img_path in image_paths:
                    try:
                        media = mastodon.media_post(img_path)
                        media_ids.append(media["id"])
                    except Exception as media_error:
                        print(f"⚠️ media_post failed on {img_path}: {media_error}")
                        raise media_error

            status = await asyncio.to_thread(
                mastodon.status_post,
                text,
                media_ids=media_ids if media_ids else None,
                in_reply_to_id=reply_to_id
            )
            return status

        except Exception as e:
            print(f"⚠️ Mastodon post attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"⏳ Retrying whole post in {delay_seconds} seconds...")
                await asyncio.sleep(delay_seconds)
            else:
                print("❌ All attempts to post to Mastodon failed.")
                return None


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

    # Знайти повідомлення з текстом
    msg_with_text = next((m for m in album_msgs if m.raw_text), None)
    caption_text = msg_with_text.raw_text if msg_with_text else "📢 Новий пост"

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
        text_cap = shorten(caption_text) + f"\n\n<a href=\"{link}\">Джерело</a>"
        first = True
        for m in album_msgs:
            if m.photo:
                with tempfile.TemporaryDirectory() as td:
                    p = pathlib.Path(td) / "img.jpg"
                    await m.download_media(file=p)
                    caption = text_cap if first else ""
                    await asyncio.to_thread(send_photo_via_bot, p, caption=caption)
                    first = False
                    await rate_sleep()
        if first:
            await asyncio.to_thread(send_text_via_bot, shorten(caption_text), link)
            await rate_sleep()

    # ==== Mastodon ====
    try:
        print("I'm in Mastodon try: ", caption_text)

        caption_text_with_links = reveal_hidden_links(msg_with_text) if msg_with_text else caption_text
        if chat_name:
            caption_text_with_links += f"\nВід {chat_name}\n#книги"

        image_batches = []
        current_batch = []

        for i, m in enumerate(album_msgs):
            if not m.photo:
                continue

            img_path = config.TEMP_DIR / f"img_{i}.jpg"
            await m.download_media(file=img_path)
            current_batch.append(str(img_path))

            if len(current_batch) == config.MASTODON_MAX_IMAGES:
                image_batches.append(current_batch)
                current_batch = []

        if current_batch:
            image_batches.append(current_batch)
    
        if not image_batches:
            await post_to_mastodon_with_retries(caption_text_with_links)
            return

        reply_to_id = None
        total_parts = len(image_batches)

        for part_index, batch in enumerate(image_batches):
            part_text = caption_text_with_links
            if total_parts > 1:
                part_text += f" 📌 ({part_index + 1}/{total_parts})"

            status = await post_to_mastodon_with_retries(
                part_text,
                image_paths=batch,
                reply_to_id=reply_to_id
            )
            if not status:
                print("❌ Lost part of the thread")
                break
            reply_to_id = status["id"]

        global delete_timer_task
        if delete_timer_task and not delete_timer_task.done():
            delete_timer_task.cancel()
        delete_timer_task = asyncio.create_task(delayed_cleanup())

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
    else:
        if msg.id in SEEN_MSG_IDS:
            return
        SEEN_MSG_IDS.add(msg.id)

    album = await get_album(msg)
    caption_text = next((m.raw_text for m in album if m.raw_text), "") or ""
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
