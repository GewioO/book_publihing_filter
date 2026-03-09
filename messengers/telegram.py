import asyncio
import pathlib
import tempfile
import time

import requests
from telethon.errors import FloodWaitError
from telethon.tl.types import MessageEntityTextUrl

import config
from messengers.base import PostPayload


def bot_api(method: str, **params):
    return requests.post(f"{config.BOT_API}/{method}", **params, timeout=60)


def shorten(text: str, limit: int = 1000) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


async def rate_sleep(sec: float = 2.0):
    await asyncio.sleep(sec)


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


def send_text_via_bot(text: str, link: str):
    payload = dict(
        chat_id=config.TARGET_CHAT,
        parse_mode="HTML",
        disable_web_page_preview=False,
        text=f"{text}\n\n<a href=\"{link}\">Джерело</a>",
    )
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
        data = dict(chat_id=config.TARGET_CHAT, caption=caption, parse_mode="HTML")
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


class TelegramMessenger:
    def __init__(self, client):
        self.client = client

    async def publish(self, payload: PostPayload) -> None:
        if config.FORWARD_MODE:
            await self._forward(payload)
        else:
            await self._send_via_bot(payload)

    async def _forward(self, payload: PostPayload) -> None:
        try:
            await self.client.forward_messages(
                config.TARGET_CHAT, payload.album_msgs, payload.source_msg.chat_id
            )
            await rate_sleep()
        except FloodWaitError as e:
            print(f"⏳ FloodWait {e.seconds}s, sleep...")
            await asyncio.sleep(e.seconds + 1)
            await self.client.forward_messages(
                config.TARGET_CHAT, payload.album_msgs, payload.source_msg.chat_id
            )
            await rate_sleep()

    async def _send_via_bot(self, payload: PostPayload) -> None:
        msg = payload.source_msg
        caption_text = payload.caption_text
        link = f"https://t.me/{msg.chat.username}/{msg.id}"
        text_cap = shorten(caption_text) + f"\n\n<a href=\"{link}\">Джерело</a>"
        first = True
        for m in payload.album_msgs:
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
