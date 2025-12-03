# -*- coding: utf-8 -*-

import asyncio
import tempfile
import pathlib
from typing import Optional, List, Callable
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
import requests
import config
from .base import MessengerAdapter, Message


class TelegramAdapter(MessengerAdapter):
    
    def __init__(self):
        self.client = TelegramClient(
            config.SESSION,
            config.API_ID,
            config.API_HASH
        )
        self._message_callback: Optional[Callable] = None
        self._rate_limit = 2.0
        self._seen_albums = set()
    
    async def connect(self):
        await self.client.connect()
        await self._init_channels()
        print("🔌 Telegram connected")
    
    async def disconnect(self):
        try:
            await self.client.disconnect()
        except:
            pass
    
    async def _init_channels(self):
        for ch in config.CHANNELS:
            try:
                await self.client.get_input_entity(ch)
            except Exception as e:
                print(f"⚠️  Cannot resolve {ch}: {e}")
    
    async def send_message(
        self,
        text: str,
        images: List[str] = None,
        **kwargs
    ) -> bool:
        try:
            source_msg = kwargs.get("source_message")
            source_url = kwargs.get("source_url")
            
            if config.FORWARD_MODE and source_msg:
                await self._forward_message(source_msg)
            else:
                await self._send_via_bot(text, source_msg, source_url)
            
            return True
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")
            return False
    
    async def _forward_message(self, msg):
        try:
            album_msgs = await self._get_album(msg)
            print(f"[TELEGRAM] Forwarding {len(album_msgs)} messages...")
            
            await self.client.forward_messages(
                config.TARGET_CHAT,
                album_msgs,
                msg.chat_id
            )
            await asyncio.sleep(self._rate_limit)
            print(f"[TELEGRAM] ✅ Forwarded {len(album_msgs)} messages")
        
        except FloodWaitError as e:
            print(f"[TELEGRAM] ⏳ FloodWait {e.seconds}s, sleep...")
            await asyncio.sleep(e.seconds + 1)
            await self._forward_message(msg)
        except Exception as e:
            print(f"[TELEGRAM] ⚠️ Forward error: {e}")
    
    async def _send_via_bot(self, text: str, msg=None, source_url: str = None):
        try:
            if msg and msg.photo:
                album_msgs = await self._get_album(msg)
                link = f"https://t.me/{msg.chat.username}/{msg.id}"
                caption = self._shorten(text) + f"\n\n<a href=\"{link}\">Джерело</a>"
                
                print(f"[TELEGRAM] Sending {len(album_msgs)} photos via bot API...")
                
                first = True
                for m in album_msgs:
                    if m.photo:
                        with tempfile.TemporaryDirectory() as td:
                            p = pathlib.Path(td) / "img.jpg"
                            await m.download_media(file=p)
                            await self._send_photo(p, caption if first else "")
                            first = False
                            await asyncio.sleep(self._rate_limit)
                
                if first:  
                    await self._send_text(text, link)
            else:
                link = source_url or "https://example.com"
                await self._send_text(text, link)
        
        except Exception as e:
            print(f"[TELEGRAM] ⚠️ Send via bot error: {e}")
    
    async def _send_text(self, text: str, link: Optional[str] = None):
        payload = dict(
            chat_id=config.TARGET_CHAT,
            parse_mode="HTML",
            disable_web_page_preview=False,
            text=f"{text}\n\n<a href=\"{link}\">Джерело</a>" if link else text
        )
        
        attempt = 0
        while attempt < 5:
            try:
                r = requests.post(
                    f"{config.BOT_API}/sendMessage",
                    json=payload,
                    timeout=60
                )
                if r.ok:
                    print("[TELEGRAM] ✅ Text message sent")
                    await asyncio.sleep(self._rate_limit)
                    return
                
                if r.status_code == 429:
                    retry = r.json().get("parameters", {}).get("retry_after", 10)
                    print(f"[TELEGRAM] 429, waiting {retry}s…")
                    await asyncio.sleep(retry + 1)
                    attempt += 1
                else:
                    print(f"[TELEGRAM] ❌ BOT ERROR: {r.text}")
                    return
            except Exception as e:
                print(f"[TELEGRAM] ⚠️ Send text error: {e}")
                return
    
    async def _send_photo(self, photo_path: pathlib.Path, caption: str = ""):
        try:
            with photo_path.open("rb") as f:
                files = {"photo": f}
                data = dict(
                    chat_id=config.TARGET_CHAT,
                    caption=caption,
                    parse_mode="HTML"
                )
                
                attempt = 0
                while attempt < 5:
                    r = requests.post(
                        f"{config.BOT_API}/sendPhoto",
                        data=data,
                        files=files,
                        timeout=60
                    )
                    if r.ok:
                        print("[TELEGRAM] ✅ Photo sent")
                        return
                    
                    if r.status_code == 429:
                        retry = r.json().get("parameters", {}).get("retry_after", 10)
                        await asyncio.sleep(retry + 1)
                        attempt += 1
                    else:
                        print(f"[TELEGRAM] ❌ BOT PHOTO ERROR: {r.text}")
                        return
        except Exception as e:
            print(f"[TELEGRAM] ⚠️ Send photo error: {e}")
    
    async def _get_album(self, msg):
        if not msg.grouped_id:
            return [msg]
        
        print(f"[TELEGRAM] Searching for album {msg.grouped_id}...")
        
        siblings = []
        async for sib in self.client.iter_messages(
            msg.chat_id,
            min_id=msg.id - 50,
            max_id=msg.id + 50,
        ):
            if sib.grouped_id == msg.grouped_id:
                siblings.append(sib)
        
        siblings = list({m.id: m for m in siblings}.values())
        siblings.sort(key=lambda m: m.id)
        
        print(f"[TELEGRAM] Found {len(siblings)} messages in album")
        return siblings
    
    @staticmethod
    def _shorten(text: str, limit: int = 1000) -> str:
        return text if len(text) <= limit else text[:limit] + "…"
    
    async def start_listening(self, callback: Callable):
        self._message_callback = callback
        
        processing_albums = set()
        
        @self.client.on(events.NewMessage(chats=config.CHANNELS))
        async def handler(event):
            msg = event.message
            grouped_ids = msg.grouped_id
            
            if grouped_ids:
                if grouped_ids not in self._seen_albums and grouped_ids not in processing_albums:
                    print(f"[TELEGRAM] 📩 First message of album {grouped_ids} received")
                    print(f"[TELEGRAM] ⏳ Waiting 2 seconds for other messages to arrive...")
                    processing_albums.add(grouped_ids)
                    
                    await asyncio.sleep(2)
                    
                    print(f"[TELEGRAM] 🔄 Processing album {grouped_ids}...")
                    try:
                        msg_obj = Message(
                            content=msg.raw_text or "",
                            source_url=f"https://t.me/{event.chat.username}/{msg.id}",
                            source_message=msg,
                            metadata={
                                "chat_name": event.chat.title,
                                "chat_username": event.chat.username,
                                "grouped_id": grouped_ids,
                            }
                        )
                        await callback(msg_obj, "telegram")
                    except Exception as e:
                        print(f"[TELEGRAM] ❌ Handler error: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        self._seen_albums.add(grouped_ids)
                        processing_albums.discard(grouped_ids)
                        print(f"[TELEGRAM] ✅ Album {grouped_ids} processed")
                else:
                    print(f"[TELEGRAM] ⏭️ Skipping message from already processed/processing album {grouped_ids}")
                    return
            else:
                msg_key = (msg.chat_id, msg.id)
                if msg_key in self._seen_albums:
                    print(f"[TELEGRAM] ⏭️ Skipping already processed message {msg_key}")
                    return
                
                print(f"[TELEGRAM] 📩 Single message {msg_key} received")
                try:
                    msg_obj = Message(
                        content=msg.raw_text or "",
                        source_url=f"https://t.me/{event.chat.username}/{msg.id}",
                        source_message=msg,
                        metadata={
                            "chat_name": event.chat.title,
                            "chat_username": event.chat.username,
                            "grouped_id": None,
                        }
                    )
                    await callback(msg_obj, "telegram")
                except Exception as e:
                    print(f"[TELEGRAM] ❌ Handler error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self._seen_albums.add(msg_key)
            
            if len(self._seen_albums) > 200:
                self._seen_albums = set(list(self._seen_albums)[-100:])
        
        print("▶️ Telegram listening…")
        await self.client.run_until_disconnected()
