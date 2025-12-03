import asyncio
import time
import shutil
from typing import Dict

import config
from messengers.telegram import TelegramAdapter
from messengers.mastodon import MastodonAdapter
from services.ocr import OCRService
from services.llm import LLMService
from services.filters import FilterService


class BotOrchestrator:    
    def __init__(self):
        print("=" * 60)
        print("🤖 INITIALIZING BOT ORCHESTRATOR")
        print("=" * 60)
        
        self.adapters: Dict[str, object] = {
            "telegram": TelegramAdapter(),
            "mastodon": MastodonAdapter(),
        }
        self.ocr_service = OCRService()
        self.llm_service = LLMService()
        self.filters = FilterService()
        self.delete_timer_task = None
        self.is_running = True
    
    async def initialize(self):
        print("\n🚀 INITIALIZING ADAPTERS")
        for name, adapter in self.adapters.items():
            try:
                await adapter.connect()
                print(f"✅ {name.capitalize()} initialized")
            except Exception as e:
                print(f"❌ {name.capitalize()} initialization failed: {e}")
                raise
    
    async def shutdown(self):
        print("\n🛑 SHUTTING DOWN ADAPTERS")
        self.is_running = False
        for name, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
                print(f"✅ {name.capitalize()} disconnected")
            except Exception as e:
                print(f"⚠️ {name.capitalize()} disconnect error: {e}")
    
    async def process_message(self, message, source: str):
        print("\n" + "=" * 60)
        print(f"📩 NEW MESSAGE FROM {source.upper()}")
        print("=" * 60)
        
        try:
            content_preview = message.content[:100] if message.content else "(empty)"
            print(f"📝 Message content: {content_preview}...")
            
            print("\n[STEP 1] Quick keyword check")
            if self.filters.quick_keyword_pass(message.content):
                print(f"✅ MATCHED: keyword-pass")
                await self.publish_message(message)
                return
            
            print("\n[STEP 2] LLM check on caption")
            if message.content and await self.llm_service.is_relevant(message.content):
                print(f"✅ MATCHED: LLM relevant (caption)")
                await self.publish_message(message)
                return
            
            print("\n[STEP 3] OCR extraction and LLM check")
            if message.source_message:
                ocr_text = await self.ocr_service.extract_from_message(
                    message.source_message
                )
                
                if ocr_text:
                    if self.filters.fuzzy_keyword(ocr_text):
                        print(f"✅ MATCHED: fuzzy-keyword pass (OCR)")
                        await self.publish_message(message)
                        return
                    
                    if await self.llm_service.is_relevant(ocr_text):
                        print(f"✅ MATCHED: LLM relevant (OCR)")
                        await self.publish_message(message)
                        return
            
            print(f"\n🚫 FILTERED OUT: message doesn't match any criteria")
            print("=" * 60)
        
        except Exception as e:
            print(f"\n❌ Error processing message: {e}")
            import traceback
            traceback.print_exc()
    
    async def _download_media(self, source_msg) -> list:
        try:
            if not source_msg:
                print("[MEDIA] ❌ No source message")
                return []
            
            print("[MEDIA] Getting album messages...")
            telegram_adapter = self.adapters["telegram"]
            album_msgs = await telegram_adapter._get_album(source_msg)
            
            print(f"[MEDIA] Album contains {len(album_msgs)} messages")
            
            media_paths = []
            config.TEMP_DIR.mkdir(exist_ok=True)
            
            print(f"[MEDIA] Processing all {len(album_msgs)} messages...")
            for idx, m in enumerate(album_msgs):
                if not self.is_running:
                    print("[MEDIA] ⚠️ Bot shutting down, stopping download")
                    break
                
                if m.photo:
                    try:
                        perm_path = config.TEMP_DIR / f"img_{m.id}.jpg"
                        print(f"[MEDIA] Downloading photo {idx + 1}/{len(album_msgs)}...")
                        await m.download_media(file=perm_path)
                        media_paths.append(str(perm_path))
                        print(f"[MEDIA] ✅ Downloaded image {idx + 1}")
                    except Exception as e:
                        print(f"[MEDIA] ❌ Failed to download photo: {e}")
                
                elif m.video:
                    try:
                        perm_path = config.TEMP_DIR / f"video_{m.id}.mp4"
                        print(f"[MEDIA] Downloading video {idx + 1}/{len(album_msgs)}...")
                        await m.download_media(file=perm_path)
                        media_paths.append(str(perm_path))
                        print(f"[MEDIA] ✅ Downloaded video {idx + 1}")
                    except Exception as e:
                        print(f"[MEDIA] ❌ Failed to download video: {e}")
            
            print(f"[MEDIA] 📊 Successfully downloaded {len(media_paths)} media files")
            return media_paths
        
        except Exception as e:
            print(f"[MEDIA] ❌ Download media error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _cleanup_media(self, delay=180):
        print(f"\n[CLEANUP] Scheduling cleanup in {delay} seconds...")
        await asyncio.sleep(delay)
        try:
            shutil.rmtree(config.TEMP_DIR, ignore_errors=True)
            config.TEMP_DIR.mkdir(exist_ok=True)
            print("[CLEANUP] ✅ Temp media cleaned up")
        except Exception as e:
            print(f"[CLEANUP] ⚠️ Cleanup error: {e}")
    
    async def publish_message(self, message):
        print("\n" + "=" * 60)
        print("📤 PUBLISHING MESSAGE")
        print("=" * 60)
        
        try:
            text = self.filters.shorten(message.content) if message.content else ""
            
            if not text and message.metadata.get("chat_name"):
                text = f"📌 Від {message.metadata.get('chat_name')}"
            
            print(f"📝 Final text: '{text}' ({len(text)} chars)")
            
            print("\n[PUBLISH] Sending to Telegram FIRST...")
            telegram_result = await self.adapters["telegram"].send_message(
                text,
                source_message=message.source_message,
                source_url=message.source_url,
            )
            if telegram_result:
                print("[PUBLISH] ✅ Telegram sent successfully")
            else:
                print("[PUBLISH] ❌ Telegram send failed")
            
            print("\n[PUBLISH] Downloading media for other platforms...")
            media_paths = await self._download_media(message.source_message)
            
            if not media_paths:
                print("[PUBLISH] ⚠️ No media to send to other platforms")
                return
            
            print("\n[PUBLISH] Waiting before Mastodon send...")
            await asyncio.sleep(2)
            
            chat_name = message.metadata.get("chat_name")
            mastodon_text = text
            if chat_name and "#книги" not in mastodon_text:
                mastodon_text += f"\n\nВід {chat_name}\n#книги"
            
            print(f"📝 Mastodon text: '{mastodon_text}' ({len(mastodon_text)} chars)")
            
            print("[PUBLISH] Sending to Mastodon...")
            mastodon_result = await asyncio.wait_for(
                self.adapters["mastodon"].send_message(
                    mastodon_text,
                    images=media_paths,
                ),
                timeout=600  
            )
            if mastodon_result:
                print("[PUBLISH] ✅ Mastodon sent successfully")
            else:
                print("[PUBLISH] ❌ Mastodon send failed")
            
            if self.delete_timer_task and not self.delete_timer_task.done():
                self.delete_timer_task.cancel()
            self.delete_timer_task = asyncio.create_task(self._cleanup_media(delay=180))
            
            print("\n" + "=" * 60)
            print("✅ MESSAGE PUBLISHED")
            print("=" * 60 + "\n")
        
        except asyncio.TimeoutError:
            print("\n❌ Mastodon request timed out (>600s)")
            print("[PUBLISH] Continuing to next message...")
        except Exception as e:
            print(f"\n❌ Error publishing message: {e}")
            import traceback
            traceback.print_exc()
    
    async def run(self):
        try:
            await self.initialize()
            print("\n🚀 Bot started successfully")
            
            listener_task = asyncio.create_task(
                self.adapters["telegram"].start_listening(self.process_message)
            )
            
            await listener_task
        
        except asyncio.CancelledError:
            print("\n🛑 Bot listener cancelled")
        except Exception as e:
            print(f"\n❌ Bot error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.shutdown()


async def main_async():
    bot = BotOrchestrator()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n👋 Keyboard interrupt, shutting down…")
        await bot.shutdown()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 Shutting down…")


if __name__ == "__main__":
    main()
