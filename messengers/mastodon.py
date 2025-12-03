# -*- coding: utf-8 -*-

import asyncio
import os
from typing import Optional, List, Callable
from mastodon import Mastodon, MastodonNetworkError
import config
from .base import MessengerAdapter, Message


class MastodonAdapter(MessengerAdapter):
    MAX_MEDIA_SIZE = 41943040  
    MAX_MEDIA_PER_POST = 4
    UPLOAD_TIMEOUT = 120  
    POST_TIMEOUT = 60     
    
    def __init__(self):
        self.client = Mastodon(
            access_token=config.MASTODON_ACCESS_TOKEN,
            api_base_url=config.MASTODON_API_BASE_URL
        )
        self._message_callback: Optional[Callable] = None
        self._max_retries = 3
        self._retry_delay = 2
    
    async def connect(self):
        try:
            self.client.account_verify_credentials()
            print("🔌 Mastodon connected")
        except Exception as e:
            print(f"❌ Mastodon connection failed: {e}")
            raise
    
    async def disconnect(self):
        pass
    
    async def send_message(
        self,
        text: str,
        images: List[str] = None,
        **kwargs
    ) -> bool:
        reply_to_id = kwargs.get("reply_to_id")
        
        print(f"\n[MASTODON] 📨 send_message called with text='{text[:50]}...' and {len(images or [])} images")
        
        if not images:
            print(f"[MASTODON] No images, sending text only")
            return await self._post_text(text, reply_to_id)
        
        valid_media = []
        for media_path in images:
            if not os.path.exists(media_path):
                print(f"[MASTODON] ⚠️ File not found: {media_path}")
                continue
            
            file_size = os.path.getsize(media_path)
            if file_size > self.MAX_MEDIA_SIZE:
                print(f"[MASTODON] ⚠️ File too large ({file_size} bytes): {media_path}")
                continue
            
            valid_media.append(media_path)
        
        if not valid_media:
            print("[MASTODON] ❌ No valid media files")
            return await self._post_text(text, reply_to_id)
        
        print(f"[MASTODON] 📊 Validated {len(valid_media)} media files")
        
        photos = [m for m in valid_media if m.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
        videos = [m for m in valid_media if m.lower().endswith(('.mp4', '.webm', '.mov'))]
        
        print(f"[MASTODON] 🖼️ {len(photos)} photos, 🎬 {len(videos)} videos")
        
        photo_batches = [
            photos[i:i + self.MAX_MEDIA_PER_POST]
            for i in range(0, len(photos), self.MAX_MEDIA_PER_POST)
        ] if photos else []
        
        video_batches = [[v] for v in videos]
        
        all_batches = photo_batches + video_batches
        total_parts = len(all_batches)
        
        if total_parts == 0:
            print("[MASTODON] ⚠️ No media to post")
            return await self._post_text(text, reply_to_id)
        
        print(f"[MASTODON] 📦 Split into {total_parts} batch(es): {len(photo_batches)} фото + {len(video_batches)} відео")
        
        for part_index, batch in enumerate(all_batches):
            part_text = text
            if total_parts > 1:
                part_text += f" 📌 ({part_index + 1}/{total_parts})"
            
            is_video = batch[0].lower().endswith(('.mp4', '.webm', '.mov'))
            media_type = "🎬 video" if is_video else f"🖼️ photo({len(batch)})"
            print(f"\n[MASTODON] 📝 Processing batch {part_index + 1}/{total_parts} with {len(batch)} {media_type}")
            print(f"[MASTODON] 📄 Text: '{part_text[:60]}...'")
            
            try:
                status = await asyncio.wait_for(
                    self._post_with_media(part_text, batch, reply_to_id),
                    timeout=600 
                )
                
                if status:
                    print(f"[MASTODON] ✅ Batch {part_index + 1} posted successfully")
                    reply_to_id = status["id"]
                    await asyncio.sleep(1)
                else:
                    print(f"[MASTODON] ❌ Failed to post batch {part_index + 1}")
            
            except asyncio.TimeoutError:
                print(f"[MASTODON] ⏱️ Batch {part_index + 1} timed out, skipping...")
            except Exception as e:
                print(f"[MASTODON] ❌ Error posting batch {part_index + 1}: {e}")
        
        return True
    
    async def _post_text(self, text: str, reply_to_id: Optional[str] = None) -> bool:
        print(f"[MASTODON] _post_text called with text='{text}'")
        
        for attempt in range(1, self._max_retries + 1):
            try:
                print(f"[MASTODON] 📝 Posting text (attempt {attempt})...")
                self.client.status_post(
                    text,
                    in_reply_to_id=reply_to_id
                )
                print(f"[MASTODON] ✅ Text post sent")
                return True
            
            except Exception as e:
                print(f"[MASTODON] ⚠️ Attempt {attempt} failed: {e}")
                
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay)
        
        return False
    
    def _get_mime_type(self, file_path: str) -> str:
        ext = file_path.lower().split(".")[-1]
        
        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "mp4": "video/mp4",
            "webm": "video/webm",
            "mov": "video/quicktime",
        }
        
        return mime_types.get(ext, "application/octet-stream")
    
    async def _upload_media_single(self, media_path: str) -> Optional[dict]:
        try:
            mime_type = self._get_mime_type(media_path)
            file_size = os.path.getsize(media_path)
            filename = os.path.basename(media_path)
            
            is_video = filename.lower().endswith(('.mp4', '.webm', '.mov'))
            
            print(f"[MASTODON] 📤 Uploading ({file_size} bytes): {filename}")
            
            loop = asyncio.get_event_loop()
            media = await asyncio.wait_for(
                loop.run_in_executor(None, 
                    lambda: self.client.media_post(
                        media_path,
                        mime_type=mime_type
                    )
                ),
                timeout=self.UPLOAD_TIMEOUT
            )
            
            print(f"[MASTODON] ✅ Media uploaded, ID: {media['id']}")
            return media
        
        except asyncio.TimeoutError:
            print(f"[MASTODON] ⏱️ Upload timeout for {os.path.basename(media_path)}")
            return None
        except Exception as e:
            print(f"[MASTODON] ❌ Upload failed: {e}")
            return None
    
    async def _post_with_media(
        self,
        text: str,
        media_paths: List[str],
        reply_to_id: Optional[str] = None
    ) -> Optional[dict]:
        print(f"[MASTODON] _post_with_media called with text='{text[:50]}...' and {len(media_paths)} files")
        
        is_video = any(m.lower().endswith(('.mp4', '.webm', '.mov')) for m in media_paths)
        
        for attempt in range(1, self._max_retries + 1):
            try:
                media_ids = []
                
                print(f"[MASTODON] 📥 Uploading {len(media_paths)} media (attempt {attempt})...")
                
                for idx, media_path in enumerate(media_paths):
                    print(f"[MASTODON] [{idx + 1}/{len(media_paths)}]", end=" ")
                    media = await self._upload_media_single(media_path)
                    
                    if media:
                        media_ids.append(media["id"])
                    else:
                        print(f"[MASTODON] ⚠️ Skipped: {os.path.basename(media_path)}")
                    
                    await asyncio.sleep(0.5)
                
                if not media_ids:
                    print("[MASTODON] ⚠️ No media uploaded successfully")
                    print(f"[MASTODON] 📝 Sending text-only post as fallback with text: '{text}'")
                    
                    loop = asyncio.get_event_loop()
                    status = await asyncio.wait_for(
                        loop.run_in_executor(None,
                            lambda: self.client.status_post(
                                text,
                                in_reply_to_id=reply_to_id
                            )
                        ),
                        timeout=self.POST_TIMEOUT
                    )
                    print(f"[MASTODON] ✅ Fallback text post sent")
                    return status
                
                if is_video:
                    print(f"[MASTODON] ⏳ Waiting 10 seconds for video processing...")
                    await asyncio.sleep(10)
                else:
                    print(f"[MASTODON] ⏳ Waiting 2 seconds before posting...")
                    await asyncio.sleep(2)
                
                print(f"[MASTODON] 📝 Posting with {len(media_ids)} media and text: '{text[:50]}...'")
                
                loop = asyncio.get_event_loop()
                status = await asyncio.wait_for(
                    loop.run_in_executor(None,
                        lambda: self.client.status_post(
                            text,
                            media_ids=media_ids,
                            in_reply_to_id=reply_to_id
                        )
                    ),
                    timeout=self.POST_TIMEOUT
                )
                
                print(f"[MASTODON] ✅ Post sent with {len(media_ids)} media")
                return status
            
            except asyncio.TimeoutError as e:
                print(f"[MASTODON] ⏱️ Request timed out (attempt {attempt}): {e}")
                if attempt < self._max_retries:
                    wait_time = 5 * attempt
                    print(f"[MASTODON] ⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
            
            except Exception as e:
                error_msg = str(e)
                
                if "файли, оброблення яких ще не закінчилося" in error_msg or "processing" in error_msg.lower():
                    print(f"[MASTODON] ⏳ Media still processing, waiting longer...")
                    if attempt < self._max_retries:
                        wait_time = 10 * attempt
                        print(f"[MASTODON] ⏳ Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                else:
                    print(f"[MASTODON] ⚠️ Error (attempt {attempt}): {e}")
                    if attempt < self._max_retries:
                        await asyncio.sleep(self._retry_delay * 2)
        
        return None
    
    async def start_listening(self, callback: Callable):
        """Mastodon send-only"""
        print("⚠️ Mastodon adapter is send-only")
        pass
