import asyncio
import shutil

from mastodon import Mastodon

import config
from messengers.base import PostPayload


class MastodonMessenger:
    def __init__(self):
        self._client = Mastodon(
            access_token=config.MASTODON_ACCESS_TOKEN,
            api_base_url=config.MASTODON_API_BASE_URL,
        )
        self._delete_timer_task = None

    async def publish(self, payload: PostPayload) -> None:
        try:
            print("I'm in Mastodon try: ", payload.caption_text)

            image_batches = []
            current_batch = []

            for i, m in enumerate(payload.album_msgs):
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
                await self._post_with_retries(payload.caption_text_with_links)
                return

            reply_to_id = None
            total_parts = len(image_batches)

            for part_index, batch in enumerate(image_batches):
                part_text = payload.caption_text_with_links
                if total_parts > 1:
                    part_text += f" 📌 ({part_index + 1}/{total_parts})"

                status = await self._post_with_retries(
                    part_text,
                    image_paths=batch,
                    reply_to_id=reply_to_id,
                )
                if not status:
                    print("❌ Lost part of the thread")
                    break
                reply_to_id = status["id"]

            if self._delete_timer_task and not self._delete_timer_task.done():
                self._delete_timer_task.cancel()
            self._delete_timer_task = asyncio.create_task(self._delayed_cleanup())

        except Exception as e:
            print(f"⚠️ Mastodon post failed: {e}")

    async def _post_with_retries(
        self, text, image_paths=None, max_retries=5, delay_seconds=10, reply_to_id=None
    ):
        for attempt in range(1, max_retries + 1):
            print(f"Mastodon try #{attempt}")
            try:
                media_ids = []
                if image_paths:
                    for img_path in image_paths:
                        try:
                            media = self._client.media_post(img_path)
                            media_ids.append(media["id"])
                        except Exception as media_error:
                            print(f"⚠️ media_post failed on {img_path}: {media_error}")
                            raise media_error

                status = await asyncio.to_thread(
                    self._client.status_post,
                    text,
                    media_ids=media_ids if media_ids else None,
                    in_reply_to_id=reply_to_id,
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

    async def _delayed_cleanup(self, delay=60):
        await asyncio.sleep(delay)
        try:
            shutil.rmtree(config.TEMP_DIR, ignore_errors=True)
            config.TEMP_DIR.mkdir(exist_ok=True)
            print("🧹 Temp files cleaned up.")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
