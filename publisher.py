from messengers.base import PostPayload
from messengers.telegram import reveal_hidden_links
import config


class Publisher:
    def __init__(self, *messengers):
        self._messengers = messengers

    async def publish(self, source_msg, album_msgs, chat_name=None):
        msg_with_text = next((m for m in album_msgs if m.raw_text), None)
        caption_text = msg_with_text.raw_text if msg_with_text else "📢 Новий пост"
        caption_text_with_links = (
            reveal_hidden_links(msg_with_text) if msg_with_text else caption_text
        )
        if chat_name:
            caption_text_with_links += f"\nВід {chat_name}\n#книги"

        payload = PostPayload(
            source_msg=source_msg,
            album_msgs=album_msgs,
            caption_text=caption_text,
            caption_text_with_links=caption_text_with_links,
            chat_name=chat_name,
        )

        for messenger in self._messengers:
            await messenger.publish(payload)
