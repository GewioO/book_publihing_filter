from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PostPayload:
    source_msg: Any
    album_msgs: list
    caption_text: str
    caption_text_with_links: str
    chat_name: Optional[str] = None
