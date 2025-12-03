from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass, field

@dataclass
class Message:
    content: str
    images: List[str] = field(default_factory=list)
    source_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    source_message: Optional[object] = None

class MessengerAdapter(ABC):
    
    @abstractmethod
    async def connect(self):
        pass
    
    @abstractmethod
    async def disconnect(self):
        pass
    
    @abstractmethod
    async def send_message(self, text: str, images: List[str] = None, **kwargs) -> bool:
        pass
    
    @abstractmethod
    async def start_listening(self, callback):
        pass
