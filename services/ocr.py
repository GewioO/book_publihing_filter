import easyocr
import cv2
import numpy as np
import aiohttp
import re
from typing import Union


class OCRService:    
    def __init__(self, languages=['uk', 'en']):
        print("🔧 Initializing OCR service...")
        self.reader = easyocr.Reader(languages, gpu=False)
        print("✅ OCR service initialized")
    
    async def extract_from_message(self, telegram_msg) -> str:
        print("🔍 Starting OCR extraction from message...")
        texts = []
        
        # OCR з першого фото тільки
        if telegram_msg.photo or (
            telegram_msg.document
            and telegram_msg.document.mime_type.startswith("image/")
        ):
            print("📸 Found photo/image in message, starting OCR...")
            txt = await self._ocr_from_message(telegram_msg)
            if txt:
                texts.append(txt)
                print(f"✅ OCR from main message: {txt[:100]}...")
        
        url_match = re.search(
            r'https://telegra\.ph/file/\S+\.(jpg|jpeg|png)',
            telegram_msg.text or ""
        )
        if url_match:
            print(f"🔗 Found telegraph link, extracting OCR...")
            txt = await self.extract_from_url(url_match.group(0))
            if txt:
                texts.append(txt)
                print(f"✅ OCR from telegraph URL: {txt[:100]}...")
        
        result = "\n".join(t for t in texts if t)
        print(f"📊 OCR extraction complete. Total text: {len(result)} characters")
        return result
    
    async def _ocr_from_message(self, msg) -> str:
        try:
            print(f"🔬 Running OCR on message {msg.id}...")
            raw = await msg.client.download_media(msg, bytes)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"⚠️ Failed to decode image {msg.id}")
                return ""
            
            print(f"📐 Image shape: {img.shape}")
            lines = self.reader.readtext(img, detail=0, paragraph=False)
            text = "\n".join(lines).strip()
            print(f"✅ OCR completed on message {msg.id}: {len(text)} characters")
            return text
        except Exception as e:
            print(f"❌ OCR error on message {msg.id}: {e}")
            return ""
    
    async def extract_from_url(self, url: str) -> str:
        try:
            print(f"🌐 Downloading image from URL: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        print(f"⚠️ URL returned status {resp.status}")
                        return ""
                    raw = await resp.read()
            
            print(f"📥 Downloaded {len(raw)} bytes from URL")
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"⚠️ Failed to decode image from URL")
                return ""
            
            print(f"📐 Image shape: {img.shape}")
            lines = self.reader.readtext(img, detail=0, paragraph=False)
            text = "\n".join(lines).strip()
            print(f"✅ OCR from URL completed: {len(text)} characters")
            return text
        
        except Exception as e:
            print(f"❌ OCR from URL error: {e}")
            return ""
