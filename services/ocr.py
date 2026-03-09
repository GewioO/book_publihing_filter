import easyocr
import cv2
import numpy as np
import aiohttp

reader = easyocr.Reader(['uk', 'en'], gpu=False)


async def ocr_photo_to_text(message_or_url, tg_client=None) -> str:
    if isinstance(message_or_url, str):
        return await ocr_from_image_url(message_or_url)
    else:
        raw = await tg_client.download_media(message_or_url, bytes)
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
