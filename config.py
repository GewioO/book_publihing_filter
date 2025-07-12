import json

with open("keys.json", "r", encoding="utf-8") as f:
    keys = json.load(f)
with open("channels.json", encoding="utf-8") as f:
    CHANNELS = json.load(f)["channels"]

API_ID   = keys["api_id"]
API_HASH = keys["api_hash"]
SESSION  = "books_monitor"

TARGET_CHAT = "@new_book_filter"               
BOT_TOKEN   = keys["bot_token"]
BOT_API     = f"https://api.telegram.org/bot{BOT_TOKEN}"
OPENAI_API_KEY = keys["openai_api_key"]
OPENAI_MODEL = "gpt-4o-mini"

BACKFILL_HOURS = 1
RUN_BACKFILL = True 
FORWARD_MODE = True 

KEYWORDS = ("передзамов", "у друці")

SYSTEM_PROMPT = (
    # --- Role ---
    "You are a binary classifier that must answer ONLY \"yes\" or \"no\".\n\n"

    # --- Task ---
    "Decide whether the Telegram post below FACTUALLY announces a specific new book "
    "or books in any of the following ways:\n"
    " • preorder (передзамовлення) has just started or is still open;\n"
    " • last day / final hours of an ongoing preorder;\n"
    " • a new book (paper, e‑book, or audiobook) has just gone on sale;\n"
    " • new books (paper, e‑book, or audiobook) will be sale in this mounth;\n"
    " • a concrete forthcoming title is officially announced (анонс) AND preorder / sale info is given.\n\n"

    # --- (YES) ---
    "Treat posts as *YES* if they contain phrases like:\n"
    "  - “Передзамовлення відкрите”, “Старт передзамовлення”, “Уже у передпродажі”;\n"
    "  - “Передзамовлення триває до кінця дня”, “Останній день передзамовлення”;\n"
    "  - “Новинка вже у продажу / у е‑форматі / в аудіо”; “Вийшла з друку”, “Книга вийшла”, “Новинки <назва місяця>“;\n"
    "  - “Анонс книги <Назва>: уже можна купити / доступна до замовлення”.\n"
    "  - If OCR contains the Ukrainian word “новинки” (new releases)\n\n"

    # --- (NO) ---
    "Treat posts as *NO* if they are only:\n"
    "  - future schedules, calendars, vague plans (e.g. “Плани передзамовлень на осінь”);\n"
    "  - events, presentations, conferences, discounts, giveaways, memes, reviews, quotes;\n"
    "  - general marketing without stating that preorder or sale is already open.\n\n"

    # --- Details ---
    "The post text may include OCR output extracted from images — read it as well.\n"
    "If the post is ambiguous or you're uncertain, answer “no”.\n\n"

    # --- Answer form ---
    "Answer strictly “yes” or “no”. No explanations."
)