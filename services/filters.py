import re
import config


def quick_keyword_pass(text: str) -> bool:
    return any(k in re.sub(r'[^\w\s]', '', text.lower()) for k in config.KEYWORDS)


def fuzzy_keyword(text: str) -> bool:
    low = text.lower()
    patterns = [
        r"новинк[аи]",
        r"новинк",
        r"передпродаж",
        r"анонси аудіокнижок",
        r"попереднє замовлення",
    ]
    return any(re.search(p, low) for p in patterns)
