import re
import config


class FilterService:
    
    @staticmethod
    def quick_keyword_pass(text: str) -> bool:
        if not text:
            print(f"🔑 No text to check")
            return False
        
        text_preview = text[:100] + ("..." if len(text) > 100 else "")
        print(f"🔑 Checking quick keywords for: {text_preview}")
        
        keywords = getattr(config, 'KEYWORDS', ())
        for keyword in keywords:
            if keyword.lower() in text.lower():
                print(f"✅ Found keyword: '{keyword}'")
                return True
        
        print(f"❌ No keywords found")
        return False
    
    @staticmethod
    def fuzzy_keyword(text: str) -> bool:
        if not text:
            print(f"🔍 No text to check")
            return False
        
        text_preview = text[:100] + ("..." if len(text) > 100 else "")
        print(f"🔍 Checking fuzzy keywords for: {text_preview}")
        
        fuzzy_patterns = [
            r"новинк[аи]",
            r"передпродаж",
            r"передзамов",
            r"вже в продажі",
            r"вже доступн[аи]?",
            r"вийшл[аа]? з друку",
            r"у друці",
            r"вийде",
            r"виходить",
            r"анонс",
        ]
        
        for pattern in fuzzy_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                print(f"✅ Fuzzy pattern matched: '{pattern}'")
                return True
        
        print(f"❌ No fuzzy patterns matched")
        return False
    
    @staticmethod
    def shorten(text: str, limit: int = 1000) -> str:
        if not text:
            return ""
        
        if len(text) <= limit:
            return text
        
        shortened = text[:limit] + "…"
        print(f"📝 Text shortened: {len(text)} → {len(shortened)} characters")
        return shortened
