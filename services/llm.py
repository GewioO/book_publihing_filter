import openai
import config


class LLMService:
    
    def __init__(self):
        print("🧠 Initializing LLM service...")
        self._cache: dict[str, bool] = {}
        self.max_cache_size = 2048
        print("✅ LLM service initialized")
    
    async def is_relevant(self, text: str) -> bool:
        text_preview = text[:100] + ("..." if len(text) > 100 else "")
        
        if text in self._cache:
            result = self._cache[text]
            print(f"💾 Cache hit: LLM returned '{result}' for: {text_preview}")
            return result
        
        print(f"🤖 Starting LLM check for: {text_preview}")
        
        client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        try:
            print(f"📤 Sending request to {config.OPENAI_MODEL}...")
            resp = await client.chat.completions.create(
                model=config.OPENAI_MODEL,
                temperature=0,
                max_tokens=10,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": f'Post:\n"""\n{text[:4000]}\n"""'},
                ],
            )
            
            response_text = resp.choices[0].message.content.strip().lower()
            print(f"📥 LLM response: '{response_text}'")
            
            decision = response_text.startswith("y")
            print(f"✅ LLM decision: {'YES (relevant)' if decision else 'NO (not relevant)'}")
            
            self._cache[text] = decision
            
            if len(self._cache) > self.max_cache_size:
                removed_key = next(iter(self._cache))
                self._cache.pop(removed_key)
                print(f"🗑️ Cache cleanup: removed oldest entry (cache size: {len(self._cache)})")
            
            print(f"💾 Cached: {len(self._cache)}/{self.max_cache_size}")
            return decision
        
        except Exception as e:
            print(f"❌ LLM error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def clear_cache(self):
        print(f"🗑️ Clearing LLM cache ({len(self._cache)} entries)")
        self._cache.clear()
