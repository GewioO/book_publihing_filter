import openai
import config

_llm_cache: dict[str, bool] = {}
_openai_client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)


async def llm_is_relevant(text: str) -> bool:
    if text in _llm_cache:
        return _llm_cache[text]

    try:
        resp = await _openai_client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=0,
            max_tokens=1,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": f'Post:\n"""\n{text[:4000]}\n"""'},
            ],
        )
        decision = resp.choices[0].message.content.strip().lower().startswith("y")
        _llm_cache[text] = decision

        if len(_llm_cache) > 2048:
            _llm_cache.pop(next(iter(_llm_cache)))
        return decision

    except Exception as e:
        print("‼️ LLM error:", e)
        return False
