import os
from openai import OpenAI

def setup_openai_api():
    """Настройка OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_key:
        raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY")

    if not api_base:
        raise ValueError("Необходимо установить переменную окружения OPENAI_API_BASE")
    
    client = OpenAI(api_key=api_key, base_url=api_base)
    return client

