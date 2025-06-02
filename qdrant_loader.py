import os
import requests
from typing import List
from pydantic import BaseModel

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.getenv("API_KEY")

class ChatMessage(BaseModel):
    user: str
    bot: str

def generate_answer(question: str, context: str, chat_history: List[ChatMessage]) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "Ты — Профессор КРАБ, эксперт в области права. Отвечай профессионально, чётко и с лёгким юмором."},
        {"role": "system", "content": f"Контекст:\n{context}"}
    ]

    for msg in chat_history:
        messages.append({"role": "user", "content": msg.user})
        messages.append({"role": "assistant", "content": msg.bot})

    messages.append({"role": "user", "content": question})

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": False
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Ошибка: {response.status_code} - {response.text}"
