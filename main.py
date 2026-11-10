"""Вебхук навыка Алисы «Дипсик».

Принимает запрос от Яндекс Станции, отправляет реплику в DeepSeek API
и возвращает ответ для озвучки Алисой.

Запуск локально:  uvicorn main:app --host 0.0.0.0 --port 10000

Переменные окружения:
  DEEPSEEK_API_KEY — ключ с https://platform.deepseek.com (обязательно)
  DEEPSEEK_MODEL   — модель (по умолчанию deepseek-chat)
  MAX_TOKENS       — максимум токенов в ответе (по умолчанию 700)
  CALL_TIMEOUT     — сколько секунд ждать ответ модели, чтобы уложиться
                     в лимит Алисы 4,5 с (по умолчанию 4,0)
"""

import asyncio
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "700"))
CALL_TIMEOUT = float(os.environ.get("CALL_TIMEOUT", "4.0"))

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

SYSTEM_PROMPT = (
    "Ты — DeepSeek, умный голосовой ассистент, который работает через Яндекс Станцию. "
    "Отвечай на русском языке, кратко и по делу: длинные тексты голосом слушать тяжело. "
    "Не используй разметку, символы списков и эмодзи — говори естественными фразами."
)

EXIT_WORDS = {"выход", "выйти", "завершить", "заверши", "хватит", "стоп", "пока"}

GREETING = (
    "Привет! Я Дипсик. Спрашивай что угодно, я отвечу голосом. "
    "Если я попрошу сказать «дальше», просто скажи «дальше». "
    "Чтобы выйти, скажи «выход»."
)

# История диалога по session_id и фоновые вызовы DeepSeek.
history: dict[str, list[dict]] = {}
last_activity: dict[str, float] = {}
pending: dict[str, asyncio.Task] = {}


def alice_response(text: str, end_session: bool = False) -> JSONResponse:
    return JSONResponse(
        {
            "response": {"text": text, "tts": text, "end_session": end_session},
            "version": "1.0",
        }
    )


async def call_deepseek(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def prune_history() -> None:
    """Удаляем старые сессии, чтобы память не росла бесконечно."""
    now = time.time()
    stale = [sid for sid, ts in last_activity.items() if now - ts > 3600]
    for sid in stale:
        history.pop(sid, None)
        last_activity.pop(sid, None)
        pending.pop(sid, None)
    if len(history) > 500:
        oldest = sorted(last_activity, key=last_activity.get)[:250]
        for sid in oldest:
            history.pop(sid, None)
            last_activity.pop(sid, None)
            pending.pop(sid, None)


@app.get("/")
async def health():
    return {"ok": True}


@app.post("/")
async def webhook(request: Request):
    data = await request.json()
    sess = data.get("session", {}) or {}
    req = data.get("request", {}) or {}
    sid = sess.get("session_id", "unknown")
    text = (req.get("original_utterance") or "").strip().lower()

    last_activity[sid] = time.time()
    prune_history()

    # Приветствие в начале сессии или при пустой реплике.
    if sess.get("new") or not text:
        history.setdefault(sid, [])
        return alice_response(GREETING)

    # Команда выхода из навыка.
    if text in EXIT_WORDS:
        history.pop(sid, None)
        pending.pop(sid, None)
        return alice_response("До встречи!", end_session=True)

    task = pending.get(sid)

    # Ответ модели уже готов — пользователь сказал «дальше».
    if task is not None and task.done():
        pending.pop(sid)
        if task.exception():
            return alice_response("Не получилось достучаться до сервера. Повтори вопрос.")
        answer = task.result()
        history.setdefault(sid, []).append({"role": "assistant", "content": answer})
        return alice_response(answer)

    # Модель всё ещё думает — пользователь сказал «дальше» слишком рано.
    if task is not None:
        return alice_response("Ещё секунду… Скажи «дальше».")

    # Обычный запрос: запускаем DeepSeek и ждём в пределах лимита Алисы (4,5 с).
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history.get(sid, [])[-12:]
    messages.append({"role": "user", "content": req.get("original_utterance", "").strip()})
    history.setdefault(sid, []).append(messages[-1])

    task = asyncio.create_task(call_deepseek(messages))
    pending[sid] = task
    try:
        answer = await asyncio.wait_for(asyncio.shield(task), timeout=CALL_TIMEOUT)
    except asyncio.TimeoutError:
        # Не успели — вызов продолжается в фоне, ответ заберём на «дальше».
        return alice_response("Секунду, додумываю… Скажи «дальше», и я озвучу ответ.")
    except Exception as exc:  # сеть, неверный ключ, ошибка API
        pending.pop(sid, None)
        return alice_response(f"Что-то пошло не так: {str(exc)[:120]}. Повтори вопрос.")

    pending.pop(sid, None)
    history.setdefault(sid, []).append({"role": "assistant", "content": answer})
    return alice_response(answer)
