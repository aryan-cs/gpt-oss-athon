# print("Hello, World!")

import json
import os
from typing import Generator, Iterable, Optional
import requests

from logger import logger

# MODEL = "qwen2.5:0.5b"
MODEL = "gpt-oss:20b"
TERMINAL_LOGGING = True
SYSTEM_PROMPT = "You are a helpful assistant."

def load_file_card(file_path):
    with open('file_cards/' + file_path, 'r') as file:
        return file.read()
    

def call_llm(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    method: str = "ollama",                   # Default to Ollama
    # model: str = "gpt-oss:20b",               # Default to gpt-oss:20b
    model: str = MODEL,
    temperature: float = 0.5,
    max_tokens: int = 512,
    stream: bool = True,
    base_url: str = "http://localhost:11434", # Default to Ollama
):
    
    if TERMINAL_LOGGING:
        logger.user_log(prompt)
    
    if method.lower() != "ollama":
        raise NotImplementedError(f"Unsupported method: {method}")

    base = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{base.rstrip('/')}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }

    if stream:
        def _generator() -> Generator[str, None, None]:
            with requests.post(url, json=payload, stream=True, timeout=(5, 300)) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text}")
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk, dict) and chunk.get("error"):
                        raise RuntimeError(f"Ollama error: {chunk['error']}")
                    if chunk.get("done") is True:
                        break
                    content = (chunk.get("message") or {}).get("content")
                    if content:
                        yield content
        return _generator()
    else:
        with requests.post(url, json=payload, timeout=(5, 300)) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            msg = data.get("message") or {}
            return msg.get("content", "")


def print_stream(generator: Iterable[str]):
    if TERMINAL_LOGGING:
        logger.llm_log(generator, stream=True)
    else:
        for delta in generator:
            print(delta, end="", flush=True)

if __name__ == "__main__":
    print()

    # logger.error_log("Test error...")
    logger.set_model(MODEL)

    SYSTEM_PROMPT = load_file_card("test.txt")

    logger.system_log(f"Loaded SYSTEM PROMPT: {SYSTEM_PROMPT}\n")

    print_stream(call_llm(
        prompt="Why is the sky blue?",
        system_prompt=SYSTEM_PROMPT
    ))
    
    print()