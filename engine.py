# print("Hello, World!")

import json
import os
from typing import Generator, Iterable, Optional, List, Dict, Union
import requests

from logger import logger

MODEL = "qwen2.5:0.5b"
# MODEL = "gpt-oss:20b"
TERMINAL_LOGGING = True
SYSTEM_PROMPT = "Respond in a single sentence."

def load_file_card(file_path):
    with open('file_cards/' + file_path, 'r') as file:
        return file.read()
    

Message = Dict[str, str]


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
    history: Optional[List[Message]] = None,
):
    
    if method.lower() != "ollama":
        raise NotImplementedError(f"Unsupported method: {method}")

    base = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{base.rstrip('/')}/api/chat"

    messages: List[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        for m in history:
            role = m.get("role")
            content = m.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            if role not in {"user", "assistant", "system"}:
                continue
            messages.append({"role": role, "content": content})
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


class ChatSession:
    def __init__(
        self,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        model: str = MODEL,
        base_url: str = "http://localhost:11434",
        temperature: float = 0,
        max_tokens: int = 512,
        terminal_logging: bool = True,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.terminal_logging = terminal_logging
        self.history: List[Message] = []

    def ask_stream(self, prompt: str) -> Generator[str, None, None]:
        def _gen() -> Generator[str, None, None]:
            chunks: List[str] = []
            gen = call_llm(
                prompt,
                system_prompt=self.system_prompt,
                method="ollama",
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                base_url=self.base_url,
                history=self.history,
            )
            for delta in gen:
                chunks.append(delta)
                yield delta
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": "".join(chunks)})
        return _gen()

    def ask(self, prompt: str) -> str:
        reply = call_llm(
            prompt,
            system_prompt=self.system_prompt,
            method="ollama",
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            base_url=self.base_url,
            history=self.history,
        )
        assert isinstance(reply, str)
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": reply})
        return reply


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
    TEST_QUERY = "What is the capital of France?"

    logger.system_log(f"Loaded SYSTEM PROMPT: {SYSTEM_PROMPT}\n")
    logger.user_log(TEST_QUERY)

    print_stream(call_llm(
        prompt=TEST_QUERY,
        system_prompt=SYSTEM_PROMPT
    ))
    
    print()