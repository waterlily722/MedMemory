from __future__ import annotations

import json
import os
from urllib import request


class LLMClient:
    def __init__(self, model: str = "", base_url: str = "", api_key: str = ""):
        self.model = model or os.environ.get("MEMORY_LLM_MODEL", "")
        self.base_url = (base_url or os.environ.get("MEMORY_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("MEMORY_LLM_API_KEY", "")

    def available(self) -> bool:
        return bool(self.model and self.base_url)

    def generate_json(self, prompt: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
        if not self.available():
            return "{}"

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": "Return only a JSON object."},
                {"role": "user", "content": prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = request.Request(url, data=body, headers=headers, method="POST")
            with request.urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return str((((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or "{}"))
        except Exception:
            return "{}"
