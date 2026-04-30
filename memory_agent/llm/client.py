from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMClient:
    """
    Minimal OpenAI-compatible chat client.

    Expected endpoint:
      {base_url}/chat/completions

    This client intentionally has no dependency on openai/requests, so tests can
    run in lightweight environments.
    """

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    timeout: int = 60

    def __post_init__(self) -> None:
        self.model = self.model or os.getenv("MEMORY_LLM_MODEL", "")
        self.base_url = self.base_url or os.getenv("MEMORY_LLM_BASE_URL", "")
        self.api_key = self.api_key or os.getenv("MEMORY_LLM_API_KEY", "")

        self.base_url = self.base_url.rstrip("/")

    def available(self) -> bool:
        return bool(self.model and self.base_url)

    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 1200,
        temperature: float | None = None,
    ) -> str:
        if not self.available():
            logger.warning(
                "LLMClient not available — model=%r base_url=%r",
                self.model, self.base_url,
            )
            return "{}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON generator. "
                        "Return JSON only. Do not use markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }

        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("LLMClient HTTP error: %s", exc)
            return "{}"

        try:
            data = json.loads(raw)
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "{}")
            )
            if not content or content == "{}":
                logger.warning("LLMClient returned empty content for model=%s", self.model)
            return content
        except Exception as exc:
            logger.warning("LLMClient parse error: %s", exc)
            return "{}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers