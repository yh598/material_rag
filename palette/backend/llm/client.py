from __future__ import annotations

import os
import requests
from typing import Optional


class ChatLLM:
    """
    Minimal OpenAI-compatible Chat Completions client.
    Works with gateways that expose:
      POST {base_url}/chat/completions
    with Authorization: Bearer <api_key>
    """
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "ChatLLM":
        base_url = os.getenv("AI_GATEWAY_BASE_URL", "").strip()
        api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
        model = os.getenv("AI_GATEWAY_MODEL", "gpt-4.1-mini").strip()

        if not base_url:
            raise ValueError("Missing AI_GATEWAY_BASE_URL in environment/.env")
        if not api_key:
            raise ValueError("Missing AI_GATEWAY_API_KEY in environment/.env")
        return cls(base_url=base_url, api_key=api_key, model=model)

    def chat(self, system: str, user: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        if r.status_code >= 400:
            # Provide a helpful error without leaking secrets
            raise RuntimeError(f"LLM request failed: {r.status_code} {r.text[:500]}")
        data = r.json()

        # OpenAI style: choices[0].message.content
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return str(data)
