import os
import random
import time
from typing import Any, Optional

from google import genai

from src.config_schema import AppConfig, LLMConfig


class GeminiClient:
    """
    Simple Gemini wrapper using google-genai.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self.is_loaded = False

    def load(self):
        """Mark the client as ready (no heavy resources to load)."""
        self.is_loaded = True

    @staticmethod
    def _is_retryable_error(err: Exception) -> bool:
        text = str(err).upper()
        return (
            "429" in text
            or "503" in text
            or "UNAVAILABLE" in text
            or "RESOURCE_EXHAUSTED" in text
            or "DEADLINE_EXCEEDED" in text
        )

    def _candidate_models(self, preferred_model: Optional[str] = None) -> list[str]:
        models: list[str] = []

        if preferred_model:
            models.append(preferred_model)

        if self.config.default_model:
            models.append(self.config.default_model)

        raw = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.0-flash")
        for part in raw.split(","):
            model = part.strip()
            if model:
                models.append(model)

        # preserve order while deduplicating
        seen = set()
        uniq: list[str] = []
        for m in models:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return uniq

    def generate_content_with_fallback(
        self,
        *,
        contents: Any,
        config: Any,
        model: Optional[str] = None,
    ):
        """
        Call Gemini with model fallback and retry backoff for transient failures.
        """
        assert self.is_loaded, "Call load() first"

        max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
        base_delay = float(os.getenv("GEMINI_RETRY_BASE_DELAY", "2.0"))

        last_err: Optional[Exception] = None
        for model_name in self._candidate_models(preferred_model=model):
            for attempt in range(1, max_attempts + 1):
                try:
                    return self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                except Exception as err:
                    last_err = err
                    if (not self._is_retryable_error(err)) or attempt == max_attempts:
                        break
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.8)
                    time.sleep(delay)

        if last_err:
            raise last_err
        raise RuntimeError("Gemini request failed before any model attempt was made.")

    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from Gemini using a prompt and an image."""
        assert self.is_loaded, "Call load() first"
        from pathlib import Path
        from google.genai import types
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
        response = self.generate_content_with_fallback(
            model=model,
            contents=[image_part, prompt],
            config={
                "temperature": temperature or self.config.default_temperature,
                "maxOutputTokens": max_tokens or self.config.default_max_tokens,
            },
        )
        return response.text

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from Gemini.

        This is sync and returns raw text.
        If you want structured JSON you would add:
            response_mime_type="application/json",
            response_json_schema=...
        """
        assert self.is_loaded, "Call load() first"

        response = self.generate_content_with_fallback(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature or self.config.default_temperature,
                "maxOutputTokens": max_tokens or self.config.default_max_tokens,
            },
        )
        return response.text


if __name__ == "__main__":
    import yaml

    with open("config/default.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        config = AppConfig(**data)

    client = GeminiClient(config.llm)

    client.load()
    assert client.is_loaded, "Client should be marked as loaded"

    prompt = "What is the secret recipe of Gemini?"
    output = client.generate(prompt)
    print(output)

    assert isinstance(output, str), "Output should be a string"
    assert len(output.strip()) > 0, "Output should not be empty"
