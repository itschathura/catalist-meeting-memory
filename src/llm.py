# LLM Client and Token/Cost Logger for Catalist Meeting Memory

# Features:
# 1. Fast, reliable REST client for Google Gemini (avoids gRPC socket issues)
# 2. Structured JSON generation via Gemini responseSchema
# 3. Persistent disk caching in storage/llm_cache/ to save tokens & money
# 4. Automatic retry with exponential backoff on transient errors (429, 503)
# 5. Comprehensive token usage & estimated API cost tracking saved to storage/llm_metrics.json


from __future__ import annotations

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    CACHE_DIR,
    METRICS_FILE,
    ensure_dirs,
)

# Cost estimates per 1,000,000 tokens (Standard Tier for Flash models)
INPUT_COST_PER_M = 0.075    # $0.075 / 1M prompt tokens
OUTPUT_COST_PER_M = 0.300   # $0.300 / 1M output tokens

FALLBACK_MODELS = [
    GEMINI_MODEL,
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]


class LLMClient:
    """Production-grade REST client for Google Gemini with caching and metrics."""

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        self.api_key = (api_key or GEMINI_API_KEY).strip()
        self.default_model = default_model or GEMINI_MODEL
        ensure_dirs()
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict[str, Any]:
        """Load cumulative token metrics from storage."""
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    def _save_metrics(self) -> None:
        """Persist metrics to disk."""
        ensure_dirs()
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

    def _compute_cache_key(self, prompt: str, schema: Optional[Dict[str, Any]], system: Optional[str]) -> str:
        """Compute deterministic SHA-256 cache key."""
        payload_repr = json.dumps({
            "prompt": prompt,
            "schema": schema,
            "system": system,
        }, sort_keys=True)
        return hashlib.sha256(payload_repr.encode("utf-8")).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if present."""
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save response to local disk cache."""
        ensure_dirs()
        cache_file = CACHE_DIR / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _record_usage(self, prompt_tokens: int, completion_tokens: int, from_cache: bool = False) -> None:
        """Update metrics for token usage and estimated cost."""
        self.metrics["total_calls"] += 1
        if from_cache:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1
            self.metrics["prompt_tokens"] += prompt_tokens
            self.metrics["completion_tokens"] += completion_tokens
            self.metrics["total_tokens"] += (prompt_tokens + completion_tokens)
            
            cost = (prompt_tokens / 1_000_000 * INPUT_COST_PER_M) + \
                   (completion_tokens / 1_000_000 * OUTPUT_COST_PER_M)
            self.metrics["estimated_cost_usd"] = round(self.metrics["estimated_cost_usd"] + cost, 6)

        self._save_metrics()

    def generate(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini with structured output, caching, and retry logic.

        Returns:
            Dict containing:
                "text": str
                "parsed": dict or None (if structured output)
                "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
                "cached": bool
                "model": str
        """
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add your key to .env file in the project root."
            )

        cache_key = self._compute_cache_key(prompt, response_schema, system_instruction)
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                self._record_usage(
                    prompt_tokens=cached_data.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=cached_data.get("usage", {}).get("completion_tokens", 0),
                    from_cache=True,
                )
                cached_data["cached"] = True
                return cached_data

        # Construct REST payload
        contents: List[Dict[str, Any]] = []
        if system_instruction:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System Instruction: {system_instruction}\n\nTask:\n{prompt}"}]
            })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}]
            })

        gen_config: Dict[str, Any] = {
            "temperature": temperature,
        }

        if response_schema:
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseSchema"] = response_schema

        request_body = {
            "contents": contents,
            "generationConfig": gen_config,
        }

        models_to_try = [self.default_model]
        for m in FALLBACK_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error = ""

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json=request_body,
                        timeout=45,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        candidate = data.get("candidates", [{}])[0]
                        text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
                        usage_meta = data.get("usageMetadata", {})

                        prompt_tokens = usage_meta.get("promptTokenCount", 0)
                        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
                        total_tokens = usage_meta.get("totalTokenCount", 0)

                        parsed = None
                        if response_schema and text:
                            try:
                                parsed = json.loads(text)
                            except json.JSONDecodeError:
                                # Clean markdown formatting if present
                                clean_text = text.strip()
                                if clean_text.startswith("```json"):
                                    clean_text = clean_text[7:]
                                if clean_text.endswith("```"):
                                    clean_text = clean_text[:-3]
                                parsed = json.loads(clean_text.strip())

                        result = {
                            "text": text,
                            "parsed": parsed,
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens,
                            },
                            "cached": False,
                            "model": model_name,
                        }

                        # Save to cache
                        if use_cache:
                            self._save_to_cache(cache_key, result)

                        self._record_usage(prompt_tokens, completion_tokens, from_cache=False)
                        return result

                    elif response.status_code in [429, 503]:
                        # Transient rate limit or model busy -> backoff and retry
                        backoff = (2 ** attempt) + 1.0
                        time.sleep(backoff)
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        continue
                    elif response.status_code == 404:
                        # Model not available for this key -> break to try next model in fallback list
                        last_error = f"Model {model_name} returned 404 Not Found"
                        break
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        break

                except requests.RequestException as e:
                    last_error = f"RequestException: {e}"
                    time.sleep(1.5)

        raise RuntimeError(f"All LLM generation attempts failed. Last error: {last_error}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return the current metrics summary."""
        return dict(self.metrics)


# Global singleton client instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold blue]Testing LLM Client (Gemini REST + Caching)...[/bold blue]")
    client = get_llm_client()

    schema = {
        "type": "OBJECT",
        "properties": {
            "summary": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["summary", "confidence"],
    }

    prompt = "Summarize the benefit of using SQLite FTS5 for meeting transcripts in 1 concise sentence."
    
    rprint("[dim]1. Making first call (Live API)...[/dim]")
    res1 = client.generate(prompt, response_schema=schema, use_cache=True)
    rprint("[bold green]Success (Live API):[/bold green]")
    rprint(f"  Parsed:  {res1['parsed']}")
    rprint(f"  Cached:  {res1['cached']}")
    rprint(f"  Model:   {res1['model']}")
    rprint(f"  Tokens:  {res1['usage']}")

    rprint("\n[dim]2. Making second call with same prompt (Should hit cache)...[/dim]")
    res2 = client.generate(prompt, response_schema=schema, use_cache=True)
    rprint("[bold green]Success (Cached):[/bold green]")
    rprint(f"  Cached:  [bold cyan]{res2['cached']}[/bold cyan]")
    
    rprint("\n[bold yellow]Cumulative Metrics:[/bold yellow]")
    rprint(client.get_metrics_summary())
