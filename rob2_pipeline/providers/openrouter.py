import json
import http.client
import socket
import time
import urllib.error
import urllib.request

from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ._rate_limiter import SlidingWindowRateLimiter
from .base import LLMProvider, LLMResponse, UserContent, user_content_to_text

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def _is_retryable_openrouter_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            code = None
        if code == 429 or (code is not None and 500 <= code < 600):
            return True
        if code is not None:
            return False

    message = str(exc).lower()
    retry_terms = (
        "rate limit",
        "429",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection",
        "server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
    )
    non_retry_terms = (
        "invalid api key",
        "unauthorized",
        "forbidden",
        "context length",
        "maximum context",
        "invalid request",
        "bad request",
    )
    if any(term in message for term in non_retry_terms):
        return False
    return any(term in message for term in retry_terms)


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        api_key,
        model,
        temperature=0,
        max_tokens=2000,
        rpm_limit=18,
        rpd_limit=190,
        request_timeout: float = 60,
        max_retries: int = 2,
    ):
        self._model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_retries = max(1, max_retries)
        self._rate_limiter = SlidingWindowRateLimiter(
            rpm_limit=rpm_limit, rpd_limit=rpd_limit
        )

    @property
    def model_id(self):
        return self._model

    def complete(self, system: str, user: UserContent) -> LLMResponse:
        self._rate_limiter.wait_for_slot()
        user_text = user_content_to_text(user)
        start = time.perf_counter()
        retryer = Retrying(
            wait=wait_exponential(multiplier=2, min=2, max=10),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception(_is_retryable_openrouter_error),
            reraise=True,
        )
        payload = retryer(lambda: self._post_chat_completion(system, user_text))
        latency_ms = (time.perf_counter() - start) * 1000
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = payload.get("usage") or {}
        return LLMResponse(
            content=message.get("content") or "",
            model=self._model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            reasoning_content=message.get("reasoning")
            or message.get("reasoning_content"),
        )

    def _post_chat_completion(self, system: str, user: str) -> dict:
        body = {
            "model": self._model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/AliSalman-et-al/auto-rob2",
                "X-Title": "auto-rob2",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout,
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body_text}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        except (
            ConnectionError,
            ConnectionResetError,
            TimeoutError,
            socket.timeout,
            http.client.HTTPException,
        ) as exc:
            raise RuntimeError(f"OpenRouter connection failed: {exc}") from exc
        payload = json.loads(raw)
        if "error" in payload:
            raise RuntimeError(f"OpenRouter error: {payload['error']}")
        return payload
