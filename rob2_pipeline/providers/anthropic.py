import time

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ._rate_limiter import SlidingWindowRateLimiter
from .base import LLMProvider, LLMResponse, UserContent, user_content_to_text


def _is_rate_limit_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    message = str(exc).lower()
    return "rate" in message and ("limit" in message or "429" in message)


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        api_key,
        model,
        temperature=0,
        max_tokens=2000,
        rpm_limit: int = 40,
        tpm_limit: int = 30_000,
        request_timeout: float = 60,
        max_retries: int = 2,
    ):
        from langchain_anthropic import ChatAnthropic

        self._model = model
        self.client = ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
            max_retries=max(0, max_retries - 1),
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(1, max_retries)
        self._rate_limiter = SlidingWindowRateLimiter(
            rpm_limit=rpm_limit, tpm_limit=tpm_limit
        )

    @property
    def model_id(self):
        return self._model

    def complete(self, system: str, user: UserContent) -> LLMResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        estimated_tokens = SlidingWindowRateLimiter.estimate_input_tokens(
            system, user_content_to_text(user)
        )
        self._rate_limiter.wait_for_slot(estimated_tokens=estimated_tokens)
        start = time.time()
        retryer = Retrying(
            wait=wait_exponential(multiplier=2, min=5, max=30),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception(_is_rate_limit_error),
            reraise=True,
        )
        r = retryer(
            lambda: self.client.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        )
        usage = (r.response_metadata or {}).get("usage", {})
        return LLMResponse(
            content=r.content,
            model=self._model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=(time.time() - start) * 1000,
            reasoning_content=r.additional_kwargs.get("reasoning_content"),
        )
