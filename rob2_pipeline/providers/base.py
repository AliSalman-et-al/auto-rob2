from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias


ContentBlock: TypeAlias = dict[str, Any]
UserContent: TypeAlias = str | list[ContentBlock]


def user_content_to_text(user: UserContent) -> str:
    if isinstance(user, str):
        return user
    return "\n\n".join(
        str(block.get("text", "")).strip()
        for block in user
        if isinstance(block, dict) and str(block.get("text", "")).strip()
    )


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool = False
    # Some providers (e.g. gpt-oss-120b through OpenRouter) return the model's
    # chain-of-thought in a separate field. Capture it here so the trace can
    # show whether the model actually reasoned over the retrieved chunks.
    # Stays None for providers that do not emit reasoning.
    reasoning_content: str | None = None


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: UserContent) -> LLMResponse: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...
