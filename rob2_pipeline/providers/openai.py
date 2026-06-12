import time

from .base import LLMProvider, LLMResponse, UserContent, user_content_to_text


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key,
        model,
        temperature=0,
        max_tokens=2000,
        request_timeout: float = 60,
        max_retries: int = 2,
    ):
        from langchain_openai import ChatOpenAI

        self._model = model
        self.client = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
            max_retries=max(0, max_retries - 1),
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_id(self):
        return self._model

    def complete(self, system: str, user: UserContent) -> LLMResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        user_text = user_content_to_text(user)
        start = time.time()
        r = self.client.invoke(
            [SystemMessage(content=system), HumanMessage(content=user_text)]
        )
        usage = r.usage_metadata or {}
        return LLMResponse(
            r.content,
            self._model,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            (time.time() - start) * 1000,
        )
