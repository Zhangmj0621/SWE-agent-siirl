import asyncio
import os

import aiohttp
from dotenv import load_dotenv

from ...base import DummyTokenizer, Model, ModelResponse


class OpenaiModel(Model):
    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        self.api_url = api_url if api_url is not None else os.getenv("API_URL", "http://localhost:8000")
        self.api_key = api_key if api_key is not None else os.getenv("API_KEY", None)
        self.model_name = model_name if model_name is not None else os.getenv("API_MODEL_NAME", "GLM-4.6")
        self._tokenizer = DummyTokenizer()

    @property
    def tokenizer(self):
        return self._tokenizer

    async def query(
        self,
        input_tokens: list[int],
        messages: list[dict],
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> ModelResponse:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or 65536,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                result: dict = await resp.json()

        output_content: str = result["choices"][0]["message"]["content"]
        output_tokens = self.tokenizer.encode(output_content)

        return ModelResponse(
            output=output_content,
            output_tokens=output_tokens,
            log_probs=[1.0] * len(output_tokens),
            experts=None,
            raw=result,
        )


def main():
    load_dotenv()
    model = OpenaiModel()

    # Prepare a test input
    # For demonstration, we use a simple prompt and empty messages
    test_text = "Hello, world!"
    input_tokens = model.tokenizer.encode(test_text)
    messages = [{"role": "user", "content": test_text}]

    async def test_query():
        response: ModelResponse = await model.query(
            input_tokens=input_tokens,
            messages=messages,
        )
        print("Output:", response.output)

    asyncio.run(test_query())


if __name__ == "__main__":
    main()
