"""AI API管理模块

支持多种AI服务商和API中转
"""
import asyncio
from typing import Optional, AsyncGenerator
import httpx
from loguru import logger

from config import settings


class APIManager:
    """AI API管理器

    支持:
    - OpenAI API
    - Azure OpenAI
    - 第三方中转API
    - 本地模型API
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0
    ):
        """初始化

        Args:
            api_base: API基础URL
            api_key: API密钥
            model: 模型名称
            timeout: 请求超时时间
        """
        self.api_base = (api_base or settings.ai_api_base).rstrip("/")
        self.api_key = api_key or settings.ai_api_key
        self.model = model or settings.ai_model
        self.timeout = timeout

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self._get_headers()
            )
        return self._client

    def _get_headers(self) -> dict:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_text_content(self, payload) -> str:
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
        if isinstance(payload, dict):
            for key in ("text", "content", "output_text"):
                value = payload.get(key)
                if value is not None:
                    return self._extract_text_content(value)
        return ""

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """发送聊天请求

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            AI回复内容
        """
        client = await self._get_client()

        request_data = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        try:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                json=request_data
            )
            response.raise_for_status()

            data = response.json()
            message = data["choices"][0].get("message", {})
            content = self._extract_text_content(message.get("content"))
            if not content:
                content = self._extract_text_content(data["choices"][0].get("text"))
            if not content:
                raise RuntimeError("AI API returned an empty response payload")
            return content

        except httpx.HTTPStatusError as e:
            logger.error(f"API请求失败: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"AI API请求失败: {e.response.status_code} - {e.response.text}")
        except httpx.ConnectError as e:
            logger.error(f"无法连接到AI服务: {self.api_base}, 错误: {e}")
            raise RuntimeError(f"无法连接到AI服务: {self.api_base}")
        except httpx.TimeoutException as e:
            logger.error(f"AI API请求超时: {self.api_base}")
            raise RuntimeError(f"AI API请求超时")
        except Exception as e:
            logger.error(f"API请求错误: {type(e).__name__}: {e}")
            raise RuntimeError(f"AI API错误: {type(e).__name__}: {e}")

    async def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式聊天请求

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数

        Yields:
            AI回复内容片段
        """
        client = await self._get_client()

        request_data = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs
        }

        try:
            async with client.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                json=request_data
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        import json
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            logger.error(f"流式API请求失败: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"流式API请求错误: {e}")
            raise

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """文本补全请求

        Args:
            prompt: 提示文本
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            补全内容
        """
        # 转换为chat格式
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, model, temperature, max_tokens, **kwargs)

    async def close(self):
        """关闭客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
