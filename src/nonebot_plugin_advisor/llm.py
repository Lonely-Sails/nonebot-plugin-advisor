"""OpenAI 兼容 LLM 客户端封装：文本对话、函数调用、视觉描述。"""

from __future__ import annotations

import time
import base64
from typing import Any
from pathlib import Path

from openai import (
    AsyncOpenAI,
    APIStatusError,
    RateLimitError,
    APIConnectionError,
)
from nonebot import logger

from .config import Config

_DESCRIBE_PROMPT = (
    '请用中文简洁地描述这张图片的内容，尽量说清图中元素、界面/步骤，最多 150 字。'
)


# 允许的图片 mime（用于 data url）
_MIME = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'webp': 'image/webp',
    'bmp': 'image/bmp',
}


class LLMNotConfigured(RuntimeError):
    """LLM 未配置"""


class VisionUnsupported(RuntimeError):
    """模型不支持图片输入"""


class LLMError(RuntimeError):
    """LLM 调用出错（带用户可读消息）"""


def image_data_url(path_or_bytes: bytes | str, ext: str = 'png') -> str:
    """把图片字节/文件路径转成 data url，供多模态消息使用。"""
    if isinstance(path_or_bytes, str):
        data = Path(path_or_bytes).read_bytes()
    else:
        data = path_or_bytes
    mime = _MIME.get(ext.lower().lstrip('.'), 'image/png')
    return f'data:{mime};base64,{base64.b64encode(data).decode()}'


class LLMClient:
    """OpenAI 兼容客户端。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client: Any = None

    # ── 生命周期 ────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return bool(self.cfg.advisor_llm_api_key)

    @property
    def client(self) -> Any:
        if not self.available:
            raise LLMNotConfigured(
                'advisor_llm_api_key is not configured; the LLM assistant is disabled'
            )
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.cfg.advisor_llm_base_url,
                api_key=self.cfg.advisor_llm_api_key,
                timeout=self.cfg.advisor_llm_timeout,
                max_retries=self.cfg.advisor_request_max_retries,
            )
        return self._client

    @property
    def text_model(self) -> str:
        return self.cfg.advisor_llm_model

    @property
    def vision_model(self) -> str:
        """视觉模型：优先使用独立配置，否则退回主模型。"""
        return self.cfg.advisor_vision_model or self.cfg.advisor_llm_model

    def _translate_error(self, error: Exception) -> str:
        if isinstance(error, RateLimitError):
            return 'The model service is busy (rate limited); please try again later'
        if isinstance(error, (APIConnectionError, TimeoutError)):
            return (
                'Failed to connect to the model service; '
                'check your network or try again later'
            )
        if isinstance(error, APIStatusError):
            code = getattr(error, 'status_code', '')
            message = ''
            try:
                body = getattr(error, 'body', None) or {}
                message = (body.get('error') or {}).get('message', '')
            except Exception:
                pass
            if code == 400 and 'image' in str(message).lower():
                raise VisionUnsupported(str(message)) from error
            return f'The model service returned an error ({code}): {message or error}'
        if isinstance(error, VisionUnsupported):
            return str(error)
        return f'LLM call failed: {type(error).__name__}: {error}'

    # ── 对话 ────────────────────────────────────────────────────────────
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """调用 chat.completions，返回消息体 dict。

        Returns:
            形如 {"role": "assistant", "content": str|None, "tool_calls": [...]|None}
            的工具调用由上层解析。
        """
        cfg = self.cfg
        kwargs: dict[str, Any] = {
            'model': model or self.text_model,
            'messages': messages,
            'temperature': (
                cfg.advisor_llm_temperature if temperature is None else temperature
            ),
        }
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        elif cfg.advisor_llm_max_tokens:
            kwargs['max_tokens'] = cfg.advisor_llm_max_tokens
        if tools:
            kwargs['tools'] = tools
        if tool_choice is not None:
            kwargs['tool_choice'] = tool_choice
        logger.debug(
            f'LLM chat request: model={kwargs["model"]} '
            f'messages={len(messages)} tools={len(tools) if tools else 0} '
            f'temperature={kwargs["temperature"]}'
        )
        t0 = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.debug(f'LLM chat request failed: {type(e).__name__}: {e}')
            msg = self._translate_error(e)
            if isinstance(e, VisionUnsupported):
                raise VisionUnsupported(msg) from e
            raise LLMError(msg) from e
        choice = resp.choices[0] if resp.choices else None
        if not choice:
            raise LLMError('The model returned no content')
        message = choice.message
        data: dict[str, Any] = {'role': 'assistant', 'content': message.content}
        if getattr(message, 'tool_calls', None):
            data['tool_calls'] = [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.function.name,
                        'arguments': tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        elapsed = time.perf_counter() - t0
        n_tcalls = len(data.get('tool_calls') or [])
        usage = getattr(resp, 'usage', None)
        tok = ''
        try:
            pt = getattr(usage, 'prompt_tokens', None)
            ct = getattr(usage, 'completion_tokens', None)
            if pt is not None or ct is not None:
                tok = f' prompt={pt} completion={ct}'
        except Exception:
            pass
        logger.info(
            f'LLM response: {kwargs["model"]} elapsed {elapsed:.1f}s'
            f'{tok} tool_calls={n_tcalls}'
        )
        logger.debug(f'LLM reply: {(str(data.get("content") or ""))[:500]!r}')
        return data

    # ── 视觉描述 ────────────────────────────────────────────────────────
    async def describe_image(
        self,
        image_data_url_: str,
        prompt: str = _DESCRIBE_PROMPT,
        model: str | None = None,
    ) -> str:
        """调用视觉模型描述图片（输入为 data url）。失败抛 LLMError。"""
        if not image_data_url_.startswith('data:'):
            raise ValueError('describe_image 需要 data url 输入')
        messages: list[dict[str, Any]] = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': image_data_url_}},
                ],
            }
        ]
        logger.debug(
            f'Vision request: model={model or self.vision_model} '
            f'prompt={prompt[:100]!r} image_data_url_len={len(image_data_url_)}'
        )
        t0 = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(
                model=model or self.vision_model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=256,
                temperature=0.2,
            )
        except Exception as e:
            logger.debug(f'Vision request failed: {type(e).__name__}: {e}')
            msg = self._translate_error(e)
            if isinstance(e, VisionUnsupported):
                raise VisionUnsupported(msg) from e
            raise LLMError(msg) from e
        choice = resp.choices[0] if resp.choices else None
        if not choice or not choice.message.content:
            raise LLMError('The vision model returned no description')
        desc = str(choice.message.content).strip()
        elapsed = time.perf_counter() - t0
        logger.debug(
            f'Vision response: elapsed {elapsed:.1f}s description_len={len(desc)}: '
            f'{desc[:120]!r}'
        )
        return desc

    async def describe_image_file(
        self, path: str, ext: str | None = None, prompt: str | None = None
    ) -> str:
        """便捷方法：由文件路径调用视觉模型。"""
        ext = ext or Path(path).suffix.lstrip('.')
        logger.debug(f'Describing image file: {path} (ext={ext})')
        data_url = image_data_url(path, ext)
        if prompt:
            return await self.describe_image(data_url, prompt=prompt)
        return await self.describe_image(data_url)
