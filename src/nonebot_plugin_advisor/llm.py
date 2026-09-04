"""OpenAI 兼容 LLM 客户端封装：文本对话、函数调用、视觉描述。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from nonebot import logger

from .config import Config

if TYPE_CHECKING:
    from openai import AsyncOpenAI, APIStatusError, RateLimitError, APIConnectionError

try:
    from openai import AsyncOpenAI, APIStatusError, RateLimitError, APIConnectionError
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment,misc]


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
    import base64

    if isinstance(path_or_bytes, str):
        with open(path_or_bytes, 'rb') as f:
            data = f.read()
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
            raise LLMNotConfigured('未配置 advisor_llm_api_key，LLM 客服未启用')
        if AsyncOpenAI is None:  # pragma: no cover
            raise LLMNotConfigured('未安装 openai 依赖')
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

    def _translate_error(self, e: Exception) -> str:
        if isinstance(e, RateLimitError):
            return '模型服务繁忙（限流），请稍后再试'
        if isinstance(e, (APIConnectionError,)) or isinstance(e, TimeoutError):
            return '连接模型服务失败，请检查网络或稍后再试'
        if isinstance(e, APIStatusError):
            code = getattr(e, 'status_code', '')
            msg = ''
            try:
                body = getattr(e, 'body', None) or {}
                msg = (body.get('error') or {}).get('message', '')
            except Exception:
                pass
            if code == 400 and 'image' in str(msg).lower():
                raise VisionUnsupported(str(msg)) from e
            return f'模型服务返回错误（{code}）：{msg or e}'
        if isinstance(e, VisionUnsupported):
            return str(e)
        return f'LLM 调用异常：{type(e).__name__}: {e}'

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
            'temperature': cfg.advisor_llm_temperature
            if temperature is None
            else temperature,
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
            f'LLM chat 请求: model={kwargs["model"]} '
            f'messages={len(messages)} tools={len(tools) if tools else 0} '
            f'temperature={kwargs["temperature"]}'
        )
        t0 = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.debug(f'LLM chat 请求失败: {type(e).__name__}: {e}')
            msg = self._translate_error(e)
            if isinstance(e, VisionUnsupported):
                raise VisionUnsupported(msg) from e
            raise LLMError(msg) from e
        choice = resp.choices[0] if resp.choices else None
        if not choice:
            raise LLMError('模型未返回任何内容')
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
            f'LLM 返回: {kwargs["model"]} 耗时 {elapsed:.1f}s'
            f'{tok} 工具调用={n_tcalls}'
        )
        logger.debug(f'LLM 回复: {(str(data.get("content") or ""))[:500]!r}')
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
            f'视觉请求: model={model or self.vision_model} '
            f'prompt={prompt[:100]!r} 图片data_url={len(image_data_url_)} 字符'
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
            logger.debug(f'视觉请求失败: {type(e).__name__}: {e}')
            msg = self._translate_error(e)
            if isinstance(e, VisionUnsupported):
                raise VisionUnsupported(msg) from e
            raise LLMError(msg) from e
        choice = resp.choices[0] if resp.choices else None
        if not choice or not choice.message.content:
            raise LLMError('视觉模型未返回描述')
        desc = str(choice.message.content).strip()
        elapsed = time.perf_counter() - t0
        logger.debug(
            f'视觉返回: 耗时 {elapsed:.1f}s 描述 {len(desc)} 字: '
            f'{desc[:120]!r}'
        )
        return desc

    async def describe_image_file(
        self, path: str, ext: str | None = None, prompt: str | None = None
    ) -> str:
        """便捷方法：由文件路径调用视觉模型。"""
        import os

        ext = ext or os.path.splitext(path)[1].lstrip('.')
        logger.debug(f'视觉描述图片文件: {path} (ext={ext})')
        data_url = image_data_url(path, ext)
        if prompt:
            return await self.describe_image(data_url, prompt=prompt)
        return await self.describe_image(data_url)
