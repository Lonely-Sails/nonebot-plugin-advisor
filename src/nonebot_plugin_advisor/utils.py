"""通用小工具：下载、HTML 转文本、路径安全等。"""

from __future__ import annotations

import re
import html
import json
import asyncio
from typing import Any
from pathlib import Path

import httpx

_USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/124.0 Safari/537.36 NonebotAdvisor/1.0'
)
_TAG_RE = re.compile(r'<[^>]+>')
_SCRIPT_RE = re.compile(r'<(script|style)[^>]*>.*?</\1>', re.I | re.S)
_WS_RE = re.compile(r'[ \t\xa0\u3000]+')
_BLANK_RE = re.compile(r'\n{3,}')


def strip_html(raw: str) -> str:
    """把 HTML 粗略转成纯文本（够用于阅读网页正文）。"""
    raw = _SCRIPT_RE.sub('\n', raw or '')
    raw = _TAG_RE.sub('\n', raw)
    raw = html.unescape(raw)
    raw = _WS_RE.sub(' ', raw)
    lines = [ln.strip() for ln in raw.splitlines()]
    text = '\n'.join(ln for ln in lines if ln)
    text = _BLANK_RE.sub('\n\n', text)
    return text.strip()


async def http_get_bytes(url: str, timeout_: float = 20.0) -> bytes:
    """下载二进制，带 UA 与 asyncio 超时保护。"""

    async def _fetch() -> bytes:
        async with httpx.AsyncClient(
            timeout=timeout_, follow_redirects=True, headers={'User-Agent': _USER_AGENT}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    return await asyncio.wait_for(_fetch(), timeout=timeout_)


def safe_filename(name: str) -> str:
    """清理文件名，去掉路径分隔与危险字符。"""
    name = name.replace('\\', '/').split('/')[-1]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip().strip('.')
    return name[:120] or 'file'


def ensure_ext(name: str, fallback: str = '.bin') -> str:
    """确保文件名带扩展名。"""
    return name if '.' in name else name + fallback


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + '\n…（内容过长已截断）'


def split_text(text: str, max_length: int) -> list[str]:
    """把长文本按自然边界拆成不超过 max_length 的多段。

    优先在换行、句号、逗号等自然边界处切分，避免把一句话拦腰截断。
    若某段仍超过 max_length，则硬切。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    # 自然边界（按优先级）：空行 > 换行 > 句号/问号/感叹号 > 逗号/分号
    boundaries = ['\n\n', '\n', '。', '？', '！', '；', '，', '、', ' ']
    segments: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_length:
            segments.append(remaining.strip())
            break
        # 在 max_length 范围内找最靠后的自然边界
        cut = -1
        for b in boundaries:
            idx = remaining.rfind(b, 0, max_length)
            if idx > 0:
                cut = idx + len(b)
                break
        if cut <= 0:
            # 找不到自然边界，硬切
            cut = max_length
        segments.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [s for s in segments if s]


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def looks_like_url(s: str) -> bool:
    return bool(re.match(r'^https?://', s.strip(), re.I))


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def json_load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def guess_text_encoding(data: bytes) -> str | None:
    """尝试把字节解析成文本，成功返回文本，失败返回 None。"""
    for enc in ('utf-8', 'utf-8-sig', 'gb18030'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None
