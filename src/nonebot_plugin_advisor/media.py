"""从 UniMessage（跨平台通用消息）中抽取文本与附件（图片/文件），并处理“引用消息”。"""

from __future__ import annotations

import asyncio
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass

from nonebot import logger
from nonebot.adapters import Bot, Event

from .utils import safe_filename, http_get_bytes, looks_like_url

try:
    from nonebot_plugin_alconna.uniseg import (
        At,
        File,
        Text,
        AtAll,
        Image,
        Other,
        Reply,
        Reference,
        UniMessage,
    )
    from nonebot_plugin_alconna.uniseg.tools import image_fetch
except ImportError:  # pragma: no cover
    UniMessage = None  # type: ignore[assignment]
    Text = Image = File = Reply = At = AtAll = Reference = Other = None  # type: ignore
    image_fetch = None  # type: ignore

_MEDIA_TIMEOUT = 30.0


@dataclass
class MediaItem:
    """抽取出的附件"""

    name: str
    data: bytes
    media_type: str = ''
    origin: str = 'message'  # message / quote


@dataclass
class ParsedMessage:
    """解析结果"""

    text: str = ''
    quoted_text: str = ''
    media: list[MediaItem] = field(default_factory=list)
    note: str = ''
    """无法处理的引用等信息"""

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip() or self.media)


async def _fetch_image_bytes(seg: Any, bot: Bot, event: Event) -> bytes | None:
    """跨平台下载图片字节（优先 url，其次 alconna image_fetch）。"""
    # path 直接可读
    if getattr(seg, 'path', None):
        p = Path(seg.path)
        if await asyncio.to_thread(p.exists):
            return await asyncio.to_thread(p.read_bytes)
    # url 下载
    url = getattr(seg, 'url', None)
    if url and looks_like_url(url):
        try:
            return await asyncio.wait_for(http_get_bytes(url), _MEDIA_TIMEOUT)
        except Exception as e:
            logger.debug(f'[advisor] 图片 url 下载失败 {url}: {e}')
    # 交给 alconna 的 image_fetch（各适配器有各自实现）
    if image_fetch is not None:
        try:
            data = await asyncio.wait_for(
                image_fetch(event=event, bot=bot, state={}, img=seg), _MEDIA_TIMEOUT
            )
            if data:
                return data
        except Exception as e:
            logger.debug(f'[advisor] image_fetch 失败: {e}')
    return None


async def _fetch_file_bytes(seg: Any) -> bytes | None:
    if getattr(seg, 'path', None):
        p = Path(seg.path)
        if await asyncio.to_thread(p.exists):
            return await asyncio.to_thread(p.read_bytes)
    url = getattr(seg, 'url', None)
    if url and looks_like_url(url):
        try:
            return await asyncio.wait_for(http_get_bytes(url), _MEDIA_TIMEOUT)
        except Exception as e:
            logger.debug(f'[advisor] 文件下载失败 {url}: {e}')
    raw = getattr(seg, 'raw', None)
    if raw is not None:
        try:
            return raw.getvalue() if hasattr(raw, 'getvalue') else bytes(raw)
        except Exception:
            return None
    return None


def _default_name(seg: Any, fallback: str) -> str:
    name = getattr(seg, 'name', None) or ''
    if name and name != 'file':
        return safe_filename(name)
    return fallback


async def _walk_segments(
    segments: Any,
    bot: Bot,
    event: Event,
    *,
    origin: str,
    text_parts: list[str],
    media: list[MediaItem],
    notes: list[str],
) -> None:
    if UniMessage is None:
        return
    if isinstance(segments, UniMessage):
        seg_list: list[Any] = list(segments)
    else:
        seg_list = list(segments)
    for seg in seg_list:
        if Text is not None and isinstance(seg, Text):
            if seg.text.strip():
                text_parts.append(seg.text.strip())
        elif Image is not None and isinstance(seg, Image):
            data = await _fetch_image_bytes(seg, bot, event)
            if data:
                ext = Path(_default_name(seg, 'img.png')).suffix.lstrip('.') or 'png'
                media.append(
                    MediaItem(
                        name=_default_name(seg, f'image.{ext}'),
                        data=data,
                        media_type=seg.mimetype or f'image/{ext}',
                        origin=origin,
                    )
                )
            else:
                notes.append('（有一条图片无法读取）')
        elif File is not None and isinstance(seg, File):
            data = await _fetch_file_bytes(seg)
            if data:
                media.append(
                    MediaItem(
                        name=_default_name(seg, 'file.bin'),
                        data=data,
                        media_type=seg.mimetype or '',
                        origin=origin,
                    )
                )
            else:
                notes.append(f'（有文件 {seg.name or ""} 无法读取）')
        elif Reply is not None and isinstance(seg, Reply):
            # 引用消息：其内容可能已被 alconna 解析进 msg
            if seg.msg is not None:
                try:
                    if isinstance(seg.msg, str):
                        if seg.msg.strip():
                            text_parts.append(seg.msg.strip())
                    elif UniMessage is not None:
                        inner = UniMessage.of(seg.msg, bot=bot)
                        await _walk_segments(
                            inner,
                            bot,
                            event,
                            origin=origin,
                            text_parts=text_parts,
                            media=media,
                            notes=notes,
                        )
                except Exception as e:
                    logger.debug(f'[advisor] 解析引用消息失败: {e}')
            else:
                notes.append('（用户引用了一条消息，但内容无法获取）')
        elif At is not None and isinstance(seg, At):
            if getattr(seg, 'display', None):
                text_parts.append(f'@{seg.display}')
        elif AtAll is not None and isinstance(seg, AtAll):
            text_parts.append('@全体')
        elif Reference is not None and isinstance(seg, Reference):
            # 转发/引用多条消息：尽力取 children 文本
            children = (
                getattr(seg, 'children', None) or getattr(seg, 'nodes', None) or []
            )
            if children:
                try:
                    if UniMessage is not None:
                        inner = UniMessage(children)
                        await _walk_segments(
                            inner,
                            bot,
                            event,
                            origin=origin,
                            text_parts=text_parts,
                            media=media,
                            notes=notes,
                        )
                except Exception:
                    pass
        elif Other is not None and isinstance(seg, Other):
            # 未知段：尝试转文本
            origin_seg = getattr(seg, 'origin', None)
            try:
                if (
                    origin_seg is not None
                    and getattr(origin_seg, 'is_text', lambda: False)()
                ):
                    txt = getattr(origin_seg, 'data', {}).get('text', '')
                    if txt.strip():
                        text_parts.append(txt.strip())
            except Exception:
                pass


async def parse_unimsg(
    unimsg: Any,
    bot: Bot,
    event: Event,
) -> ParsedMessage:
    """解析一条已转换好的 UniMessage（含引用）为文本 + 附件。"""
    text_parts: list[str] = []
    media: list[MediaItem] = []
    notes: list[str] = []
    await _walk_segments(
        unimsg,
        bot,
        event,
        origin='message',
        text_parts=text_parts,
        media=media,
        notes=notes,
    )
    return ParsedMessage(
        text=' '.join(text_parts),
        media=media,
        note='；'.join(notes),
    )


async def parse_event_message(event: Event, bot: Bot) -> ParsedMessage:
    """便捷入口：从事件构建 UniMessage（自动附加引用）再解析。"""
    if UniMessage is None:  # pragma: no cover
        return ParsedMessage(text=event.get_plaintext())
    try:
        raw_msg = event.get_message()
        unimsg = UniMessage.of(raw_msg, bot=bot)
        unimsg = await unimsg.attach_reply(event=event, bot=bot)
    except Exception as e:
        logger.warning(f'[advisor] 构建通用消息失败: {e}')
        return ParsedMessage(text=event.get_plaintext())
    return await parse_unimsg(unimsg, bot, event)
