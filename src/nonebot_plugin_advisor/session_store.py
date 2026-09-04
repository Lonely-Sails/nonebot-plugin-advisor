"""会话级“内存文件系统 + 对话历史”管理。

每个会话（用户/群）拥有一个隔离的 Workspace：
- 用户发来的图片/文件会被保存到本地临时目录，并在内存里登记成“虚拟文件”；
- 文本类文件提供“按行读取”能力，供 agent 用工具分段查看（避免整份塞进上下文）；
- 图片提供视觉描述能力（若配置了视觉模型）；
- 对话历史按轮保存，超时自动清理。

说明：这里的一切都是进程内存态，机器人重启后会清空（符合“内存模拟”的要求）。
"""

from __future__ import annotations

import time
import shutil
import asyncio
from typing import Any, Literal
from pathlib import Path
from dataclasses import field, dataclass

from nonebot import logger

from .llm import LLMClient, image_data_url
from .utils import safe_filename, guess_text_encoding
from .config import TEXT_EXTS, IMAGE_EXTS, Config
from .imageops import image_info, annotate_image, compress_image

FileKind = Literal['image', 'text', 'binary']


@dataclass
class VFile:
    """虚拟文件：登记在内存、内容在磁盘"""

    name: str  # 虚拟文件名（在附件目录下的 basename）
    path: Path  # 磁盘绝对路径
    kind: FileKind
    size: int = 0
    media_type: str = ''
    width: int | None = None
    height: int | None = None
    total_lines: int | None = None  # 文本类文件的行数
    description: str | None = None  # 图片的视觉描述
    sha: str = ''

    def line_count(self) -> int:
        if self.total_lines is None:
            try:
                # 快速数行：不整份读入内存
                with self.path.open('rb') as file:
                    self.total_lines = sum(1 for _ in file)
            except OSError:
                self.total_lines = 0
        return self.total_lines


@dataclass
class Turn:
    """一轮用户或客服消息"""

    role: Literal['user', 'assistant']
    text: str
    media: list[VFile] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def _media_marker(self, *, inline_images: bool = False) -> str:
        """把附件信息以文本形式“塞进历史”，让模型知道用户带了什么。"""
        if not self.media:
            return ''
        lines: list[str] = []
        for media in self.media:
            if media.kind == 'image':
                dims = f'（{media.width}×{media.height}）' if media.width else ''
                if inline_images:
                    # 图片已以多模态内联给主模型，无需文字描述或工具提示
                    lines.append(f'[附图 {media.name}{dims}]')
                elif media.description:
                    lines.append(f'[附图 {media.name}{dims}：{media.description}]')
                else:
                    lines.append(
                        f'[附图 {media.name}{dims}：可用 inspect_image 工具进一步查看]'
                    )
            elif media.kind == 'text':
                line_count = media.line_count()
                lines.append(
                    f'[附带文本文件 {media.name}（约 {line_count} 行）：'
                    f'可用 read_file 按行查看]'
                )
            else:
                lines.append(f'[附带文件 {media.name}（{fmt_size(media.size)}）]')
        return '\n' + '\n'.join(lines)

    def to_openai(self, *, inline_images: bool, max_images: int) -> dict[str, Any]:
        """转成 openai 消息。assistant 永远只有文本；user 可按需内联图片。"""
        if self.role == 'assistant':
            return {'role': 'assistant', 'content': self.text or '...'}
        text = (self.text or '') + self._media_marker(inline_images=inline_images)
        if inline_images and self.media:
            image_count = len(
                [media for media in self.media[:max_images] if media.kind == 'image']
            )
            if image_count:
                logger.debug(
                    f'Inlining {image_count} image(s) from user turn to the main model'
                )
        if not inline_images or not self.media:
            return {'role': 'user', 'content': text}
        content: list[dict[str, Any]] = [{'type': 'text', 'text': text}]
        for media in self.media[:max_images]:
            if media.kind != 'image':
                continue
            try:
                # 压缩为 JPEG 字节再内联，减小 base64 体积
                data = compress_image(str(media.path))
                content.append(
                    {
                        'type': 'image_url',
                        'image_url': {'url': image_data_url(data, 'jpg')},
                    }
                )
            except OSError as error:
                logger.warning(f'Failed to read image {media.name}: {error}')
        return {'role': 'user', 'content': content}


class Conversation:
    """单个会话的全部状态"""

    def __init__(
        self,
        key: str,
        user_name: str,
        attach_dir: Path,
        cfg: Config,
    ) -> None:
        self.key = key
        self.user_name = user_name or '用户'
        self.attach_dir = attach_dir
        self.cfg = cfg
        self.attach_dir.mkdir(parents=True, exist_ok=True)
        self.turns: list[Turn] = []
        self.files: dict[str, VFile] = {}
        # agent 本次运行中要随回复一起发给用户的媒体（标注后的图片等）
        self.pending_send: list[VFile] = []
        self.last_active = time.time()
        self.lock = asyncio.Lock()
        self._counter = 0

    # ── 历史 ────────────────────────────────────────────────────────────
    def touch(self) -> None:
        self.last_active = time.time()

    def _append(self, turn: Turn) -> None:
        self.turns.append(turn)
        # 裁剪历史：保留最近 N 轮（user+assistant 各计一轮）
        cap = self.cfg.advisor_history_max_turns * 2
        if len(self.turns) > cap:
            self.turns = self.turns[-cap:]

    def add_user_turn(self, text: str, media: list[VFile] | None = None) -> None:
        self._append(Turn(role='user', text=(text or '').strip(), media=media or []))
        self.touch()

    def add_turn(self, role: Literal['user', 'assistant'], text: str) -> None:
        text = (text or '').strip()
        if not text and role == 'assistant':
            text = '...'
        self._append(Turn(role=role, text=text))
        self.touch()

    def user_text_of_last(self) -> str:
        for t in reversed(self.turns):
            if t.role == 'user':
                return t.text
        return ''

    # ── 文件注册 ────────────────────────────────────────────────────────
    def _unique_name(self, name: str) -> str:
        name = safe_filename(name)
        base = Path(name).stem
        ext = Path(name).suffix
        candidate = name
        while candidate in self.files:
            self._counter += 1
            candidate = f'{base}_{self._counter}{ext}'
        return candidate

    async def register_media(
        self,
        raw_name: str,
        data: bytes,
        llm: LLMClient | None,
        *,
        media_type: str = '',
    ) -> VFile:
        """登记一个上传的图片/文件。数据写入磁盘，并登记为虚拟文件。"""
        if len(data) > self.cfg.advisor_upload_max_bytes:
            raise ValueError(
                f'文件过大（{len(data) // 1024 // 1024}MB），上限 '
                f'{self.cfg.advisor_upload_max_bytes // 1024 // 1024}MB'
            )
        name = self._unique_name(raw_name or 'file')
        path = self.attach_dir / name
        logger.debug(
            f'Writing attachment {name}: {len(data)} bytes media_type={media_type!r}'
        )
        path.write_bytes(data)
        ext = Path(name).suffix.lower()
        kind: FileKind
        if ext in IMAGE_EXTS or (media_type or '').startswith('image/'):
            kind = 'image'
        elif ext in TEXT_EXTS or ext in self.cfg.advisor_upload_text_exts:
            kind = 'text'
        elif guess_text_encoding(data) is not None:
            kind = 'text'
        else:
            kind = 'binary'

        vf = VFile(
            name=name,
            path=path,
            kind=kind,
            size=len(data),
            media_type=media_type,
        )
        if kind == 'image':
            try:
                info = image_info(path)
                vf.width = info['width']
                vf.height = info['height']
                vf.media_type = vf.media_type or f'image/{info["format"]}'
            except Exception:
                pass
            # 图片内联模式下，图片会以多模态直接喂给主模型，无需再转成文字描述
            if not self.cfg.advisor_image_inline and llm and llm.available:
                try:
                    ext_img = Path(name).suffix.lstrip('.') or 'png'
                    vf.description = await llm.describe_image_file(str(path), ext_img)
                except Exception as error:
                    logger.warning(f'Failed to describe image {name}: {error}')
                    vf.description = None
        elif kind == 'text':
            # 只登记行数（供按行读取），内容留在磁盘
            vf.total_lines = data.count(b'\n') + (0 if data.endswith(b'\n') else 1)
            if vf.total_lines > self.cfg.advisor_upload_max_lines:
                logger.info(
                    f'{name} has {vf.total_lines} lines, exceeding the '
                    f'{self.cfg.advisor_upload_max_lines}-line limit; '
                    'still readable by line'
                )
        self.files[name] = vf
        self.touch()
        logger.info(
            f'Session {self.key} added attachment {name}: kind={kind} '
            f'size={fmt_size(len(data))}'
            + (f' description={vf.description[:60]}' if vf.description else '')
        )
        return vf

    # ── 文本文件读取 ────────────────────────────────────────────────────
    def describe_text_file(
        self, name: str, start: int = 1, end: int | None = None
    ) -> str | None:
        """读取文本文件指定行（1 起始）。返回格式化文本或 None。"""
        vf = self.files.get(name)
        if vf is None or vf.kind != 'text':
            return None
        total = vf.line_count()
        if total <= 0:
            return f'（{name} 是空文件，0 行）'
        s = max(1, start or 1)
        e = min(total, end or min(total, s + 199))
        if s > total:
            return f'（{name} 共 {total} 行，起始行超过文件长度）'
        try:
            with vf.path.open(encoding='utf-8', errors='replace') as file:
                lines = []
                for i, line in enumerate(file, 1):
                    if i > e:
                        break
                    if i >= s:
                        lines.append(line.rstrip('\n'))
        except OSError as exc:
            return f'（读取失败：{exc}）'
        head = f'（{name} 共 {total} 行，当前显示 {s}-{e} 行）\n'
        body = '\n'.join(
            (ln if len(ln) <= 4000 else ln[:4000] + '…[行过长已截断]') for ln in lines
        )
        return head + body

    def search_in_file(self, name: str, keyword: str, limit: int = 10) -> str | None:
        """在文本文件里搜关键词，返回命中行。"""
        vf = self.files.get(name)
        if vf is None or vf.kind != 'text' or not keyword:
            return None
        kw = keyword.lower()
        hits: list[str] = []
        try:
            with vf.path.open(encoding='utf-8', errors='replace') as file:
                for i, line in enumerate(file, 1):
                    if kw in line.lower():
                        hits.append(f'{i}: {line.rstrip()[:1500]}')
                        if len(hits) >= limit:
                            break
        except OSError as exc:
            return f'（读取失败：{exc}）'
        if not hits:
            return f'（{name} 中没有找到“{keyword}”）'
        return f'（{name} 中找到 {len(hits)} 处“{keyword}”）\n' + '\n'.join(hits)

    # ── 图片读取/标注 ───────────────────────────────────────────────────
    def describe_image(self, name: str) -> str | None:
        vf = self.files.get(name)
        if vf is None or vf.kind != 'image':
            return None
        dims = f'{vf.width}×{vf.height}' if vf.width else '尺寸未知'
        if vf.description:
            return f'（{name}，{dims}，视觉描述：{vf.description}）'
        return f'（{name}，{dims}，暂无描述，可调用工具生成）'

    # ── 待发送媒体 ──────────────────────────────────────────────────────
    def attach_to_send(self, name: str) -> VFile | None:
        vf = self.files.get(name)
        if vf is None:
            return None
        if vf not in self.pending_send:
            self.pending_send.append(vf)
        return vf

    # 工具列表视图
    def list_file_infos(self) -> list[dict]:
        return [
            {
                'name': f.name,
                'kind': f.kind,
                'size': f.size,
                'lines': f.total_lines,
                'width': f.width,
                'height': f.height,
                'description': f.description,
            }
            for f in self.files.values()
        ]

    def annotate(self, name: str, ops: list[dict]) -> VFile:
        """对某张图片执行标注，结果作为新文件登记并标记待发送。"""
        vf = self.files.get(name)
        if vf is None or vf.kind != 'image':
            raise ValueError(f'找不到图片附件：{name}')
        out = annotate_image(vf.path, ops, dst_dir=self.attach_dir / '_annotated')
        vf2 = VFile(
            name=out.name,
            path=out,
            kind='image',
            size=out.stat().st_size,
            description=f'由 {name} 标注生成',
        )
        info = image_info(out)
        vf2.width = info['width']
        vf2.height = info['height']
        self.files[out.name] = vf2
        if vf2 not in self.pending_send:
            self.pending_send.append(vf2)
        return vf2


class SessionMemory:
    """全部会话的注册表，负责按 key 获取/创建会话，并定期清理过期会话。"""

    def __init__(self, cfg: Config, base_dir: Path) -> None:
        self.cfg = cfg
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._convs: dict[str, Conversation] = {}

    def get(self, key: str, user_name: str = '') -> Conversation:
        conv = self._convs.get(key)
        if conv is None:
            conv = Conversation(
                key=key,
                user_name=user_name,
                attach_dir=self.base_dir / key,
                cfg=self.cfg,
            )
            self._convs[key] = conv
            logger.debug(f'Created session {key} (attachment dir {conv.attach_dir})')
        else:
            if user_name:
                conv.user_name = user_name
            conv.touch()
        return conv

    def drop(self, key: str) -> bool:
        return self._convs.pop(key, None) is not None

    def purge_expired(self) -> int:
        """清理超过 TTL 未活跃的会话。"""
        now = time.time()
        ttl = self.cfg.advisor_conversation_ttl
        expired = [k for k, c in self._convs.items() if now - c.last_active > ttl]
        for k in expired:
            self._convs.pop(k, None)
            try:
                d = self.base_dir / k
                if d.exists() and d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
        return len(expired)

    def count(self) -> int:
        return len(self._convs)


# 供工具报告使用的辅助
def fmt_size(n: int) -> str:
    if n < 1024:
        return f'{n}B'
    if n < 1024 * 1024:
        return f'{n / 1024:.1f}KB'
    return f'{n / 1024 / 1024:.1f}MB'
