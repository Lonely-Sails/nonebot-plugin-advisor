"""文档知识库：定时从仓库拉取 md/mdx + 图片，用视觉模型为图片建立索引。

数据模型
--------
- 数据根目录下 `repo/`：文档仓库工作副本（本地目录源则直接指向该目录）
- `kb_index.json`：索引文件
    {
      "source": 源地址,
      "synced_at": 最后同步时间,
      "head": git 提交号,
      "docs":   {相对路径: {"size": 字节数}},
      "images": {相对路径: {"sha": 哈希, "width":, "height":,
                            "desc": 视觉描述, "error": 失败原因}}
    }
"""

from __future__ import annotations

import os
import re
import time
import asyncio
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import field, dataclass

from nonebot import logger

from .llm import LLMClient, VisionUnsupported
from .utils import json_dump, json_load
from .config import Config
from .imageops import image_info

_GIT_LIKE = re.compile(r'^(https?://|git@|ssh://|git://)')


def _is_git_source(source: str) -> bool:
    if source.endswith('.git'):
        return True
    if _GIT_LIKE.match(source):
        return True
    return False


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _run_git(args: list[str], cwd: Path | None = None, timeout: int = 240) -> str:
    """同步运行 git 命令，返回 stdout。失败抛 RuntimeError。"""
    proc = subprocess.run(
        ['git', *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or 'git 执行失败').strip())
    return proc.stdout.strip()


def _iter_files(root: Path, exts: tuple[str, ...]) -> list[str]:
    """递归收集相对路径（跳过隐藏目录与 .git）。"""
    out: list[str] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地过滤隐藏目录，避免进入
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith('.') and d != '.git' and d != '__pycache__'
        ]
        for fn in filenames:
            if fn.startswith('.'):
                continue
            if Path(fn).suffix.lower() in exts:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                out.append(rel)
    return sorted(out)


@dataclass
class SyncReport:
    """一次同步的结果摘要"""

    ok: bool = True
    message: str = ''
    docs: int = 0
    images: int = 0
    described: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class KnowledgeBase:
    """文档知识库（负责同步 + 索引 + 检索）"""

    def __init__(
        self,
        cfg: Config,
        data_dir: Path,
        llm: LLMClient | None = None,
    ) -> None:
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.index_path = self.data_dir / 'kb_index.json'
        self._index: dict | None = None
        self._text_cache: dict[str, str] = {}
        self._repo_dir: Path | None = None

    # ── 基本属性 ────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.advisor_kb_source.strip())

    @property
    def repo_dir(self) -> Path:
        if self._repo_dir is None:
            source = self.cfg.advisor_kb_source.strip()
            if _is_git_source(source):
                self._repo_dir = self.data_dir / 'repo'
            else:
                # 本地目录：直接指向该目录
                self._repo_dir = Path(source).expanduser()
        return self._repo_dir

    @property
    def index(self) -> dict:
        if self._index is None:
            self._index = json_load(self.index_path, {}) or {}
        return self._index or {}

    # ── 同步 ────────────────────────────────────────────────────────────
    async def sync(self, *, force: bool = False) -> SyncReport:
        """拉取/扫描仓库，并索引（新）图片。"""
        if not self.enabled:
            return SyncReport(
                ok=True, message='未配置 advisor_kb_source，跳过知识库同步'
            )
        report = SyncReport()
        started = time.perf_counter()
        try:
            source = self.cfg.advisor_kb_source.strip()
            logger.debug(f'知识库同步开始: source={source!r} force={force}')
            head: str | None = None
            if _is_git_source(source):
                await asyncio.to_thread(self._git_update, source)
                try:
                    head = _run_git(
                        ['rev-parse', 'HEAD'], cwd=self.repo_dir, timeout=60
                    )
                except Exception:
                    head = None
            elif not self.repo_dir.is_dir():
                raise RuntimeError(f'本地文档目录不存在：{self.repo_dir}')
            report = await self._scan_and_index(force=force, head=head)
        except Exception as e:
            logger.opt(exception=self.cfg.advisor_debug).error(
                f'知识库同步失败: {e}'
            )
            report.ok = False
            report.message = str(e)
        elapsed = time.perf_counter() - started
        if report.ok:
            logger.info(f'知识库同步完成（{elapsed:.0f}s）: {report.message}')
        else:
            logger.warning(
                f'知识库同步未完成（{elapsed:.0f}s）: {report.message}'
            )
        return report

    def _git_update(self, source: str) -> None:
        repo = self.repo_dir
        branch = self.cfg.advisor_kb_branch.strip()
        if not (repo / '.git').exists():
            logger.info(f'首次克隆知识库: {source}')
            args = ['clone', '--depth', '1']
            if branch:
                args += ['--branch', branch]
            args += [source, str(repo)]
            _run_git(args, timeout=600)
        else:
            logger.debug(f'git fetch origin（分支 {branch or "默认"}）')
            _run_git(['fetch', 'origin'], cwd=repo, timeout=600)
            if branch:
                _run_git(['reset', '--hard', f'origin/{branch}'], cwd=repo, timeout=300)
            else:
                _run_git(['pull', '--ff-only'], cwd=repo, timeout=600)

    async def _scan_and_index(self, *, force: bool, head: str | None) -> SyncReport:
        cfg = self.cfg
        root = self.repo_dir
        report = SyncReport()
        index = self.index if not force else {}

        docs = _iter_files(root, cfg.advisor_kb_doc_exts)
        images = _iter_files(root, cfg.advisor_kb_image_exts)

        # 清掉已删除文件对应的旧索引
        for key in list(index.get('images', {})):
            if not (root / key).exists():
                index['images'].pop(key, None)
        for key in list(index.get('docs', {})):
            if not (root / key).exists():
                index['docs'].pop(key, None)
        index['docs'] = {
            p: {'size': (root / p).stat().st_size} for p in docs if (root / p).is_file()
        }
        index['images'] = index.get('images', {})
        # 处理图片
        pending: list[str] = []
        for rel in images:
            fp = root / rel
            if not fp.is_file():
                continue
            try:
                size = fp.stat().st_size
                sha = _sha1(fp.read_bytes())
            except OSError as e:
                report.errors.append(f'{rel}: 读取失败 {e}')
                continue
            old = index['images'].get(rel)
            if (
                not force
                and old
                and old.get('sha') == sha
                and old.get('desc')
                and not old.get('error')
            ):
                continue  # 已索引过，跳过
            index['images'][rel] = {
                'sha': sha,
                'size': size,
                'desc': old.get('desc') if old and old.get('sha') == sha else None,
                'error': None,
            }
            pending.append(rel)

        pending = pending[: cfg.advisor_kb_max_images_per_sync]
        logger.debug(
            f'知识库扫描: 文档 {len(docs)} 图片 {len(images)} '
            f'本次待索引图片 {len(pending)}（force={force} head={head}）'
        )
        if pending:
            describe_report = await self._describe_images(root, index, pending)
            report.described = describe_report[0]
            report.failed = describe_report[1]
            report.errors.extend(describe_report[2])

        index['images'] = {
            k: v for k, v in index['images'].items() if (root / k).exists()
        }
        index['source'] = self.cfg.advisor_kb_source.strip()
        index['head'] = head
        index['synced_at'] = datetime.now(timezone.utc).isoformat()
        self._index = index
        json_dump(self.index_path, index)
        self._text_cache.clear()
        report.docs = len(docs)
        report.images = len(images)
        report.ok = True
        report.message = (
            f'同步完成：文档 {report.docs} 个，图片 {report.images} 张，'
            f'本次新描述 {report.described} 张'
            + (f'，失败 {report.failed} 张' if report.failed else '')
        )
        return report

    async def _describe_images(
        self, root: Path, index: dict, pending: list[str]
    ) -> tuple[int, int, list[str]]:
        """对图片调用视觉模型并写入索引。"""
        if not self.llm or not self.llm.available:
            msg = '未配置 LLM（advisor_llm_api_key），无法为图片生成描述'
            for rel in pending:
                index['images'][rel]['error'] = msg
            self._index = index
            return 0, len(pending), [msg]
        ok = failed = 0
        errors: list[str] = []
        for rel in pending:
            fp = root / rel
            try:
                info = image_info(fp)
                if self.llm is None:
                    raise RuntimeError('llm 不可用')
                desc = await self.llm.describe_image_file(
                    str(fp),
                    info['format'],
                    prompt=(
                        '这是产品文档中的一张插图/截图。请用中文描述图中内容：'
                        '它是什么页面/界面、展示了哪些关键元素与步骤。'
                        '若含报错信息请原文指出。不超过 150 字。'
                    ),
                )
                entry = index['images'].get(rel, {})
                entry['desc'] = desc
                entry['width'] = info['width']
                entry['height'] = info['height']
                entry['error'] = None
                index['images'][rel] = entry
                ok += 1
                logger.debug(f'图片已索引 {rel}: 描述 {len(desc)} 字')
            except VisionUnsupported as e:
                # 模型不支持看图 → 标记错误并跳过，避免每次都重试
                entry = index['images'].get(rel, {})
                entry['error'] = f'vision unsupported: {e}'
                index['images'][rel] = entry
                failed += 1
                errors.append(f'{rel}: 视觉模型不支持图片')
                logger.warning(f'{rel}: 视觉模型不支持图片，已跳过')
                break  # 同一模型，其余图片也不用试了
            except Exception as e:
                entry = index['images'].get(rel, {})
                entry['error'] = f'{type(e).__name__}: {e}'
                index['images'][rel] = entry
                failed += 1
                errors.append(f'{rel}: {e}')
                logger.warning(f'图片索引失败 {rel}: {e}')
            # 及时落盘，避免中断丢失
            self._index = index
            json_dump(self.index_path, index)
        return ok, failed, errors

    # ── 查询接口（供工具使用） ──────────────────────────────────────────
    def doc_count(self) -> int:
        return len(self.index.get('docs', {}))

    def image_count(self) -> int:
        return len(self.index.get('images', {}))

    def synced_at(self) -> str:
        return self.index.get('synced_at', '')

    def source(self) -> str:
        return self.index.get('source', self.cfg.advisor_kb_source.strip())

    def list_docs(self, keyword: str = '', limit: int = 60) -> list[dict]:
        docs = self.index.get('docs', {})
        names = sorted(docs)
        if keyword:
            kw = keyword.lower()
            names = [n for n in names if kw in n.lower()]
        return [{'path': n, 'size': docs[n].get('size', 0)} for n in names[:limit]]

    def list_images(self, keyword: str = '', limit: int = 60) -> list[dict]:
        imgs = self.index.get('images', {})
        names = sorted(imgs)
        if keyword:
            kw = keyword.lower()
            names = [
                n
                for n in names
                if kw in n.lower() or kw in (imgs[n].get('desc') or '').lower()
            ]
        return [
            {
                'path': n,
                'desc': imgs[n].get('desc') or imgs[n].get('error') or '',
            }
            for n in names[:limit]
        ]

    def image_path(self, rel: str) -> Path | None:
        """相对路径 → 本地绝对路径（校验路径合法性，防止穿越）。"""
        return self._safe_resolve(rel, only_image=True)

    def image_description(self, rel: str) -> str | None:
        img = self.index.get('images', {}).get(rel)
        if not img:
            return None
        return img.get('desc') or img.get('error')

    def _safe_resolve(self, rel: str, only_image: bool = False) -> Path | None:
        """把仓库相对路径解析成绝对路径，并做穿越防护。"""
        root = self.repo_dir
        target = (root / rel).resolve()
        if not target.is_relative_to(root.resolve()):
            return None
        if not target.exists():
            return None
        if only_image:
            if Path(rel).suffix.lower() not in self.cfg.advisor_kb_image_exts:
                return None
        return target

    def _read_doc_text(self, rel: str) -> str:
        if rel in self._text_cache:
            return self._text_cache[rel]
        fp = self._safe_resolve(rel)
        if fp is None or not fp.is_file():
            return ''
        try:
            text = fp.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return ''
        self._text_cache[rel] = text
        return text

    def read_doc(
        self, rel: str, start: int | None = None, end: int | None = None
    ) -> str | None:
        """读取文档指定行（1 起始，含 end）。返回文本或 None。"""
        fp = self._safe_resolve(rel)
        if fp is None or not fp.is_file():
            return None
        if Path(rel).suffix.lower() not in self.cfg.advisor_kb_doc_exts:
            return None
        try:
            lines = fp.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            return None
        total = len(lines)
        s = max(1, start or 1)
        e = min(total, end or min(total, s + 199))
        if s > total:
            return ''
        body = '\n'.join(lines[s - 1 : e])
        return f'（{rel} 共 {total} 行，显示 {s}-{e} 行）\n{body}'

    def search_docs(self, query: str, limit: int = 5) -> list[dict]:
        """关键词搜索文档内容，返回带行号片段的命中。"""
        if not self.enabled or not self.repo_dir.is_dir():
            return []
        tokens = [
            t for t in re.split(r'[\s,，。;；、/\\|]+', query.lower()) if len(t) >= 2
        ]
        if not tokens:
            return []
        doc_list = list(self.index.get('docs', {}).keys())
        scored: list[tuple[int, str, str, int]] = []
        for rel in doc_list:
            text = self._read_doc_text(rel)
            if not text:
                continue
            lower = text.lower()
            score = 0
            first = -1
            for tok in tokens:
                if not tok:
                    continue
                count = lower.count(tok)
                if count:
                    score += count
                    idx = lower.find(tok)
                    first = idx if first < 0 else first
            # 文件名命中也加权
            if any(tok in rel.lower() for tok in tokens):
                score += 5
            if score:
                scored.append((score, rel, text, first))
        scored.sort(key=lambda x: -x[0])
        hits: list[dict] = []
        for score, rel, text, idx in scored[:limit]:
            lines = text.splitlines()
            line_no = 1
            # 找到命中所在行
            if idx >= 0:
                line_no = text.count('\n', 0, idx) + 1
            start_line = max(1, line_no - 3)
            end_line = min(len(lines), line_no + 6)
            snippet = '\n'.join(lines[start_line - 1 : end_line])
            if len(snippet) > 800:
                snippet = snippet[:800] + '…'
            hits.append(
                {
                    'path': rel,
                    'score': score,
                    'line': line_no,
                    'total_lines': len(lines),
                    'snippet': snippet,
                }
            )
        return hits

    def status_text(self) -> str:
        if not self.enabled:
            return '知识库：未启用（未配置 advisor_kb_source）'
        synced = self.synced_at()
        parts = [
            f'源：{self.source()}',
            f'文档 {self.doc_count()} 个',
            f'图片 {self.image_count()} 张',
        ]
        if synced:
            parts.append(f'上次同步：{synced}')
        return '知识库：' + '，'.join(parts)
