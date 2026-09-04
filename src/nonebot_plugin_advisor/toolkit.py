"""Agent 工具集：把文档、会话文件、图片标注、联网搜索暴露成 function calling。

每个工具都有一个 JSON Schema 与一个 async 处理器。处理器通过 ToolContext
拿到当前会话 / 知识库 / LLM 客户端，返回给模型的纯文本结果。
"""
# ruff: noqa: E501 —— 本文件含大量面向模型的中文长描述，属有意为之

from __future__ import annotations

import json
from typing import Any
from dataclasses import field, dataclass
from collections.abc import Callable, Awaitable

from nonebot import logger

from .llm import LLMClient, VisionUnsupported
from .config import Config
from .imageops import image_info
from .searcher import web_search, fetch_webpage
from .docs_store import KnowledgeBase
from .session_store import VFile, Conversation, fmt_size

Handler = Callable[[dict[str, Any], 'ToolContext'], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler


@dataclass
class ToolContext:
    cfg: Config
    llm: LLMClient
    conv: Conversation
    kb: KnowledgeBase | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def kb_or_none(self) -> KnowledgeBase | None:
        return self.kb if (self.kb and self.kb.enabled) else None


# ────────────────────────────────────────────────────────────────────────
# 内部小工具：失败返回字符串而非抛异常（对模型更友好）
# ────────────────────────────────────────────────────────────────────────
def _ok(msg: str) -> str:
    return msg


def _err(msg: str) -> str:
    return f'错误：{msg}'


def _need_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


async def _describe_attachment_image(
    conv: Conversation, llm: LLMClient, name: str
) -> str:
    vf = conv.files.get(name)
    if vf is None or vf.kind != 'image':
        return _err(f'不是可查看的图片附件：{name}（先 list_files 确认名字）')
    if vf.description:
        return vf.description
    if not (llm and llm.available):
        return _err(
            '没有配置视觉模型（advisor_vision_model / LLM），无法生成图片描述，请让用户描述图片内容'
        )
    try:
        ext = name.rsplit('.', 1)[-1] if '.' in name else 'png'
        desc = await llm.describe_image_file(str(vf.path), ext)
        vf.description = desc
        return desc
    except VisionUnsupported as e:
        return _err(f'模型不支持看图：{e}。请让用户用文字描述图片内容')
    except Exception as e:
        logger.warning(f'Failed to describe {name}: {e}')
        return _err(f'图片描述失败：{e}')


# ────────────────────────────────────────────────────────────────────────
# 1. 会话文件工具
# ────────────────────────────────────────────────────────────────────────
async def _list_files(args: dict, ctx: ToolContext) -> str:
    infos = ctx.conv.list_file_infos()
    if not infos:
        return '当前会话没有任何用户上传/附带的文件。（用户后续发图片或 txt/log 会自动出现在这里）'
    lines = ['当前会话中的文件：']
    for f in infos:
        extra = f'，{f["width"]}×{f["height"]}px' if f.get('width') else ''
        if f['kind'] == 'text':
            extra = f'，{f.get("lines", 0)} 行'
        desc = f.get('description')
        head = f'- {f["name"]}（{f["kind"]}{extra}，{fmt_size(f["size"])}）'
        lines.append(head + (f'：{desc}' if desc else ''))
    return '\n'.join(lines)


async def _read_file(args: dict, ctx: ToolContext) -> str:
    name = str(args.get('filename') or args.get('name') or '').strip()
    if not name:
        return _err('缺少 filename')
    vf = ctx.conv.files.get(name)
    if vf is None:
        return _err(f'会话中没有这个文件：{name}。先 list_files 看看有哪些')
    if vf.kind != 'text':
        return _err(
            f'{name} 不是文本文件（kind={vf.kind}），请用 inspect_image 查看图片'
        )
    text = ctx.conv.describe_text_file(
        name,
        _need_int(args.get('start_line'), 1),
        _need_int(args.get('end_line'), 0) or None,
    )
    return text or _err(f'读取失败：{name}')


async def _search_file(args: dict, ctx: ToolContext) -> str:
    name = str(args.get('filename') or args.get('name') or '').strip()
    kw = str(args.get('keyword') or args.get('query') or '').strip()
    if not name or not kw:
        return _err('需要 filename 与 keyword')
    vf = ctx.conv.files.get(name)
    if vf is None:
        return _err(f'会话中没有这个文件：{name}')
    if vf.kind != 'text':
        return _err(f'{name} 不是文本文件')
    return ctx.conv.search_in_file(name, kw) or _err('搜索失败')


async def _inspect_image(args: dict, ctx: ToolContext) -> str:
    name = str(args.get('filename') or args.get('name') or '').strip()
    if not name:
        return _err('缺少 filename')
    vf = ctx.conv.files.get(name)
    if vf is None:
        return _err(f'会话中没有这个文件：{name}。先 list_files 看看')
    if vf.kind != 'image':
        return _err(f'{name} 不是图片，请用 read_file')
    info = ctx.conv.describe_image(name) or f'（{name}，{vf.width}×{vf.height}）'
    return info + (
        '\n若想标注并回发，可用 annotate_image（坐标 0~1000 归一化）'
        if ctx.cfg.advisor_enable_image_tools
        else ''
    )


async def _annotate_image(args: dict, ctx: ToolContext) -> str:
    if not ctx.cfg.advisor_enable_image_tools:
        return _err('图片标注功能已禁用')
    name = str(args.get('filename') or args.get('name') or '').strip()
    ops = args.get('ops')
    if not name or not isinstance(ops, list) or not ops:
        return _err('需要 filename 与 ops（标注操作数组）')
    vf = ctx.conv.files.get(name)
    if vf is None:
        return _err(f'会话中没有这个文件：{name}')
    if vf.kind != 'image':
        return _err(f'{name} 不是图片，无法标注')
    try:
        nf = ctx.conv.annotate(name, ops)
    except Exception as e:
        return _err(str(e))
    return _ok(
        f'标注完成，已生成 {nf.name}（会自动随下一条回复一起发给用户）。'
        f'可在同一回复里配上说明文字，例如‘照着红框这里点一下’。'
    )


async def _send_image(args: dict, ctx: ToolContext) -> str:
    """把会话中的图片随回复一起发给用户。"""
    name = str(args.get('filename') or args.get('name') or '').strip()
    if not name:
        return _err('缺少 filename')
    vf = ctx.conv.files.get(name)
    if vf is None:
        return _err(f'会话中没有这个文件：{name}')
    ctx.conv.attach_to_send(name)
    return _ok(f'已把 {name} 加入待发送，会随下一条回复发给用户')


# ────────────────────────────────────────────────────────────────────────
# 2. 知识库工具
# ────────────────────────────────────────────────────────────────────────
async def _kb_list_docs(args: dict, ctx: ToolContext) -> str:
    kb = ctx.kb_or_none()
    if not kb:
        return _err('知识库未启用（未配置 advisor_kb_source）')
    docs = kb.list_docs(str(args.get('keyword') or ''))
    if not docs:
        return '知识库中还没有文档。'
    return '知识库文档：' + '\n'.join(f'- {d["path"]}' for d in docs[:50])


async def _kb_search(args: dict, ctx: ToolContext) -> str:
    kb = ctx.kb_or_none()
    if not kb:
        return _err('知识库未启用')
    q = str(args.get('query') or '').strip()
    if not q:
        return _err('缺少 query')
    hits = kb.search_docs(q)
    if not hits:
        return f'知识库中没有找到与“{q}”相关的内容，可换关键词，或联网搜索。'
    out = [f'找到 {len(hits)} 条与“{q}”相关的内容：']
    for i, h in enumerate(hits, 1):
        out.append(f'[{i}] {h["path"]}（第 {h["line"]} 行附近）\n{h["snippet"]}')
    return '\n\n'.join(out)


async def _kb_read(args: dict, ctx: ToolContext) -> str:
    kb = ctx.kb_or_none()
    if not kb:
        return _err('知识库未启用')
    path = str(args.get('path') or args.get('file') or '').strip()
    if not path:
        return _err('缺少 path')
    text = kb.read_doc(
        path,
        _need_int(args.get('start_line'), 1),
        _need_int(args.get('end_line'), 0) or None,
    )
    if text is None:
        return _err(f'知识库中没有文档 {path}（可用 list_kb_docs 查看全部路径）')
    return text


async def _kb_list_images(args: dict, ctx: ToolContext) -> str:
    kb = ctx.kb_or_none()
    if not kb:
        return _err('知识库未启用')
    imgs = kb.list_images(str(args.get('keyword') or ''))
    if not imgs:
        return '知识库中还没有图片。'
    lines = ['知识库图片（文档中引用的图/截图）：']
    for im in imgs[:50]:
        lines.append(f'- {im["path"]}：{im["desc"] or "（无描述）"}')
    return '\n'.join(lines)


async def _kb_describe_image(args: dict, ctx: ToolContext) -> str:
    kb = ctx.kb_or_none()
    if not kb:
        return _err('知识库未启用')
    path = str(args.get('path') or args.get('file') or '').strip()
    if not path:
        return _err('缺少 path')
    fp = kb.image_path(path)
    if fp is None:
        return _err(f'知识库中没有图片 {path}')
    desc = kb.image_description(path)
    if desc:
        return f'{path}：{desc}'
    # 未索引过 → 尝试实时描述
    if not (ctx.llm and ctx.llm.available):
        return _err(f'{path} 尚未索引且未配置视觉模型，无法描述')
    try:
        info = image_info(fp)
        d = await ctx.llm.describe_image_file(str(fp), info['format'])
        return f'{path}：{d}'
    except VisionUnsupported as e:
        return _err(f'模型不支持看图：{e}')
    except Exception as e:
        return _err(f'描述失败：{e}')


async def _kb_send_image(args: dict, ctx: ToolContext) -> str:
    """把知识库图片（文档里的图/截图）随回复发给用户。"""
    kb = ctx.kb_or_none()
    if not kb:
        return _err('知识库未启用')
    path = str(args.get('path') or args.get('file') or '').strip()
    if not path:
        return _err('缺少 path')
    fp = kb.image_path(path)
    if fp is None:
        return _err(f'知识库中没有图片 {path}')
    # 直接登记为指向仓库文件的 VFile，这样发送逻辑统一
    if ctx.conv.files.get(path) is None:
        info = image_info(fp)
        vf = VFile(
            name=path,
            path=fp,
            kind='image',
            size=fp.stat().st_size,
            width=info['width'],
            height=info['height'],
            description=kb.image_description(path),
        )
        ctx.conv.files[path] = vf
    ctx.conv.attach_to_send(path)
    return _ok(f'已把知识库图片 {path} 加入待发送，会随下一条回复发给用户')


# ────────────────────────────────────────────────────────────────────────
# 3. 联网工具
# ────────────────────────────────────────────────────────────────────────
async def _web_search(args: dict, ctx: ToolContext) -> str:
    q = str(args.get('query') or '').strip()
    if not q:
        return _err('缺少 query')
    try:
        results = await web_search(
            q,
            searxng_url=ctx.cfg.advisor_searxng_url,
            max_results=_need_int(
                args.get('num_results'), ctx.cfg.advisor_search_max_results
            )
            or ctx.cfg.advisor_search_max_results,
            fetch_content=True,
            content_max_chars=ctx.cfg.advisor_fetch_max_chars,
        )
    except Exception as e:
        return _err(str(e))
    if not results:
        return f'没有搜到与“{q}”相关的结果。'
    out = [f'“{q}”的搜索结果：']
    for r in results:
        out.append(f'- {r.get("title")}\n  {r.get("url")}\n  {r.get("snippet")}')
        content = r.get('content')
        if content:
            out.append(f'  正文：{content}')
    return '\n'.join(out)


async def _open_url(args: dict, ctx: ToolContext) -> str:
    url = str(args.get('url') or '').strip()
    if not url:
        return _err('缺少 url')
    try:
        text = await fetch_webpage(url, max_chars=ctx.cfg.advisor_fetch_max_chars)
    except Exception as e:
        return _err(str(e))
    return f'{url} 的正文内容（节选）：\n{text}'


# ────────────────────────────────────────────────────────────────────────
# 注册表
# ────────────────────────────────────────────────────────────────────────
def _img_op_props() -> dict[str, Any]:
    return {
        'op': {
            'type': 'string',
            'enum': ['rect', 'fill', 'ellipse', 'arrow', 'text'],
            'description': '操作类型',
        },
        'color': {
            'type': 'string',
            'description': '颜色，如 #ff0000 / #ff0000aa 或 red',
        },
        'x': {'type': 'number', 'description': 'X 坐标（0~1000 归一化，左→右）'},
        'y': {'type': 'number', 'description': 'Y 坐标（0~1000 归一化，上→下）'},
        'w': {'type': 'number', 'description': '宽度（归一化 0~1000）'},
        'h': {'type': 'number', 'description': '高度（归一化 0~1000）'},
        'width': {'type': 'integer', 'description': '线条粗细（像素）'},
        'x1': {'type': 'number', 'description': '箭头/线 起点 X'},
        'y1': {'type': 'number', 'description': '箭头/线 起点 Y'},
        'x2': {'type': 'number', 'description': '箭头/线 终点 X'},
        'y2': {'type': 'number', 'description': '箭头/线 终点 Y'},
        'alpha': {'type': 'integer', 'description': 'fill 透明度 0~255'},
        'text': {'type': 'string', 'description': '文字标注内容（≤10 字）'},
        'size': {'type': 'integer', 'description': '文字大小（像素，建议 32~60）'},
        'background': {'type': 'string', 'description': '文字背景色，可选'},
    }


def _param(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {'type': 'object', 'properties': props, 'required': required or []}


def build_tools(cfg: Config) -> list[Tool]:
    tools: list[Tool] = []

    if cfg.advisor_enable_files:
        tools += [
            Tool(
                'list_files',
                '列出用户本次会话中上传/附带的全部文件（截图、txt、log、文档等）。返回文件名、类型、行数/尺寸与图片描述。'
                '用户说“看我发的图/日志”时先调用它确认文件名。',
                _param({}),
                _list_files,
            ),
            Tool(
                'read_file',
                '读取会话中某个文本文件（log/txt 等）的指定行区间，适合分段看日志与报错。不传 end_line 时默认读取最多 200 行。',
                _param(
                    {
                        'filename': {
                            'type': 'string',
                            'description': '会话中列出的文件名',
                        },
                        'start_line': {
                            'type': 'integer',
                            'description': '起始行（1 起）',
                        },
                        'end_line': {'type': 'integer', 'description': '结束行（含）'},
                    },
                    ['filename'],
                ),
                _read_file,
            ),
            Tool(
                'search_file',
                '在会话中的文本文件里按关键词搜索，返回命中行与行号，适合快速定位报错关键字（如 Traceback、Error）。',
                _param(
                    {
                        'filename': {'type': 'string', 'description': '文件名'},
                        'keyword': {'type': 'string', 'description': '要查找的关键字'},
                    },
                    ['filename', 'keyword'],
                ),
                _search_file,
            ),
        ]

    if cfg.advisor_enable_image_tools:
        tools += [
            Tool(
                'inspect_image',
                '查看用户发来的某张图片：返回尺寸与内容描述（必要时调用视觉模型），用于确认截图里是什么。',
                _param(
                    {
                        'filename': {
                            'type': 'string',
                            'description': '图片文件名（来自 list_files）',
                        }
                    },
                    ['filename'],
                ),
                _inspect_image,
            ),
            Tool(
                'annotate_image',
                '在用户发来的图片上画标注，生成一张新图并自动随下一条回复发给用户（用于圈出要点的位置）。'
                '坐标 x/y/w/h 一律用 0~1000 归一化（x 向右、y 向下、w 宽、h 高）。'
                'ops 为操作数组：\n'
                '- rect：空心矩形 {op:rect,x,y,w,h,color,width}\n'
                '- fill：半透明高亮块 {op:fill,x,y,w,h,color,alpha}\n'
                '- ellipse：空心圆/椭圆 {op:ellipse,x,y,w,h,color,width}\n'
                '- arrow：箭头 {op:arrow,x1,y1,x2,y2,color,width}\n'
                '- text：文字 {op:text,x,y,text,size,color}，text 必须不超过 10 个字',
                _param(
                    {
                        'filename': {
                            'type': 'string',
                            'description': '要标注的图片文件名',
                        },
                        'ops': {
                            'type': 'array',
                            'description': '标注操作数组，可一次画多个框/写多个文字',
                            'items': _img_op_props(),
                        },
                    },
                    ['filename', 'ops'],
                ),
                _annotate_image,
            ),
            Tool(
                'send_image',
                '把会话中的某张图片（用户发来的或标注生成的）随下一条回复一起发给用户。',
                _param(
                    {'filename': {'type': 'string', 'description': '文件名'}},
                    ['filename'],
                ),
                _send_image,
            ),
        ]

    if cfg.advisor_enable_knowledge:
        tools += [
            Tool(
                'search_kb',
                '在产品文档知识库中搜索与用户问题相关的内容，返回文档路径与命中片段。回答产品/用法问题前优先用它。',
                _param(
                    {'query': {'type': 'string', 'description': '搜索关键词/问题'}},
                    ['query'],
                ),
                _kb_search,
            ),
            Tool(
                'read_kb_doc',
                '读取知识库某篇文档的指定行区间，用于细看具体步骤/配置。文档路径来自 search_kb / list_kb_docs。',
                _param(
                    {
                        'path': {
                            'type': 'string',
                            'description': '文档相对路径，如 docs/xxx.md',
                        },
                        'start_line': {
                            'type': 'integer',
                            'description': '起始行（1 起）',
                        },
                        'end_line': {'type': 'integer', 'description': '结束行（含）'},
                    },
                    ['path'],
                ),
                _kb_read,
            ),
            Tool(
                'list_kb_docs',
                '列出知识库中的文档文件（可用 keyword 过滤文件名）。',
                _param(
                    {'keyword': {'type': 'string', 'description': '可选，按文件名过滤'}}
                ),
                _kb_list_docs,
            ),
            Tool(
                'list_kb_images',
                '列出知识库中（文档插图/截图）的图片及其已索引描述。当用户问文档里的某张图/界面长啥样时可用。',
                _param(
                    {
                        'keyword': {
                            'type': 'string',
                            'description': '可选，按文件名/描述过滤',
                        }
                    }
                ),
                _kb_list_images,
            ),
            Tool(
                'describe_kb_image',
                '获取知识库中某张图片的描述（没有则调用视觉模型现场生成），用于把文档截图内容讲给用户听。',
                _param(
                    {'path': {'type': 'string', 'description': '图片相对路径'}},
                    ['path'],
                ),
                _kb_describe_image,
            ),
            Tool(
                'send_kb_image',
                '把知识库中的某张图片（文档插图/截图）随下一条回复一起发给用户，配合说明文字使用。',
                _param(
                    {'path': {'type': 'string', 'description': '图片相对路径'}},
                    ['path'],
                ),
                _kb_send_image,
            ),
        ]

    if cfg.advisor_enable_web:
        tools += [
            Tool(
                'web_search',
                '联网搜索最新信息（用于知识库没有、或需要确认版本/更新动态的问题）。返回标题、链接与摘要。',
                _param(
                    {
                        'query': {'type': 'string', 'description': '搜索内容'},
                        'num_results': {'type': 'integer', 'description': '结果条数'},
                    },
                    ['query'],
                ),
                _web_search,
            ),
            Tool(
                'open_url',
                '打开某个网页链接并读取其正文文本（配合 web_search 的链接使用，或用户直接给链接）。',
                _param(
                    {'url': {'type': 'string', 'description': 'http(s) 链接'}}, ['url']
                ),
                _open_url,
            ),
        ]
    return tools


def tools_to_openai(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            'type': 'function',
            'function': {
                'name': t.name,
                'description': t.description,
                'parameters': t.parameters,
            },
        }
        for t in tools
    ]


async def dispatch_tool(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> str:
    logger.debug(
        f'Tool call {tool.name}: {json.dumps(args, ensure_ascii=False)[:200]}'
    )
    try:
        result = await tool.handler(args, ctx)
    except Exception as e:
        logger.warning(f'Tool {tool.name} execution failed: {e}')
        return f'错误：工具执行失败（{type(e).__name__}）：{e}'
    logger.debug(f'Tool {tool.name} returned {len(result)} chars: {result[:300]!r}')
    return result
