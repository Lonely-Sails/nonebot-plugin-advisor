"""nonebot-plugin-advisor：基于 OpenAI 兼容格式的跨平台 LLM 客服。

核心能力：
- @机器人提问 → agent 读取文档/会话文件/联网搜索后，以“真人客服”口吻回答；
- 用户发图/文件（txt、log 等）会自动进入“内存文件系统”，供 agent 分段查看；
- 引用消息里的图片/文件同样会进入会话；
- 图片可用工具圈注（画框/箭头/≤10 字文字）后回发给用户；
- 可配置定时从文档仓库拉取 md/图片，用视觉模型为图片建立索引；
- 跨平台：基于 nonebot-plugin-alconna / nonebot-plugin-uninfo。
"""

from __future__ import annotations

import asyncio
from typing import Any

from nonebot import logger, require, get_driver, on_message
from nonebot.rule import to_me
from arclet.alconna import Alconna
from nonebot.plugin import PluginMetadata, inherit_supported_adapters
from nonebot.adapters import Bot, Event

require('nonebot_plugin_uninfo')
require('nonebot_plugin_alconna')
require('nonebot_plugin_localstore')
require('nonebot_plugin_apscheduler')

from nonebot_plugin_uninfo import Uninfo, get_session
from nonebot_plugin_alconna import on_alconna
from nonebot_plugin_localstore import get_plugin_data_dir, get_plugin_cache_dir
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_alconna.uniseg import Image, UniMessage

from .llm import LLMClient, LLMNotConfigured
from .agent import run_agent
from .media import parse_event_message
from .config import Config, is_superuser
from .config import plugin_config as cfg
from .prompts import build_help_text, build_system_prompt
from .toolkit import Tool, build_tools
from .docs_store import KnowledgeBase
from .session_store import SessionMemory

__plugin_meta__ = PluginMetadata(
    name='LLM 客服 Advisor',
    description=(
        'OpenAI 兼容格式的跨平台 LLM 客服：读文档、查日志、看图、标注回发、联网搜索'
    ),
    usage=(
        '在群里 @我（或私聊）提问即可；可发送截图/txt/log 协助排查。\n'
        '发送「客服重置」清空记忆；「客服帮助」查看说明；「客服状态」查看状态。'
    ),
    type='application',
    homepage='https://github.com/owner/nonebot-plugin-advisor',
    config=Config,
    supported_adapters=inherit_supported_adapters(
        'nonebot_plugin_alconna', 'nonebot_plugin_uninfo'
    ),
    extra={'author': 'owner <your@mail.com>'},
)

driver = get_driver()

# ── 运行时单例（懒加载） ────────────────────────────────────────────────
_llm: LLMClient | None = None
_kb: KnowledgeBase | None = None
_memory: SessionMemory | None = None
_tools: list[Tool] = []
_services_ready = False
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    """创建后台任务并持有引用，防止被 GC 提前回收。"""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient(cfg)
    return _llm


def _get_memory() -> SessionMemory:
    global _memory
    if _memory is None:
        base = get_plugin_cache_dir() / 'sessions'
        _memory = SessionMemory(cfg, base)
    return _memory


def _get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        data_dir = get_plugin_data_dir() / 'knowledge'
        _kb = KnowledgeBase(cfg, data_dir, llm=_get_llm())
    return _kb


def _ensure_services() -> None:
    global _tools, _services_ready
    if not _services_ready:
        _tools = build_tools(cfg)
        _services_ready = True


# ── 工具/会话 key ───────────────────────────────────────────────────────
def _session_key(session: Uninfo) -> str:
    """会话唯一 key：私聊=用户；群聊=群+用户（每个用户独立的客服会话）。"""
    # session.id 已按此规则由 uninfo 计算
    return str(getattr(session, 'id', '') or session.scene_path or session.user.id)


def _kb_line(session: Uninfo | None = None) -> str:
    """系统提示中关于知识库的一句话描述。"""
    if _kb and _kb.enabled:
        parts = [f'知识库文档 {_kb.doc_count()} 篇、图片 {_kb.image_count()} 张']
        if cfg.advisor_kb_description:
            parts.insert(0, cfg.advisor_kb_description)
        if _kb.synced_at():
            parts.append(f'（同步于 {_kb.synced_at()[:16].replace("T", " ")}）')
        return '你可以通过工具检索知识库后回答用户。' + '，'.join(parts) + '。'
    if cfg.advisor_kb_description:
        return (
            f'关于「{cfg.advisor_kb_description}」的问题，请以通用技术支持的方式帮助用户，'
            '不确定时引导用户提供截图/日志或建议人工。知识库未启用。'
        )
    return (
        '知识库未启用：请以通用技术支持的方式帮助用户；不确定时引导用户'
        '提供截图/日志或建议人工。'
    )


# ── 控制命令识别（避免误入客服对话） ───────────────────────────────────
_CONTROL_PLAIN: set[str] = {
    '客服重置',
    '客服帮助',
    '客服状态',
    '客服同步',
    'advisorreset',
    'advisor help',
    'advisor status',
    'advisor sync',
}


def _normalize(text: str) -> str:
    return ''.join(text.split()).lower()


def _is_control(text: str) -> bool:
    norm = _normalize(text)
    return any(norm == _normalize(c) for c in _CONTROL_PLAIN)


async def _not_control(bot: Bot, event: Event) -> bool:
    try:
        return not _is_control(event.get_plaintext())
    except Exception:
        return True


async def _not_from_self(bot: Bot, event: Event) -> bool:
    try:
        self_id = str(getattr(bot, 'self_id', '') or '')
        return event.get_user_id() != self_id
    except Exception:
        return True


# ── 主客服对话（@触发） ────────────────────────────────────────────────
chat = on_message(
    rule=to_me() & _not_control & _not_from_self,
    priority=20,
    block=True,
)


async def _register_attachments(
    conv,
    parsed,
) -> list[Any]:
    """把解析出的附件登记进会话（虚拟文件系统），返回 VFile 列表。"""
    vfs: list[Any] = []
    for item in parsed.media:
        try:
            vf = await conv.register_media(
                item.name, item.data, _get_llm(), media_type=item.media_type
            )
            vfs.append(vf)
        except ValueError as e:
            parsed.note = f'{parsed.note}；{e}'.lstrip('；')
    return vfs


async def _handle_chat(bot: Bot, event: Event) -> None:
    """客服主流程。未配置 LLM 时直接提示（不触发 uninfo/模型调用）。"""
    if not cfg.advisor_llm_api_key:
        await UniMessage.text(
            '客服未启用：请在环境变量中配置 ADVISOR_LLM_API_KEY 等参数。'
        ).send()
        return
    _ensure_services()
    session = await get_session(bot, event)
    if session is None:  # 极少适配器拿不到会话信息时的兜底
        key = event.get_session_id()
        user_name = ''
        scene_name = '聊天'
    else:
        key = _session_key(session)
        user_name = getattr(session.user, 'name', '') or ''
        scene_name = session.scene.name or getattr(session.scene, 'id', '') or '聊天'
    conv = _get_memory().get(key, user_name)

    # 1) 解析当前消息 + 引用消息
    parsed = await parse_event_message(event, bot)
    vfs = await _register_attachments(conv, parsed)
    user_text = parsed.text.strip()
    if parsed.note:
        user_text = f'{user_text}\n（系统提示：{parsed.note}）'.strip()
    if not user_text and not vfs:
        user_text = '（用户只是 @ 了我，没有发具体内容）'
    conv.add_user_turn(user_text, vfs)

    # 2) 先回一句“收到”，再异步处理（体感更接近真人）
    if cfg.advisor_ack_text:
        try:
            await UniMessage.text(cfg.advisor_ack_text).send(reply_to=True)
        except Exception:
            pass

    # 3) 加锁运行 agent（同一会话串行，避免消息错乱）
    async with conv.lock:
        try:
            result = await run_agent(
                cfg=cfg,
                llm=_get_llm(),
                conv=conv,
                kb=_kb if _kb and _kb.enabled else None,
                tools=_tools,
                system_prompt=build_system_prompt(
                    cfg,
                    user_name=conv.user_name,
                    platform_desc=scene_name,
                    kb_line=_kb_line(session),
                    tool_names=[t.name for t in _tools],
                ),
            )
        except LLMNotConfigured:
            await UniMessage.text('客服未配置 LLM，请检查 ADVISOR_LLM_API_KEY。').send(
                reply_to=True
            )
            return
        except Exception as e:
            logger.opt(exception=True).error(f'[advisor] agent 执行异常: {e}')
            await UniMessage.text(cfg.advisor_cannot_answer_text).send(reply_to=True)
            return

    # 4) 发送结果（文本 + 可能的标注图）
    parts: list[Any] = [result.text]
    parts.extend(Image(path=str(v.path)) for v in result.images)
    await UniMessage(parts).send(reply_to=True)


@chat.handle()
async def _(bot: Bot, event: Event):
    # 交给真正的处理函数，方便单元测试直接调用 _handle_chat
    await _handle_chat(bot, event)


# ── 控制命令（alconna） ────────────────────────────────────────────────
def _user_id_of(session: Uninfo) -> str:
    return str(session.user.id)


reset_matcher = on_alconna(
    Alconna('客服重置'), aliases={'客服 重置', 'advisor reset'}, priority=5, block=True
)
help_matcher = on_alconna(
    Alconna('客服帮助'), aliases={'客服 帮助', 'advisor help'}, priority=5, block=True
)
status_matcher = on_alconna(
    Alconna('客服状态'), aliases={'客服 状态', 'advisor status'}, priority=5, block=True
)
sync_matcher = on_alconna(
    Alconna('客服同步'), aliases={'客服 同步', 'advisor sync'}, priority=5, block=True
)


@reset_matcher.handle()
async def _reset(session: Uninfo):
    key = _session_key(session)
    if _memory is not None:
        _memory.drop(key)
    await UniMessage.text('好哒，已忘记之前的对话，重新开始～').send()


@help_matcher.handle()
async def _help():
    await UniMessage.text(build_help_text(cfg)).send()


@status_matcher.handle()
async def _status(session: Uninfo):
    _ensure_services()
    llm_state = (
        f'LLM：{"已配置" if cfg.advisor_llm_api_key else "未配置"}'
        f'（{cfg.advisor_llm_model}）'
    )
    conv = _get_memory().get(_session_key(session))
    lines = [
        llm_state,
        f'联网搜索：{"开" if cfg.advisor_enable_web else "关"}',
        _kb_line(session) if _kb else '知识库：未初始化',
        f'会话文件工具：{"开" if cfg.advisor_enable_files else "关"}；'
        f'图片工具：{"开" if cfg.advisor_enable_image_tools else "关"}',
        f'当前会话记忆：{len(conv.turns)} 条',
    ]
    await UniMessage.text('\n'.join(lines)).send()


@sync_matcher.handle()
async def _sync(bot: Bot, event: Event, session: Uninfo):
    if not is_superuser(_user_id_of(session)):
        await UniMessage.text('只有超级管理员才能触发文档同步哦～').send()
        return
    kb = _get_kb()
    if not kb.enabled:
        await UniMessage.text('未配置文档仓库（ADVISOR_KB_SOURCE），无法同步。').send()
        return
    await UniMessage.text('收到，开始同步文档仓库，可能需要几分钟…').send()

    # 后台执行，完成后主动通知
    async def _run():
        report = await kb.sync(force=True)
        msg = report.message or '同步失败'
        if report.errors:
            msg += '\n' + '\n'.join(report.errors[:5])
        try:
            await UniMessage.text(f'知识库同步完成：{msg}').send(target=event, bot=bot)
        except Exception as e:
            logger.error(f'[advisor] 同步结果发送失败: {e}')

    _spawn(_run())


# ── 定时任务 ────────────────────────────────────────────────────────────
async def _kb_sync_job() -> None:
    try:
        report = await _get_kb().sync(force=False)
        logger.info(f'[advisor] 定时知识库同步: {report.message or report.ok}')
    except Exception as e:
        logger.error(f'[advisor] 定时知识库同步失败: {e}')


async def _purge_job() -> None:
    try:
        n = _get_memory().purge_expired()
        if n:
            logger.info(f'[advisor] 清理过期会话 {n} 个')
    except Exception as e:
        logger.warning(f'[advisor] 清理过期会话失败: {e}')


@driver.on_startup
async def _on_startup() -> None:
    _ensure_services()
    # 定期清理过期会话
    scheduler.add_job(
        _purge_job,
        'interval',
        minutes=max(5, cfg.advisor_conversation_ttl // 120),
        id='advisor_purge',
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    # 知识库定时同步
    kb = _get_kb()
    if kb.enabled:
        if cfg.advisor_kb_sync_cron.strip():
            scheduler.add_job(
                _kb_sync_job,
                'cron',
                **(_cron_kwargs(cfg.advisor_kb_sync_cron)),
                id='advisor_kb_sync',
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            logger.info(f'[advisor] 知识库定时同步已注册: {cfg.advisor_kb_sync_cron}')
        elif cfg.advisor_kb_sync_interval > 0:
            scheduler.add_job(
                _kb_sync_job,
                'interval',
                minutes=cfg.advisor_kb_sync_interval,
                id='advisor_kb_sync',
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            logger.info(
                '[advisor] 知识库定时同步已注册: 每 '
                f'{cfg.advisor_kb_sync_interval} 分钟'
            )
        # 启动后同步一次（不阻塞启动）
        _spawn(_kb_sync_job())


def _cron_kwargs(expr: str) -> dict[str, str]:
    """把 '分 时 日 月 周' 转成 apscheduler cron 参数。"""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError('cron 表达式需要 5 段：分 时 日 月 周')
    names = ['minute', 'hour', 'day', 'month', 'day_of_week']
    out: dict[str, str] = {}
    for name, val in zip(names, fields):
        if val not in ('*', '?', ''):
            out[name] = val
    return out
