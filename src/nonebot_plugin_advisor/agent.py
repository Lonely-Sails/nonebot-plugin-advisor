"""Agent 主循环：让模型通过 function calling 读取文档/文件/图片、联网搜索，
最后以客服式口吻给出回答。
"""

from __future__ import annotations

import json
from typing import Any
from dataclasses import field, dataclass

from nonebot import logger

from .llm import LLMError, LLMClient, LLMNotConfigured
from .config import Config
from .toolkit import Tool, ToolContext, dispatch_tool, tools_to_openai
from .docs_store import KnowledgeBase
from .session_store import VFile, Conversation


@dataclass
class AgentResult:
    text: str
    images: list[VFile] = field(default_factory=list)
    ok: bool = True
    note: str = ''


def _json_args(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw or '{}')
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


async def run_agent(
    *,
    cfg: Config,
    llm: LLMClient,
    conv: Conversation,
    kb: KnowledgeBase | None,
    tools: list[Tool],
    system_prompt: str,
    platform: str = '',
    user_id: str = '',
) -> AgentResult:
    """执行一轮 agent 对话，返回最终回答与要发送的图片。"""
    if not llm.available:
        raise LLMNotConfigured('LLM 未配置')

    # 每次运行前清空“待发送”图片，避免残留
    conv.pending_send.clear()

    ctx = ToolContext(cfg=cfg, llm=llm, conv=conv, kb=kb)

    # 组装消息：系统提示 + 会话历史（最后一条用户消息按需内联图片）
    messages: list[dict[str, Any]] = [{'role': 'system', 'content': system_prompt}]
    for i, turn in enumerate(conv.turns):
        inline = cfg.advisor_image_inline and i == len(conv.turns) - 1
        messages.append(turn.to_openai(inline_images=inline, max_images=8))

    openai_tools = tools_to_openai(tools) if tools else []

    max_rounds = cfg.advisor_agent_max_rounds or 1
    rounds = 0
    final_text: str | None = None
    error_msg = ''

    try:
        while rounds < max_rounds:
            rounds += 1
            logger.debug(f'[advisor] agent round {rounds}/{max_rounds}')
            msg = await llm.chat(
                messages,
                tools=openai_tools or None,
                tool_choice='auto' if openai_tools else None,
            )
            messages.append(msg)

            tool_calls = msg.get('tool_calls')
            if not tool_calls:
                final_text = (msg.get('content') or '').strip()
                break

            # 执行本轮所有工具
            for tc in tool_calls:
                fn = tc.get('function') or {}
                name = fn.get('name', '')
                tool = next((t for t in tools if t.name == name), None)
                if tool is None:
                    result = f'错误：未知工具 {name}'
                    logger.warning(f'[advisor] 模型调用了未知工具: {name}')
                else:
                    args = _json_args(fn.get('arguments', '{}'))
                    result = await dispatch_tool(tool, args, ctx)
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': tc.get('id', ''),
                        'content': result,
                    }
                )
            # 工具全部执行完，若已达轮次上限则尝试让模型收尾
            if rounds >= max_rounds:
                try:
                    msg = await llm.chat(messages, tools=None)
                    final_text = (msg.get('content') or '').strip()
                except Exception as e:
                    error_msg = str(e)
                break

        if final_text is None and not error_msg:
            error_msg = 'Agent 未能在轮次限制内给出回答'
    except LLMNotConfigured:
        raise
    except LLMError as e:
        error_msg = str(e)
    except Exception as e:
        logger.opt(exception=cfg.advisor_debug).error(f'[advisor] agent 运行异常: {e}')
        error_msg = f'{type(e).__name__}: {e}'

    # 兜底
    if final_text is None:
        final_text = (
            cfg.advisor_cannot_answer_text or '抱歉，我这边暂时处理不了，请稍后再试～'
        )
        if error_msg and cfg.advisor_debug:
            final_text += f'\n（内部错误：{error_msg}）'

    # 写入历史（仅保留最终文本）
    conv.add_turn('assistant', final_text)
    images = list(conv.pending_send)
    conv.pending_send.clear()
    conv.touch()
    return AgentResult(text=final_text, images=images, ok=not error_msg, note=error_msg)


def build_messages_preview(conv: Conversation) -> list[dict[str, Any]]:
    """供调试：查看即将发送的会话历史（不含系统提示）。"""
    return [t.to_openai(inline_images=False, max_images=0) for t in conv.turns]
