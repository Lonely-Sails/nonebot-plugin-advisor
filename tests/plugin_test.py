"""nonebug 集成测试：控制命令与主客服入口（LLM 未配置时的兜底）。"""

from __future__ import annotations

import pytest
from fake import fake_group_message_event_v11
from nonebug import App


@pytest.mark.asyncio
async def test_help_command(app: App):
    import nonebot
    from nonebot.adapters.onebot.v11 import Bot, Message
    from nonebot.adapters.onebot.v11 import Adapter as OnebotV11Adapter

    from nonebot_plugin_advisor import help_matcher
    from nonebot_plugin_advisor.config import plugin_config
    from nonebot_plugin_advisor.prompts import build_help_text

    event = fake_group_message_event_v11(message='客服帮助')
    async with app.test_matcher(help_matcher) as ctx:
        adapter = nonebot.get_adapter(OnebotV11Adapter)
        bot = ctx.create_bot(base=Bot, adapter=adapter)
        ctx.receive_event(bot, event)
        ctx.should_call_send(
            event, Message(build_help_text(plugin_config)), result=None, bot=bot
        )
        ctx.should_finished()


@pytest.mark.asyncio
async def test_reset_command(app: App):
    import nonebot
    from nonebot.exception import ActionFailed
    from nonebot.adapters.onebot.v11 import Bot, Message
    from nonebot.adapters.onebot.v11 import Adapter as OnebotV11Adapter

    from nonebot_plugin_advisor import reset_matcher

    event = fake_group_message_event_v11(message='客服重置')
    async with app.test_matcher(reset_matcher) as ctx:
        adapter = nonebot.get_adapter(OnebotV11Adapter)
        bot = ctx.create_bot(base=Bot, adapter=adapter)
        # uninfo 注入 Session 时会拉取群信息，这里让它失败并优雅降级
        ctx.should_call_api(
            'get_group_info',
            {'group_id': 87654321},
            exception=ActionFailed('mocked'),
        )
        ctx.should_call_api(
            'get_group_member_info',
            {'group_id': 87654321, 'user_id': 12345678, 'no_cache': True},
            exception=ActionFailed('mocked'),
        )
        ctx.receive_event(bot, event)
        ctx.should_call_send(
            event, Message('好哒，已忘记之前的对话，重新开始～'), result=None, bot=bot
        )
        ctx.should_finished()


@pytest.mark.asyncio
async def test_chat_without_llm_config(app: App):
    """未配置 LLM 时 @机器人提问，应提示未启用，且不调用模型/不发 ack。"""
    import nonebot
    from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment
    from nonebot.adapters.onebot.v11 import Adapter as OnebotV11Adapter

    from nonebot_plugin_advisor import chat

    msg = Message([MessageSegment.at('1'), MessageSegment.text('怎么排查问题?')])
    event = fake_group_message_event_v11(message=msg, to_me=True)
    async with app.test_matcher(chat) as ctx:
        adapter = nonebot.get_adapter(OnebotV11Adapter)
        bot = ctx.create_bot(base=Bot, adapter=adapter)
        ctx.receive_event(bot, event)
        expected = Message(
            '客服未启用：请在环境变量中配置 ADVISOR_LLM_API_KEY 等参数。'
        )
        ctx.should_call_send(event, expected, result=None, bot=bot)
        ctx.should_finished()
