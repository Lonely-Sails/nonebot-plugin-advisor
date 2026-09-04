"""核心纯逻辑单测：会话虚拟文件系统、图片标注、提示词。"""

from __future__ import annotations

import io

import pytest


def _png_bytes(width: int = 200, height: int = 100) -> bytes:
    from PIL import Image

    img = Image.new('RGB', (width, height), 'white')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@pytest.mark.asyncio
async def test_session_text_file_line_reading(tmp_path):
    """上传 txt/log 后可以按行读取、搜索。"""
    from nonebot_plugin_advisor.config import Config
    from nonebot_plugin_advisor.session_store import Conversation

    conv = Conversation(key='k', user_name='u', attach_dir=tmp_path / 'a', cfg=Config())
    lines = [f'line{i:03d} hello' for i in range(1, 101)]
    data = ('\n'.join(lines) + '\n').encode('utf-8')
    vf = await conv.register_media('error.log', data, None)
    assert vf.kind == 'text'
    assert vf.total_lines == 100

    out = conv.describe_text_file('error.log', 1, 3)
    assert 'error.log' in out
    assert '共 100 行' in out
    assert 'line001' in out

    tail = conv.describe_text_file('error.log', 99, 100)
    assert 'line100' in tail

    hits = conv.search_in_file('error.log', 'line050')
    assert 'line050 hello' in hits
    miss = conv.search_in_file('error.log', 'not-exist-word')
    assert '没有找到' in miss


@pytest.mark.asyncio
async def test_session_image_and_marker(tmp_path):
    """图片附件进入会话历史时带有附图标记，方便纯文本模型感知。"""
    from nonebot_plugin_advisor.config import Config
    from nonebot_plugin_advisor.session_store import Conversation

    conv = Conversation(
        key='k2', user_name='u', attach_dir=tmp_path / 'b', cfg=Config()
    )
    vf = await conv.register_media('shot.png', _png_bytes(200, 100), None)
    assert vf.kind == 'image'
    assert vf.width == 200
    assert vf.height == 100

    conv.add_user_turn('看看这张图', [vf])
    turn = conv.turns[-1]
    msg = turn.to_openai(inline_images=False, max_images=0)
    assert '附图 shot.png' in msg['content']
    assert msg['role'] == 'user'


def test_imageops_annotate(tmp_path):
    """标注：画矩形+英文文字成功；超过 10 字文字报错。"""
    from nonebot_plugin_advisor.imageops import annotate_image

    src = tmp_path / 'src.png'
    with open(src, 'wb') as f:
        f.write(_png_bytes(400, 300))

    dst_dir = tmp_path / 'out'
    out = annotate_image(
        src,
        [
            {'op': 'rect', 'x': 100, 'y': 100, 'w': 200, 'h': 80, 'color': '#ff0000'},
            {'op': 'text', 'x': 120, 'y': 120, 'text': 'OK', 'color': 'red'},
        ],
        dst_dir,
    )
    assert out.exists()

    with pytest.raises(ValueError, match='最多 10 个字'):
        annotate_image(
            src,
            [{'op': 'text', 'x': 10, 'y': 10, 'text': '这是一段超过十个字的长标注'}],
            dst_dir,
        )


def test_prompt_contains_persona(tmp_path):
    """系统提示词包含昵称与产品信息。"""
    from nonebot_plugin_advisor.config import Config
    from nonebot_plugin_advisor.prompts import build_system_prompt

    cfg = Config()
    cfg.advisor_nickname = '小顾'
    cfg.advisor_product_name = 'NoneBot'
    cfg.advisor_kb_description = 'NoneBot 官方文档'
    text = build_system_prompt(
        cfg,
        user_name='张三',
        platform_desc='QQ 群',
        kb_line='知识库：NoneBot 文档',
        tool_names=['read_file'],
    )
    assert '小顾' in text
    assert 'NoneBot' in text
    assert '张三' in text
