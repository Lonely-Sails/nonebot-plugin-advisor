"""知识库单测：对本地目录做扫描/索引/检索（不依赖 LLM 与 git）。"""

from __future__ import annotations

import io

import pytest


def _png_bytes() -> bytes:
    from PIL import Image

    img = Image.new('RGB', (80, 60), 'lightblue')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _make_repo(root):
    root.mkdir(parents=True, exist_ok=True)
    docs = root / 'docs'
    docs.mkdir(exist_ok=True)
    md = docs / 'quickstart.md'
    md.write_text(
        '\n'.join(
            [
                '# 快速开始',
                '',
                '第一步：安装本项目，使用 `pip install xxx`。',
                '第二步：在配置里填入 token。',
                '第三步：启动并 @机器人提问。',
                '常见问题：连接超时请检查网络与端口。',
                '更多细节见进阶章节。',
            ]
        ),
        encoding='utf-8',
    )
    (docs / 'pic.png').write_bytes(_png_bytes())
    # 隐藏目录应被忽略
    (root / '.git').mkdir(exist_ok=True)
    (root / '.git' / 'ignored.md').write_text('ignored', encoding='utf-8')
    return md


@pytest.mark.asyncio
async def test_kb_scan_local_and_search(tmp_path):
    from nonebot_plugin_advisor.config import Config
    from nonebot_plugin_advisor.docs_store import KnowledgeBase

    repo = tmp_path / 'repo-src'
    _make_repo(repo)
    cfg = Config()
    cfg.advisor_kb_source = str(repo)

    data_dir = tmp_path / 'data'
    kb = KnowledgeBase(cfg, data_dir=data_dir, llm=None)
    report = await kb.sync(force=True)
    assert report.ok
    assert report.docs == 1
    assert report.images == 1

    # 文档列表 / 内容检索 / 行读取
    names = [d['path'] for d in kb.list_docs()]
    assert any(n.endswith('quickstart.md') for n in names)

    hits = kb.search_docs('连接超时')
    assert hits
    assert 'quickstart.md' in hits[0]['path']

    text = kb.read_doc('docs/quickstart.md', 1, 4)
    assert text
    assert 'pip install' in text

    # 无 LLM 时图片索引会标记错误但不会崩溃
    assert kb.image_description('docs/pic.png') is not None

    # 路径穿越防护
    assert kb._safe_resolve('../../etc/passwd') is None
    assert kb.image_path('../x.png') is None


@pytest.mark.asyncio
async def test_kb_disabled(tmp_path):
    from nonebot_plugin_advisor.config import Config
    from nonebot_plugin_advisor.docs_store import KnowledgeBase

    kb = KnowledgeBase(Config(), data_dir=tmp_path / 'data', llm=None)
    report = await kb.sync()
    assert report.ok
    assert kb.search_docs('anything') == []
