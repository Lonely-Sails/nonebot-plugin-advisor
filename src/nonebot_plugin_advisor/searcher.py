"""联网搜索：SearXNG JSON API 优先，DuckDuckGo HTML 兜底；外加网页正文抓取。"""

from __future__ import annotations

import re
import asyncio
from urllib.parse import urlencode

import httpx

from .utils import truncate, strip_html, http_get_text

_UA = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'
)


class SearchError(RuntimeError):
    """搜索失败"""


async def _searxng_search(query: str, base_url: str, max_results: int) -> list[dict]:
    url = base_url.rstrip('/') + '/search?' + urlencode({'q': query, 'format': 'json'})
    headers = {'User-Agent': _UA, 'Accept': 'application/json'}
    async with httpx.AsyncClient(
        timeout=15.0, follow_redirects=True, headers=headers
    ) as c:
        resp = await c.get(url)
        resp.raise_for_status()
        data = resp.json()
    results = []
    for item in data.get('results', []):
        if len(results) >= max_results:
            break
        title = item.get('title') or item.get('url') or ''
        snippet = item.get('content') or item.get('snippet') or ''
        results.append({'title': title, 'url': item.get('url', ''), 'snippet': snippet})
    return results


# DuckDuckGo html 结果结构（尽力而为的解析）
_DDG_ITEM_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.I | re.S,
)
_DDG_NEW_RE = re.compile(
    r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.I | re.S,
)
_TAG_RE = re.compile(r'<[^>]+>')


def _ddg_href(raw: str) -> str:
    # DuckDuckGo 的跳转链接，取出真实 url
    m = re.search(r'uddg=([^&]+)', raw)
    if m:
        from urllib.parse import unquote

        return unquote(m.group(1))
    return raw


async def _duckduckgo_search(query: str, max_results: int) -> list[dict]:
    url = 'https://html.duckduckgo.com/html/?' + urlencode({'q': query})
    html_text = await http_get_text(url, timeout=15.0)
    results: list[dict] = []
    for regex in (_DDG_NEW_RE, _DDG_ITEM_RE):
        for m in regex.finditer(html_text):
            if len(results) >= max_results:
                break
            url_ = _ddg_href(m.group(1))
            title = _TAG_RE.sub('', m.group(2))
            snippet = _TAG_RE.sub('', m.group(3))
            results.append({'title': title, 'url': url_, 'snippet': snippet})
        if results:
            break
    return results


async def web_search(
    query: str,
    *,
    searxng_url: str = '',
    max_results: int = 5,
    timeout_: float = 25.0,
) -> list[dict]:
    """执行联网搜索，返回 [{title,url,snippet}]。失败抛 SearchError。"""
    errors: list[str] = []
    try:
        if searxng_url.strip():
            async with asyncio.timeout(timeout_):
                return await _searxng_search(query, searxng_url, max_results)
    except Exception as e:
        errors.append(f'searxng: {e}')
    try:
        async with asyncio.timeout(timeout_):
            return await _duckduckgo_search(query, max_results)
    except Exception as e:
        errors.append(f'duckduckgo: {e}')
    raise SearchError('搜索失败：' + ('；'.join(errors) or '未知原因'))


async def fetch_webpage(url: str, max_chars: int = 6000, timeout_: float = 20.0) -> str:
    """抓取网页并转纯文本。失败抛 SearchError。"""
    try:
        # 先看是不是可直接下载的文本/图片类资源
        ext = url.split('?')[0].rsplit('.', 1)[-1].lower()
        if ext in {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf', 'zip', 'exe', 'bin'}:
            raise SearchError(f'该链接是 {ext} 文件，无法直接阅读其文字内容')
        raw = await http_get_text(url, timeout_=timeout_)
        text = strip_html(raw) if '<' in raw else raw
        return truncate(text, max_chars) if text else '（页面没有可读文本）'
    except SearchError:
        raise
    except Exception as e:
        raise SearchError(f'抓取网页失败：{type(e).__name__}: {e}') from e
