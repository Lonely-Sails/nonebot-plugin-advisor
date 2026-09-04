"""联网搜索：SearXNG JSON API 优先，Bing（curl_cffi 模拟浏览器）兜底；外加网页正文抓取。"""  # noqa: E501

from __future__ import annotations

import re
import html
import random
import asyncio
from urllib.parse import urlencode

import httpx
from nonebot import logger
from curl_cffi.requests import AsyncSession

from .utils import truncate, strip_html

_USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'
)

# Bing 搜索用的浏览器 UA 池（随机挑选，降低被识别为爬虫的概率）
_BING_UAS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
]


class SearchError(RuntimeError):
    """搜索失败"""


async def _searxng_search(query: str, base_url: str, max_results: int) -> list[dict]:
    url = base_url.rstrip('/') + '/search?' + urlencode({'q': query, 'format': 'json'})
    headers = {'User-Agent': _USER_AGENT, 'Accept': 'application/json'}
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


# 用正则解析 Bing 结果页（避免引入 bs4 依赖）
_BING_ALGO_RE = re.compile(r'<li class="b_algo"[^>]*>(.*?)</li>', re.S)
_BING_TITLE_RE = re.compile(r'<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>(.*?)</a></h2>', re.S)
_BING_SNIPPET_RE = re.compile(r'<p class="b_lineclamp[^"]*"[^>]*>(.*?)</p>', re.S)
_TAG_RE = re.compile(r'<[^>]+>')


def _strip_tags(raw: str) -> str:
    """去掉 HTML 标签并反转义实体。"""
    return html.unescape(_TAG_RE.sub('', raw)).strip()


async def _bing_search(query: str, max_results: int) -> list[dict]:
    """用 curl_cffi 异步模拟浏览器抓取 Bing 搜索结果。"""
    headers = {
        'User-Agent': random.choice(_BING_UAS),
        'Accept': (
            'text/html,application/xhtml+xml,application/xml;q=0.9,'
            'image/avif,image/webp,*/*;q=0.8'
        ),
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.bing.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    async with AsyncSession() as session:
        resp = await session.get(
            'https://www.bing.com/search',
            params={'q': query},
            headers=headers,
            impersonate='chrome120',
            timeout=15,
        )
        resp.raise_for_status()
        page = resp.text

    results: list[dict] = []
    for m in _BING_ALGO_RE.finditer(page):
        if len(results) >= max_results:
            break
        block = m.group(1)
        tm = _BING_TITLE_RE.search(block)
        if not tm:
            continue
        href = tm.group(1)
        title = _strip_tags(tm.group(2))
        sm = _BING_SNIPPET_RE.search(block)
        snippet = _strip_tags(sm.group(1)) if sm else ''
        results.append({'title': title, 'url': href, 'snippet': snippet})
    return results


async def _fetch_contents(
    results: list[dict], max_chars: int, timeout_: float, concurrency: int = 3
) -> list[dict]:
    """并发抓取每个搜索结果的正文，追加到 content 字段。单个失败不中断。"""
    sem = asyncio.Semaphore(concurrency)

    async def _one(r: dict) -> dict:
        async with sem:
            try:
                content = await fetch_webpage(
                    r['url'], max_chars=max_chars, timeout_=timeout_
                )
                r['content'] = content
            except Exception as e:
                logger.debug(
                    f'Failed to fetch search result content {r.get("url")}: {e}'
                )
                r['content'] = ''
        return r

    return await asyncio.gather(*(_one(r) for r in results))


async def web_search(
    query: str,
    *,
    searxng_url: str = '',
    max_results: int = 5,
    timeout_: float = 25.0,
    fetch_content: bool = False,
    content_max_chars: int = 6000,
) -> list[dict]:
    """执行联网搜索，返回 [{title,url,snippet,content?}]。失败抛 SearchError。

    fetch_content=True 时并发抓取每个结果的正文到 content 字段。
    """
    errors: list[str] = []
    try:
        if searxng_url.strip():
            results = await asyncio.wait_for(
                _searxng_search(query, searxng_url, max_results), timeout=timeout_
            )
            logger.debug(
                f'Web search {query!r} returned {len(results)} result(s) (searxng)'
            )
            if fetch_content and results:
                results = await _fetch_contents(
                    results, content_max_chars, min(timeout_, 15.0)
                )
            return results
    except Exception as e:
        errors.append(f'searxng: {e}')
    try:
        results = await asyncio.wait_for(
            _bing_search(query, max_results), timeout=timeout_
        )
        logger.debug(f'Web search {query!r} returned {len(results)} result(s) (bing)')
        if fetch_content and results:
            results = await _fetch_contents(
                results, content_max_chars, min(timeout_, 15.0)
            )
        return results
    except Exception as e:
        errors.append(f'bing: {e}')
    logger.warning(f'Web search failed {query!r}: {errors}')
    raise SearchError('搜索失败：' + ('；'.join(errors) or '未知原因'))


async def fetch_webpage(url: str, max_chars: int = 6000, timeout_: float = 20.0) -> str:
    """抓取网页并转纯文本。失败抛 SearchError。"""
    try:
        # 先看是不是可直接下载的文本/图片类资源
        ext = url.split('?')[0].rsplit('.', 1)[-1].lower()
        if ext in {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf', 'zip', 'exe', 'bin'}:
            raise SearchError(f'该链接是 {ext} 文件，无法直接阅读其文字内容')
        headers = {
            'User-Agent': random.choice(_BING_UAS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        async with AsyncSession() as session:
            resp = await session.get(
                url,
                headers=headers,
                impersonate='chrome120',
                timeout=timeout_,
            )
            resp.raise_for_status()
            raw = resp.text
        text = strip_html(raw) if '<' in raw else raw
        result = truncate(text, max_chars) if text else '（页面没有可读文本）'
        logger.debug(
            f'Fetched webpage {url}: body {len(text)} chars -> '
            f'returned {len(result)} chars'
        )
        return result
    except SearchError:
        raise
    except Exception as e:
        raise SearchError(f'抓取网页失败：{type(e).__name__}: {e}') from e
