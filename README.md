<div align="center">
    <a href="https://v2.nonebot.dev/store">
    <img src="https://raw.githubusercontent.com/fllesser/nonebot-plugin-template/refs/heads/resource/.docs/NoneBotPlugin.svg" width="310" alt="logo"></a>

# NoneBot-Plugin-Advisor
</div>

基于 **OpenAI 兼容格式**的跨平台 LLM 客服。让机器人像一个**真人客服**一样为你的项目提供支持：读文档、查日志、看截图、在图上圈注并回发、必要时联网搜索。

- 跨平台：基于 `nonebot-plugin-alconna` + `nonebot-plugin-uninfo`
- 定时从文档仓库（git）拉取 md / 图片，并用**视觉模型**给每张图片建立描述索引
- @ 机器人提问 → agent 检索文档回答，口吻自然、短句、不甩长文、不输出 Markdown，引导用户一步步排查
- 用户发来的**图片 / txt / log** 会自动进入会话的“内存文件系统”，agent 可用工具**按行读取、搜索关键词**
- **引用消息**里的图片/文件也会进入会话
- 提供**图片标注工具**：画矩形/椭圆/箭头/高亮、写不超过 10 个字的文字，标注后随回复回发给用户
- 可联网搜索（SearXNG / Bing 兜底）

## 📦 安装

```bash
nb plugin install nonebot-plugin-advisor
```

或在项目里安装后，在 `pyproject.toml` 的 `[tool.nonebot]` 中加入：

```toml
plugins = ["nonebot_plugin_advisor"]
```

> 依赖：`nonebot2 >= 2.5`、`nonebot-plugin-alconna`、`nonebot-plugin-uninfo`、`nonebot-plugin-localstore`、`nonebot-plugin-apscheduler`、`openai >= 3.8`、`pillow`、`httpx`、`curl-cffi`

## ⚙️ 配置

在 `.env` / `.env.prod` 中配置（环境变量前缀 `ADVISOR_`）：

```dotenv
# ── LLM（OpenAI 兼容） ──
ADVISOR_LLM_BASE_URL=https://api.openai.com/v1
ADVISOR_LLM_API_KEY=sk-xxxx
ADVISOR_LLM_MODEL=gpt-4o-mini
# 视觉模型（图片描述/索引用）。留空则尝试使用主模型
ADVISOR_VISION_MODEL=

# ── 人设 / 行为 ──
ADVISOR_NICKNAME=小顾
ADVISOR_PRODUCT_NAME=NoneBot
ADVISOR_KB_DESCRIPTION=关于 NoneBot 的官方文档
ADVISOR_ACK_TEXT=收到~我帮您看下，稍等一下哦
ADVISOR_IMAGE_INLINE=false   # 是否把用户图片直接以多模态塞给主模型（主模型需支持看图）

# ── 联网搜索 ──
ADVISOR_ENABLE_WEB=true
# 自建 SearXNG（推荐，无 key）；留空自动退回 Bing
ADVISOR_SEARXNG_URL=

# ── 文档知识库：定时拉取仓库并给图片做视觉索引 ──
ADVISOR_KB_SOURCE=            # git 仓库地址或本地目录；留空禁用
ADVISOR_KB_BRANCH=
ADVISOR_KB_SYNC_CRON=0 3 * * *   # cron；留空且 INTERVAL>0 时用间隔
ADVISOR_KB_SYNC_INTERVAL=0       # 分钟；都为 0 则仅启动时同步一次
ADVISOR_KB_MAX_IMAGES_PER_SYNC=50
```

### 完整配置项

| 配置项 | 必填 | 默认值 | 说明 |
| --- | :-: | --- | --- |
| `ADVISOR_LLM_BASE_URL` | 是 | `https://api.openai.com/v1` | OpenAI 兼容接口地址 |
| `ADVISOR_LLM_API_KEY` | 是 | 无 | 密钥（留空则客服不启用） |
| `ADVISOR_LLM_MODEL` | 否 | `gpt-4o-mini` | 客服主模型 |
| `ADVISOR_VISION_MODEL` | 否 | 空 | 视觉模型（描述/索引图片） |
| `ADVISOR_LLM_TEMPERATURE` | 否 | `0.3` | 温度 |
| `ADVISOR_LLM_MAX_TOKENS` | 否 | `1500` | 单次回答上限 |
| `ADVISOR_NICKNAME` | 否 | `小顾` | 客服昵称 |
| `ADVISOR_PRODUCT_NAME` | 否 | 空 | 服务的产品名 |
| `ADVISOR_KB_DESCRIPTION` | 否 | 空 | 知识库范围的一句话描述 |
| `ADVISOR_REPLY_TRIGGER` | 否 | `true` | 群聊回复机器人也算对话 |
| `ADVISOR_MENTION_ONLY` | 否 | `true` | 群聊必须 @ 才应答 |
| `ADVISOR_ACK_TEXT` | 否 | `收到~…` | 先回复的提示语；留空关闭 |
| `ADVISOR_IMAGE_INLINE` | 否 | `false` | 图片以多模态直接喂给主模型 |
| `ADVISOR_HISTORY_MAX_TURNS` | 否 | `12` | 记忆的最大轮数 |
| `ADVISOR_CONVERSATION_TTL` | 否 | `3600` | 会话空闲多久遗忘（秒） |
| `ADVISOR_UPLOAD_MAX_BYTES` | 否 | `30MB` | 单个附件大小上限 |
| `ADVISOR_UPLOAD_MAX_LINES` | 否 | `10000` | 文本文件最多行数（可按行读） |
| `ADVISOR_ENABLE_KNOWLEDGE` | 否 | `true` | 文档工具开关 |
| `ADVISOR_ENABLE_FILES` | 否 | `true` | 文件读取工具开关 |
| `ADVISOR_ENABLE_IMAGE_TOOLS` | 否 | `true` | 看图/标注工具开关 |
| `ADVISOR_ENABLE_WEB` | 否 | `true` | 联网工具开关 |
| `ADVISOR_AGENT_MAX_ROUNDS` | 否 | `12` | 单次最多调用工具次数 |
| `ADVISOR_SEARXNG_URL` | 否 | 空 | SearXNG 地址；空则 Bing |
| `ADVISOR_SEARCH_MAX_RESULTS` | 否 | `5` | 每次搜索条数 |
| `ADVISOR_KB_SOURCE` | 否 | 空 | 文档仓库 git url / 本地目录 |
| `ADVISOR_KB_BRANCH` | 否 | 空 | 分支（git 时有效） |
| `ADVISOR_KB_SYNC_CRON` | 否 | 空 | 定时同步 cron |
| `ADVISOR_KB_SYNC_INTERVAL` | 否 | `0` | 定时同步间隔（分钟） |
| `ADVISOR_KB_MAX_IMAGES_PER_SYNC` | 否 | `50` | 每次最多索引图片数 |
| `ADVISOR_KB_REBUILD` | 否 | `false` | 强制全量重建索引 |

## 🎉 使用

### 对话

- **群聊**：`@机器人 <问题>`；支持直接发截图 / txt / log / 引用带图消息。
- **私聊**：直接发消息即可。
- 发送后机器人会先回一句“收到”，随后以客服口吻给出**短回复**并引导你一步步排查；必要时会在你发的图上**画框标注**并回发给你，配一句“照着红框这里点一下”。

### 指令

| 指令 | 权限 | 说明 |
| --- | --- | --- |
| `客服重置` / `advisor reset` | 所有人 | 清空当前会话记忆 |
| `客服帮助` / `advisor help` | 所有人 | 查看使用说明 |
| `客服状态` / `advisor status` | 所有人 | 查看 LLM / 工具 / 知识库状态 |
| `客服同步` / `advisor sync` | 超级用户 | 立即拉取并重建文档仓库 + 图片索引 |

### 知识库

配置 `ADVISOR_KB_SOURCE` 后：

- 插件启动时会先同步一次（后台进行，不阻塞启动）；
- 按 `ADVISOR_KB_SYNC_CRON` 或 `ADVISOR_KB_SYNC_INTERVAL` 定时更新；
- 每次同步会拉取 git 仓库，收集全部 `md/mdx` 文档与图片；**新增/变更的图片**会调用视觉模型生成描述，写入本地索引（`localstore` 数据目录下的 `knowledge/kb_index.json`）。

agent 回答时会通过 `search_kb` / `read_kb_doc` 检索文档，并可用 `send_kb_image` 把文档里的截图发给用户，或先标注再发送。

> 提示：知识库内容缓存于本地数据目录，可随时用 `客服同步` 手动刷新；机器人重启后索引仍在（文档/图片索引持久化），只有**聊天记忆**与**上传附件**是内存态、重启即清空。

## 🧠 工具一览

| 工具 | 作用 |
| --- | --- |
| `search_kb` / `read_kb_doc` / `list_kb_docs` | 检索、按行阅读知识库文档 |
| `list_kb_images` / `describe_kb_image` / `send_kb_image` | 查看/发送文档插图 |
| `list_files` / `read_file` / `search_file` | 查看用户上传的文件、按行读、搜关键词 |
| `inspect_image` | 查看用户图片（尺寸 + 视觉描述） |
| `annotate_image` / `send_image` | 在图上画矩形/箭头/文字（≤10 字）并回发 |
| `web_search` / `open_url` | 联网搜索与抓取网页 |

## 📁 目录结构

```
src/nonebot_plugin_advisor/
├── __init__.py        # 插件入口：matcher、控制命令、定时调度
├── config.py          # 全部配置项
├── agent.py           # agent 工具调用主循环
├── llm.py             # OpenAI 兼容客户端（对话/视觉）
├── toolkit.py         # 工具集（文档/文件/图片/搜索）
├── docs_store.py      # 知识库：git 拉取 + md/图片索引 + 检索
├── session_store.py   # 会话内存虚拟文件系统 + 对话历史
├── media.py           # 从 UniMessage 抽取文本/图片/文件/引用
├── imageops.py        # 图片标注（Pillow，归一化坐标）
├── searcher.py        # 联网搜索
├── prompts.py         # 客服人设提示词
└── utils.py           # 通用工具
```

## 🧪 开发 / 测试

```bash
uv sync
uv run pytest        # 跑测试
uv run ruff check .  # lint
uv run ruff format . # 格式化
```

## 📄 协议

MIT
