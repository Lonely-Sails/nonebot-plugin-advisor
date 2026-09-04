from nonebot import get_driver, get_plugin_config
from pydantic import BaseModel

# 文档/图片扩展名
DOC_EXTS: tuple[str, ...] = ('.md', '.mdx', '.markdown', '.rst')
IMAGE_EXTS: tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')
# 当作文本读取的文件扩展名
TEXT_EXTS: tuple[str, ...] = (
    '.txt',
    '.log',
    '.md',
    '.mdx',
    '.json',
    '.yml',
    '.yaml',
    '.toml',
    '.ini',
    '.cfg',
    '.conf',
    '.py',
    '.js',
    '.ts',
    '.tsx',
    '.java',
    '.go',
    '.rs',
    '.c',
    '.h',
    '.cpp',
    '.hpp',
    '.sh',
    '.ps1',
    '.bat',
    '.csv',
    '.xml',
    '.html',
    '.css',
    '.sql',
    '.env',
    '.out',
    '.err',
    '.diff',
    '.patch',
)


class Config(BaseModel):
    """插件配置项（环境变量前缀：ADVISOR_）"""

    model_config = {'extra': 'ignore'}

    # ── LLM ─────────────────────────────────────────────────────────────
    advisor_llm_base_url: str = 'https://api.openai.com/v1'
    """OpenAI 兼容 API 地址"""
    advisor_llm_api_key: str = ''
    """API 密钥（留空则插件禁用）"""
    advisor_llm_model: str = 'gpt-4o-mini'
    """客服使用的主模型"""
    advisor_vision_model: str | None = None
    """视觉模型（用于描述图片）。留空时尝试使用主模型"""
    advisor_llm_temperature: float = 0.3
    """采样温度，越低越稳定"""
    advisor_llm_max_tokens: int = 1500
    """单次回答最大 token"""
    advisor_llm_timeout: float = 180.0
    """单次请求超时（秒）"""
    advisor_request_max_retries: int = 2
    """请求失败重试次数"""

    # ── 人设 / 触发 ─────────────────────────────────────────────────────
    advisor_nickname: str = '小顾'
    """客服昵称"""
    advisor_company_name: str = '技术支持'
    """客服部门名称"""
    advisor_product_name: str = ''
    """服务的产品名称（用于提示词，如 NoneBot2）"""
    advisor_kb_description: str = ''
    """知识库文档范围的一句话描述，如“关于 XXX 的官方文档”"""
    advisor_reply_trigger: bool = True
    """群聊中“回复机器人消息”也算作与客服对话"""
    advisor_mention_only: bool = True
    """群聊中是否必须 @ 机器人才会应答（避免打扰）"""
    advisor_ack_text: str = '收到~我帮您看下，稍等一下哦'
    """处理前先发送的提示语，留空则不发"""
    advisor_cannot_answer_text: str = (
        '抱歉，这个问题超出我的能力范围啦。建议把完整报错/截图发给我，'
        '或联系人工客服进一步排查～'
    )
    """出错时的兜底回复，留空则不发"""
    advisor_image_inline: bool = False
    """是否把用户发的图片以多模态形式直接塞给主模型（需要主模型支持看图）"""
    advisor_split_message: bool = False
    """是否把回答拆成多条消息逐条发送（更像真人客服）"""
    advisor_split_max_length: int = 50
    """分段发送时每段的最大字符数"""
    advisor_split_interval: float = 0.5
    """分段发送时每段之间的间隔（秒）"""
    advisor_history_max_turns: int = 12
    """每个会话保留的最大对话轮数（用户+客服各算一轮）"""
    advisor_conversation_ttl: int = 3600
    """会话空闲多久（秒）后自动遗忘，默认 1 小时"""

    # ── 附件 ────────────────────────────────────────────────────────────
    advisor_upload_max_bytes: int = 30 * 1024 * 1024
    """单个上传文件大小上限（默认 30MB）"""
    advisor_upload_max_lines: int = 10000
    """文本类文件最多读取/载入多少行"""
    advisor_upload_text_exts: tuple[str, ...] = TEXT_EXTS
    """识别为“文本类文件”的扩展名"""

    # ── 工具开关 ────────────────────────────────────────────────────────
    advisor_enable_knowledge: bool = True
    """是否启用文档知识库工具"""
    advisor_enable_files: bool = True
    """是否启用“文件系统”类工具（读取用户上传的文件）"""
    advisor_enable_image_tools: bool = True
    """是否启用图片查看/标注工具"""
    advisor_enable_web: bool = True
    """是否启用联网搜索工具"""
    advisor_agent_max_rounds: int = 12
    """agent 单次最多调用工具的次数（防死循环）"""

    # ── 联网搜索 ────────────────────────────────────────────────────────
    advisor_searxng_url: str = ''
    """自建 SearXNG 实例地址；留空则用 Bing 兜底"""
    advisor_search_max_results: int = 5
    """每次搜索返回的条数"""
    advisor_fetch_max_chars: int = 6000
    """打开网页时最多读取多少字符"""

    # ── 知识库（定时从仓库拉取 md + 图片，用视觉模型索引） ────────────────
    advisor_kb_source: str = ''
    """文档仓库地址（git url）或本地目录。留空则禁用知识库同步"""
    advisor_kb_branch: str = ''
    """文档仓库分支，留空用远端默认分支"""
    advisor_kb_sync_interval: int = 0
    """定时同步间隔（分钟）。0 表示仅启动时同步一次；配合 cron 使用"""
    advisor_kb_sync_cron: str = ''
    """定时同步的 cron 表达式（如 '0 3 * * *' 每天三点）。优先于 interval"""
    advisor_kb_max_images_per_sync: int = 50
    """每次同步最多索引多少张新图片（防止一次性调用过多视觉模型）"""
    advisor_kb_rebuild: bool = False
    """每次同步是否强制重建全部索引（图片会全部重新描述）"""
    advisor_kb_doc_exts: tuple[str, ...] = DOC_EXTS
    """知识库收录的文档扩展名"""
    advisor_kb_image_exts: tuple[str, ...] = IMAGE_EXTS
    """知识库收录的图片扩展名"""

    # ── 其他 ────────────────────────────────────────────────────────────
    advisor_debug: bool = False
    """打印更详细的调试日志"""


# 配置加载
plugin_config: Config = get_plugin_config(Config)
global_config = get_driver().config

# 全局名称（机器人在各平台的昵称，用于过滤“自己发的消息”）
NICKNAME: str = next(iter(global_config.nickname), '') if global_config.nickname else ''

# 超级用户集合（用于权限判断，格式与平台有关，如 onebot 的 QQ 号）
SUPERUSERS: set[str] = {str(x) for x in global_config.superusers}


def is_superuser(user_id: str) -> bool:
    """判断是否为超级用户"""
    return user_id in SUPERUSERS
