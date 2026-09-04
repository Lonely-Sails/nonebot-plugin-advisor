"""图片查看与标注工具（基于 Pillow）。

坐标系统：所有 x/y/w/h 使用 **0~1000 归一化坐标**（横向 x、纵向 y、向下为正），
这样与图片实际分辨率无关，方便模型标注。底层会自动按图片尺寸换算。

支持的操作（op）：
- rect      绘制空心矩形
- ellipse   绘制空心椭圆
- fill      绘制半透明填充矩形（高亮/遮盖）
- arrow     绘制箭头线（从点 A 指向点 B）
- text      写文字（限制 <=10 个字，避免标注过大遮挡内容）
"""

from __future__ import annotations

import io
from typing import Any
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MAX_LABEL_LEN = 10  # 文字标注上限
_COORD = 1000  # 归一化坐标系


def image_info(path: str | Path) -> dict[str, Any]:
    """返回图片宽高与格式。"""
    with Image.open(path) as im:
        w, h = im.size
        return {
            'width': w,
            'height': h,
            'format': (im.format or '').lower() or Path(path).suffix.lstrip('.'),
            'mode': im.mode,
        }


def _cjk_font_path() -> str | None:
    """尽力找一套支持中文的字体。"""
    candidates = [
        # macOS
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        # Linux
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        # Windows
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/msyh.ttf',
        'C:/Windows/Fonts/simhei.ttf',
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _parse_color(
    color: str | None, default: tuple[int, int, int, int] = (255, 0, 0, 255)
) -> tuple[int, int, int, int]:
    """解析 #RRGGBB / #RRGGBBAA 或 r,g,b / r,g,b,a。"""
    color = str(color or '').strip()
    if not color:
        return default
    if color.startswith('#'):
        c = color[1:]
        if len(c) == 3:
            c = ''.join(ch * 2 for ch in c) + 'ff'
        elif len(c) == 6:
            c += 'ff'
        elif len(c) == 8:
            pass
        else:
            return default
        try:
            return (
                int(c[0:2], 16),
                int(c[2:4], 16),
                int(c[4:6], 16),
                int(c[6:8], 16),
            )
        except ValueError:
            return default
    parts = color.replace(';', ',').split(',')
    if len(parts) >= 3:
        try:
            r, g, b = (int(p) for p in parts[:3])
            a = int(parts[3]) if len(parts) > 3 else 255
            return (r, g, b, a)
        except ValueError:
            return default
    # 常见颜色名
    names = {
        'red': (255, 0, 0, 255),
        'green': (0, 180, 0, 255),
        'blue': (0, 120, 255, 255),
        'yellow': (255, 200, 0, 255),
        'orange': (255, 130, 0, 255),
        'white': (255, 255, 255, 255),
        'black': (0, 0, 0, 255),
        'cyan': (0, 200, 220, 255),
    }
    return names.get(color.lower(), default)


def _load_image(path: str | Path):
    im = Image.open(path)
    if im.mode not in ('RGB', 'RGBA', 'L'):
        im = im.convert('RGB')
    if im.mode == 'L':
        im = im.convert('RGB')
    return im


def _font(size: int, text: str):
    try:
        font_path = _cjk_font_path()
        if font_path:
            return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    # 回退：仅支持 ASCII
    if text.isascii():
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # 旧版 Pillow 不支持 size 参数
            return ImageFont.load_default()
    raise ValueError(
        '当前环境无中文字体，无法绘制中文标注。请改用纯英文/数字标注'
        '（如 OK、No.1），或先安装中文字体。'
    )


def _norm_to_px(value: float, dim: int) -> int:
    return max(0, min(round(float(value) / _COORD * dim), dim))


def annotate_image(
    src: str | Path,
    ops: list[dict[str, Any]] | None,
    dst_dir: str | Path,
    *,
    padding: int = 8,
) -> Path:
    """在图片副本上绘制标注，返回新文件路径。

    ops 为操作列表，坐标归一化到 0~1000。
    """
    if not ops:
        raise ValueError('没有要执行的标注操作')
    im = _load_image(src).convert('RGBA')
    w, h = im.size
    overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for op in ops:
        kind = str(op.get('op') or op.get('type') or '').lower()
        color = _parse_color(op.get('color'), (255, 0, 0, 255))
        if kind in ('rect', 'rectangle', 'box'):
            x = _norm_to_px(op.get('x', 0), w)
            y = _norm_to_px(op.get('y', 0), h)
            ww = max(1, _norm_to_px(op.get('w', 0), w))
            hh = max(1, _norm_to_px(op.get('h', 0), h))
            draw.rectangle(
                [x, y, x + ww, y + hh],
                outline=color,
                width=int(op.get('width', 4)) or 4,
            )
        elif kind in ('ellipse', 'circle', 'oval'):
            x = _norm_to_px(op.get('x', 0), w)
            y = _norm_to_px(op.get('y', 0), h)
            ww = max(1, _norm_to_px(op.get('w', 0), w))
            hh = max(1, _norm_to_px(op.get('h', 0), h))
            draw.ellipse(
                [x, y, x + ww, y + hh],
                outline=color,
                width=int(op.get('width', 4)) or 4,
            )
        elif kind in ('fill', 'highlight', 'mask'):
            x = _norm_to_px(op.get('x', 0), w)
            y = _norm_to_px(op.get('y', 0), h)
            ww = max(1, _norm_to_px(op.get('w', 0), w))
            hh = max(1, _norm_to_px(op.get('h', 0), h))
            alpha = int(op.get('alpha', 70))
            fill_color = (*color[:3], alpha)
            draw.rectangle([x, y, x + ww, y + hh], fill=fill_color)
        elif kind in ('arrow', 'line'):
            x1 = _norm_to_px(op.get('x1', op.get('x', 0)), w)
            y1 = _norm_to_px(op.get('y1', op.get('y', 0)), h)
            x2 = _norm_to_px(op.get('x2', op.get('x2', 0)), w)
            y2 = _norm_to_px(op.get('y2', op.get('y2', 0)), h)
            width = int(op.get('width', 5)) or 5
            draw.line([x1, y1, x2, y2], fill=color, width=width)
            if kind == 'arrow':
                draw.ellipse(
                    [x2 - width * 2, y2 - width * 2, x2 + width * 2, y2 + width * 2],
                    fill=color,
                )
        elif kind in ('text', 'label'):
            text = str(op.get('text') or '').strip()
            if not text:
                continue
            if len(text) > MAX_LABEL_LEN:
                raise ValueError(
                    f'文字标注最多 {MAX_LABEL_LEN} 个字，'
                    f'收到 {len(text)} 个字（{text!r}）。请缩短后再试。'
                )
            size = max(12, int(op.get('size', 42)))
            font = _font(size, text)
            x = _norm_to_px(op.get('x', 0), w)
            y = _norm_to_px(op.get('y', 0), h)
            # 文字若有背景更清晰
            if op.get('background'):
                bg = _parse_color(op.get('background'), (0, 0, 0, 160))
                bbox = draw.textbbox((x, y), text, font=font)
                draw.rectangle(
                    [
                        bbox[0] - padding,
                        bbox[1] - padding,
                        bbox[2] + padding,
                        bbox[3] + padding,
                    ],
                    fill=bg,
                )
            draw.text((x, y), text, font=font, fill=color)
        else:
            raise ValueError(f'不支持的标注操作：{kind}')

    im = Image.alpha_composite(im, overlay)
    # 输出
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    out_path = dst_dir / f'{stem}_annotated.png'
    out_path = im.convert('RGB').save(str(out_path), 'PNG') or out_path
    return out_path


def get_valid_ops() -> dict[str, str]:
    """供模型了解可用标注操作（写入工具说明）。"""
    return {
        'rect': '空心矩形：x,y,w,h,color,width。用于圈出按钮/输入框等',
        'fill': '半透明填充矩形（高亮/遮盖）：x,y,w,h,color,alpha',
        'ellipse': '空心椭圆：x,y,w,h,color,width',
        'arrow': '箭头：x1,y1,x2,y2,color,width（从起点指向终点）',
        'text': f'文字标注：x,y,text,size,color（text 不超过 {MAX_LABEL_LEN} 个字）',
    }


def compress_image(
    path: str | Path,
    *,
    quality: int = 85,
) -> bytes:
    """压缩图片为 JPEG 字节，供多模态内联传给模型。

    保持原尺寸，仅转 JPEG 压缩，显著减小 base64 体积。
    若原图已是 JPEG，则直接返回原字节。
    """
    src = Path(path)
    try:
        with Image.open(src) as im:
            fmt = (im.format or '').lower()
            # 已是 JPEG：直接返回原字节
            if fmt == 'jpeg':
                return src.read_bytes()
            # 统一转 RGB（JPEG 不支持透明通道）
            if im.mode in ('RGBA', 'LA', 'P'):
                im = im.convert('RGBA')
                bg = Image.new('RGB', im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode != 'RGB':
                im = im.convert('RGB')
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=quality, optimize=True)
            return buf.getvalue()
    except Exception:
        # 压缩失败时退回原字节，保证功能可用
        return src.read_bytes()
