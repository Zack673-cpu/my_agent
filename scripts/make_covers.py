"""一次性工具：用阿里百炼 Qwen-Image-2.0-Pro 生成绘本封面与封底插画。

生成的两张图是一组配套的水彩风格插画，保存到 static/assets/：
- cover.png：封面（星夜森林里的小狐狸仰望萤火虫）
- back_cover.png：封底（同一森林，小狐狸提灯远去的背影）

用法：py -3 scripts/make_covers.py
"""

from __future__ import annotations

import shutil
import sys
from http import HTTPStatus
from pathlib import Path

# 保证从项目根目录能 import src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from dashscope import MultiModalConversation

from src.image_tool import (
    DASHSCOPE_MODEL,
    _attr,
    _extract_dashscope_urls,
    _get_dashscope_key,
    _save_url,
)

load_dotenv()

ASSETS_DIR = Path(__file__).resolve().parent.parent / "static" / "assets"

COVER_PROMPT = (
    "儿童绘本封面插画，柔和梦幻的水彩风格。星空下的暮色森林空地，一只可爱的小橘狐狸"
    "坐在长满青苔的小土丘上，仰望着漫天飞舞的发光萤火虫和一弯新月，暖色灯笼光晕，"
    "画面以柔和的蓝色、青色为主，点缀温暖的琥珀色，纸张纹理质感，画面上方三分之一"
    "留出干净的天空区域用于放置书名，竖版书籍封面构图。整幅画面不出现任何文字、"
    "字母、水印或标志。"
)

BACK_PROMPT = (
    "儿童绘本内页整幅插画，与封面同一组的柔和梦幻水彩风格。同一片星空下的暮色森林，"
    "同一只可爱的小橘狐狸提着一盏发光的小灯笼，只留一个温暖的小小背影，正走向远处"
    "的树林深处，头顶有几只发光萤火虫相伴，画面比封面更简洁安静、留白更多，柔和的"
    "蓝色青色配温暖琥珀色，纸张纹理质感，竖版构图。重要：画面必须铺满整个画幅、"
    "四边到边，不要画成一本书或书壳样机，不要任何边框、白边、内框或立体书本，"
    "不出现任何文字、字母、水印或标志。"
)


def generate(prompt: str, save_as: str) -> None:
    """生成一张插画并保存到 static/assets/。"""
    response = MultiModalConversation.call(
        api_key=_get_dashscope_key(),
        model=DASHSCOPE_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        stream=False,
        n=1,
        watermark=False,
        prompt_extend=True,
        size="1024*1280",
    )
    if _attr(response, "status_code") != HTTPStatus.OK:
        raise RuntimeError(
            f"生成失败：{_attr(response, 'code')} {_attr(response, 'message')}"
        )
    urls = _extract_dashscope_urls(response)
    if not urls:
        raise RuntimeError("模型没有返回图片。")
    local = _save_url(urls[0])
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    target = ASSETS_DIR / save_as
    shutil.copyfile(local, target)
    print(f"已保存：{target}")


if __name__ == "__main__":
    generate(COVER_PROMPT, "cover.png")
    generate(BACK_PROMPT, "back_cover.png")
    print("封面与封底生成完成。")
