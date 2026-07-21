"""图片编辑工具：双模型 provider 封装。

输入固定为「一组图片 + 提示词」，两个模型都执行图像编辑：

- 阿里百炼 DashScope：Qwen-Image-2.0-Pro（qwen-image-2.0-pro），生成+编辑
  融合模型，图片以 base64 data URI 放入 messages 内容。
- 硅基流动 SiliconFlow：Qwen/Qwen-Image-Edit-2509，图片放入 image/image2/
  image3 字段。

模型的选择与自动降级逻辑在 agent.py 中编排，本模块只负责各自的 API 调用。
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from http import HTTPStatus
from pathlib import Path

import requests
from dashscope import MultiModalConversation

# 阿里百炼 - 千问图像生成与编辑模型 ID
DASHSCOPE_MODEL = "qwen-image-2.0-pro"

# 硅基流动 - 千问图像编辑模型 ID
SILICONFLOW_MODEL = "Qwen/Qwen-Image-Edit-2509"

# 硅基流动图片编辑接口
SILICONFLOW_URL = "https://api.siliconflow.cn/v1/images/generations"

# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# 两个模型单次都最多支持 3 张输入图片
MAX_INPUT_IMAGES = 3

# 两个模型单次都最多输出 6 张
MAX_OUTPUT_IMAGES = 6

# 网络请求超时（秒）。图像生成耗时较长，给足余量。
REQUEST_TIMEOUT = 300


class ImageToolError(RuntimeError):
    """图片编辑过程中的可读错误。"""


def _get_dashscope_key() -> str:
    """读取阿里百炼 API Key。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ImageToolError(
            "未检测到 DASHSCOPE_API_KEY，请在 .env 中配置阿里百炼的 API Key。"
        )
    return api_key


def _get_siliconflow_key() -> str:
    """读取硅基流动 API Key。"""
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise ImageToolError(
            "未检测到 SILICONFLOW_API_KEY，请在 .env 中配置硅基流动的 API Key。"
        )
    return api_key


def _attr(obj, key):
    """兼容 dict 与对象两种访问方式，取出字段值。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _validate_inputs(image_paths: list[str], prompt: str) -> None:
    """校验通用输入：至少一张图片、非空提示词、图片数量上限。"""
    if not prompt or not prompt.strip():
        raise ImageToolError("提示词不能为空。")
    if not image_paths:
        raise ImageToolError("没有可用的输入图片，请先上传图片。")
    if len(image_paths) > MAX_INPUT_IMAGES:
        raise ImageToolError(
            f"输入图片过多（{len(image_paths)} 张），最多支持 {MAX_INPUT_IMAGES} 张。"
        )


def _encode_image(path: str) -> str:
    """把本地图片编码成 data URI（data:{mime};base64,{data}）。"""
    p = Path(path)
    if not p.is_file():
        raise ImageToolError(f"找不到图片文件：{path}")
    mime, _ = mimetypes.guess_type(p.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    try:
        data = base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        raise ImageToolError(f"无法读取图片 {path}：{exc}") from exc
    return f"data:{mime};base64,{data}"


def _save_url(url: str) -> str:
    """下载远程图片 URL 并保存到 outputs/，返回本地路径。

    两个模型返回的都是有时效的临时 URL（百炼 24 小时、硅基流动 1 小时），
    需要立即下载转存到本地。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise ImageToolError(f"下载生成图片失败：{exc}") from exc

    content_type = resp.headers.get("Content-Type", "image/png").split(";")[0].strip()
    ext = mimetypes.guess_extension(content_type) or ".png"
    filename = f"qwen_{int(time.time() * 1000)}_{os.urandom(3).hex()}{ext}"
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(resp.content)
    return str(out_path)


def _extract_dashscope_urls(response) -> list[str]:
    """从 DashScope 多模态响应中提取生成图片的 URL。"""
    urls: list[str] = []
    output = _attr(response, "output")
    choices = _attr(output, "choices") or []
    for choice in choices:
        message = _attr(choice, "message")
        content = _attr(message, "content") or []
        for item in content:
            image = _attr(item, "image")
            if image:
                urls.append(image)
    return urls


def edit_with_dashscope(
    image_paths: list[str],
    prompt: str,
    num_outputs: int = 1,
    size: str = "2048*2048",
) -> list[str]:
    """用阿里百炼 Qwen-Image-2.0-Pro 按提示词编辑图片。

    Args:
        image_paths: 输入图片本地路径（1-3 张）。
        prompt: 编辑指令。
        num_outputs: 期望输出的图片数量（1-6）。
        size: 输出分辨率“宽*高”，总像素需在 512*512 至 2048*2048 之间。

    Returns:
        生成图片的本地路径列表。
    """
    _validate_inputs(image_paths, prompt)
    num_outputs = max(1, min(MAX_OUTPUT_IMAGES, int(num_outputs)))
    api_key = _get_dashscope_key()

    content: list[dict] = [{"image": _encode_image(p)} for p in image_paths]
    content.append({"text": prompt})
    messages = [{"role": "user", "content": content}]

    try:
        response = MultiModalConversation.call(
            api_key=api_key,
            model=DASHSCOPE_MODEL,
            messages=messages,
            stream=False,
            n=num_outputs,
            watermark=False,
            negative_prompt=" ",
            prompt_extend=True,
            size=size,
        )
    except Exception as exc:  # noqa: BLE001
        raise ImageToolError(f"调用 Qwen-Image-2.0-Pro 失败：{exc}") from exc

    if _attr(response, "status_code") != HTTPStatus.OK:
        raise ImageToolError(
            "Qwen-Image-2.0-Pro 返回错误："
            f"{_attr(response, 'code')} {_attr(response, 'message')}"
        )

    urls = _extract_dashscope_urls(response)
    if not urls:
        raise ImageToolError("Qwen-Image-2.0-Pro 没有返回图片，请调整提示词或稍后重试。")
    return [_save_url(u) for u in urls]


def edit_with_siliconflow(
    image_paths: list[str],
    prompt: str,
    num_outputs: int = 1,
) -> list[str]:
    """用硅基流动 Qwen-Image-Edit-2509 按提示词编辑图片。

    Args:
        image_paths: 输入图片本地路径（1-3 张）。
        prompt: 编辑指令。
        num_outputs: 期望输出的图片数量。该接口单次返回 1 张图，
            当 num_outputs > 1 时对同一请求多次调用。

    Returns:
        生成图片的本地路径列表。
    """
    _validate_inputs(image_paths, prompt)
    num_outputs = max(1, min(MAX_OUTPUT_IMAGES, int(num_outputs)))
    api_key = _get_siliconflow_key()

    encoded = [_encode_image(p) for p in image_paths]
    payload: dict = {
        "model": SILICONFLOW_MODEL,
        "prompt": prompt,
        "num_inference_steps": 20,
        "guidance_scale": 4,
        "image": encoded[0],
    }
    # 第 2、3 张图分别放入 image2 / image3 字段
    for idx, data_uri in enumerate(encoded[1:], start=2):
        payload[f"image{idx}"] = data_uri

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    saved_paths: list[str] = []
    for _ in range(num_outputs):
        try:
            resp = requests.post(
                SILICONFLOW_URL,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            raise ImageToolError(f"调用 Qwen-Image-Edit-2509 失败：{exc}") from exc

        if resp.status_code != 200:
            raise ImageToolError(
                f"Qwen-Image-Edit-2509 返回错误（HTTP {resp.status_code}）：{resp.text}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ImageToolError(
                f"无法解析 Qwen-Image-Edit-2509 的响应：{resp.text}"
            ) from exc

        urls = [
            img.get("url")
            for img in (data.get("images") or [])
            if isinstance(img, dict) and img.get("url")
        ]
        if not urls:
            raise ImageToolError(
                "Qwen-Image-Edit-2509 没有返回图片，请调整提示词或稍后重试。"
            )
        saved_paths.extend(_save_url(u) for u in urls)

    return saved_paths
