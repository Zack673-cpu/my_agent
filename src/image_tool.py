"""Nano Banana 2 Lite 图片合成工具。

封装 Google Gemini 3.1 Flash-Lite Image (gemini-3.1-flash-lite-image) 的多图输入
合成能力，并暴露为可被 LangChain 智能体调用的工具。
"""

from __future__ import annotations

import functools
import mimetypes
import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from langchain_core.tools import StructuredTool
from PIL import Image

# Nano Banana 2 Lite 的模型 ID
MODEL_NAME = "gemini-3.1-flash-lite-image"

# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# 官方支持最多 14 张输入图片
MAX_INPUT_IMAGES = 14


class ImageToolError(RuntimeError):
    """图片合成过程中的可读错误。"""


def _get_client() -> genai.Client:
    """创建 genai 客户端。

    自动读取环境变量 GOOGLE_API_KEY（等价于 curl 里的 X-goog-api-key）。
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ImageToolError(
            "未检测到 GOOGLE_API_KEY，请在 .env 中配置 AI Studio 的 API Key。"
        )
    return genai.Client(api_key=api_key)


def _load_images(image_paths: list[str]) -> list[Image.Image]:
    """读取并校验输入图片。"""
    if not image_paths:
        raise ImageToolError("没有可用的输入图片，请先上传图片。")
    if len(image_paths) > MAX_INPUT_IMAGES:
        raise ImageToolError(
            f"输入图片过多（{len(image_paths)} 张），Nano Banana 2 Lite 最多支持 "
            f"{MAX_INPUT_IMAGES} 张。"
        )

    images: list[Image.Image] = []
    for path in image_paths:
        p = Path(path)
        if not p.is_file():
            raise ImageToolError(f"找不到图片文件：{path}")
        try:
            img = Image.open(p)
            img.load()
        except Exception as exc:  # noqa: BLE001
            raise ImageToolError(f"无法读取图片 {path}：{exc}") from exc
        images.append(img)
    return images


def _extract_images(response) -> list[bytes]:
    """从响应中提取图片字节。"""
    image_blobs: list[bytes] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", None) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and getattr(inline_data, "data", None):
                image_blobs.append(inline_data.data)
    return image_blobs


def _save_blob(blob: bytes, mime_type: str = "image/png") -> str:
    """把图片字节保存到 outputs/ 并返回路径。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ext = mimetypes.guess_extension(mime_type) or ".png"
    filename = f"nb_{int(time.time() * 1000)}_{os.urandom(3).hex()}{ext}"
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(blob)
    return str(out_path)


def combine_images(
    image_paths: list[str],
    prompt: str,
    num_outputs: int = 1,
) -> list[str]:
    """按提示词把一组图片合成为一张或多张图片。

    Args:
        image_paths: 输入图片的本地路径列表（最多 14 张）。
        prompt: 描述如何合成/编辑图片的提示词。
        num_outputs: 期望输出的图片数量。该模型单次调用返回 1 张图，
            当 num_outputs > 1 时对同一提示词多次调用。

    Returns:
        生成图片的本地路径列表。
    """
    if not prompt or not prompt.strip():
        raise ImageToolError("提示词不能为空。")

    num_outputs = max(1, int(num_outputs))
    images = _load_images(image_paths)
    client = _get_client()

    saved_paths: list[str] = []
    for _ in range(num_outputs):
        contents = [prompt, *images]
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["Image"]),
            )
        except Exception as exc:  # noqa: BLE001
            raise ImageToolError(f"调用 Nano Banana 2 Lite 失败：{exc}") from exc

        blobs = _extract_images(response)
        if not blobs:
            raise ImageToolError(
                "模型没有返回图片，请调整提示词或稍后重试。"
            )
        for blob in blobs:
            saved_paths.append(_save_blob(blob))

    return saved_paths


def build_combine_tool(image_paths: list[str], num_outputs: int = 1) -> StructuredTool:
    """构建绑定了本次请求图片的 LangChain 工具。

    智能体只需传入提示词，输入图片路径通过 partial 注入，避免全局状态污染。
    """
    bound = functools.partial(
        combine_images, image_paths, num_outputs=num_outputs
    )

    def _run(prompt: str) -> str:
        paths = bound(prompt)
        return "已生成图片：\n" + "\n".join(paths)

    return StructuredTool.from_function(
        func=_run,
        name="combine_images",
        description=(
            "把用户已上传的一组图片按提示词合成/编辑为图片。"
            "参数 prompt 为合成的文字描述。返回生成图片的本地路径。"
        ),
    )
