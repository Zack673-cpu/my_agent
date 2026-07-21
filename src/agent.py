"""模型调度器：选择图片编辑模型并在默认模型失败时自动降级。

输入固定为「一组图片 + 提示词」。支持三种模式：

- auto（默认）：先用阿里百炼 Qwen-Image-2.0-Pro，失败时自动切换到硅基流动
  Qwen-Image-Edit-2509。
- bailian：仅使用阿里百炼 Qwen-Image-2.0-Pro。
- siliconflow：仅使用硅基流动 Qwen-Image-Edit-2509。
"""

from __future__ import annotations

from dotenv import load_dotenv

from .image_tool import (
    ImageToolError,
    edit_with_dashscope,
    edit_with_siliconflow,
)

load_dotenv()

# 模型选择标识
MODEL_AUTO = "auto"
MODEL_BAILIAN = "bailian"
MODEL_SILICONFLOW = "siliconflow"

# 供前端使用的可选项：(标识, 展示名)
MODEL_CHOICES: list[tuple[str, str]] = [
    (MODEL_AUTO, "自动（默认百炼，失败切换硅基流动）"),
    (MODEL_BAILIAN, "阿里百炼 Qwen-Image-2.0-Pro"),
    (MODEL_SILICONFLOW, "硅基流动 Qwen-Image-Edit-2509"),
]

_BAILIAN_LABEL = "阿里百炼 Qwen-Image-2.0-Pro"
_SILICONFLOW_LABEL = "硅基流动 Qwen-Image-Edit-2509"


def run_edit(
    image_paths: list[str],
    prompt: str,
    num_outputs: int = 1,
    model: str = MODEL_AUTO,
) -> tuple[list[str], str]:
    """按所选模型执行图片编辑，返回（生成图片路径列表, 状态说明）。

    - model=bailian：仅调用阿里百炼。
    - model=siliconflow：仅调用硅基流动。
    - model=auto：先调用阿里百炼，抛出 ImageToolError 时自动降级到硅基流动。
    """
    if model == MODEL_BAILIAN:
        paths = edit_with_dashscope(image_paths, prompt, num_outputs=num_outputs)
        return paths, f"由{_BAILIAN_LABEL}生成。"

    if model == MODEL_SILICONFLOW:
        paths = edit_with_siliconflow(image_paths, prompt, num_outputs=num_outputs)
        return paths, f"由{_SILICONFLOW_LABEL}生成。"

    # 默认 auto：先百炼，失败降级硅基流动
    try:
        paths = edit_with_dashscope(image_paths, prompt, num_outputs=num_outputs)
        return paths, f"由{_BAILIAN_LABEL}生成。"
    except ImageToolError as primary_exc:
        try:
            paths = edit_with_siliconflow(
                image_paths, prompt, num_outputs=num_outputs
            )
        except ImageToolError as fallback_exc:
            raise ImageToolError(
                "两个模型均调用失败。\n"
                f"- {_BAILIAN_LABEL}：{primary_exc}\n"
                f"- {_SILICONFLOW_LABEL}：{fallback_exc}"
            ) from fallback_exc
        return (
            paths,
            f"{_BAILIAN_LABEL}调用失败（{primary_exc}），"
            f"已自动切换到{_SILICONFLOW_LABEL}生成。",
        )
