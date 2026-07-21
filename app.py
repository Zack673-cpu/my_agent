"""Gradio 网页界面：上传图片 + 提示词 + 选择模型，执行图片编辑。

- 默认走阿里百炼 Qwen-Image-2.0-Pro；调用失败时自动切换到硅基流动
  Qwen-Image-Edit-2509。
- 也可在前端手动指定使用某一个模型。
"""

from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv

from src.agent import MODEL_AUTO, MODEL_CHOICES, run_edit
from src.image_tool import MAX_INPUT_IMAGES, ImageToolError

load_dotenv()

# gr.Radio 的选项：(展示名, 标识)
_RADIO_CHOICES = [(label, value) for value, label in MODEL_CHOICES]


def _to_paths(files) -> list[str]:
    """把 Gradio 上传的文件对象转成本地路径列表。"""
    if not files:
        return []
    paths: list[str] = []
    for f in files:
        # gr.File 在不同版本可能返回带 .name 的对象或直接是路径字符串
        path = getattr(f, "name", None) or f
        paths.append(str(path))
    return paths


def generate(files, prompt: str, num_outputs: int, model: str):
    image_paths = _to_paths(files)

    if not image_paths:
        raise gr.Error("请先上传至少一张图片。")
    if len(image_paths) > MAX_INPUT_IMAGES:
        raise gr.Error(f"最多支持 {MAX_INPUT_IMAGES} 张输入图片。")
    if not prompt or not prompt.strip():
        raise gr.Error("请输入提示词。")

    try:
        outputs, status = run_edit(
            image_paths, prompt.strip(), int(num_outputs), model=model
        )
    except ImageToolError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"生成失败：{exc}") from exc

    if not outputs:
        raise gr.Error("没有生成任何图片，请调整提示词后重试。")

    return outputs, status


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen-Image 图片编辑智能体") as demo:
        gr.Markdown(
            "# Qwen-Image 图片编辑智能体\n"
            f"上传图片（最多 {MAX_INPUT_IMAGES} 张）并输入提示词进行编辑。\n"
            "- 默认使用阿里百炼 `Qwen-Image-2.0-Pro`，调用失败时自动切换到"
            "硅基流动 `Qwen/Qwen-Image-Edit-2509`；\n"
            "- 也可在下方手动指定要使用的模型。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(
                    label=f"上传图片（最多 {MAX_INPUT_IMAGES} 张）",
                    file_count="multiple",
                    file_types=["image"],
                )
                prompt = gr.Textbox(
                    label="提示词",
                    placeholder="例如：把背景换成雪山；将两张图中的人物合成到同一场景",
                    lines=3,
                )
                model = gr.Radio(
                    label="模型选择",
                    choices=_RADIO_CHOICES,
                    value=MODEL_AUTO,
                )
                num_outputs = gr.Number(
                    label="输出张数（最多 6 张）",
                    value=1,
                    minimum=1,
                    maximum=6,
                    step=1,
                    precision=0,
                )
                run_btn = gr.Button("开始编辑", variant="primary")
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="生成结果",
                    columns=2,
                    height="auto",
                )
                status = gr.Textbox(label="状态", interactive=False)

        run_btn.click(
            fn=generate,
            inputs=[files, prompt, num_outputs, model],
            outputs=[gallery, status],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
