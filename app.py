"""Gradio 网页界面：上传一组图片 + 提示词，调用智能体用 Nano Banana 2 Lite 合成。"""

from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv

from src.agent import run_agent
from src.image_tool import ImageToolError

load_dotenv()


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


def generate(files, prompt: str, num_outputs: int):
    image_paths = _to_paths(files)

    if not image_paths:
        raise gr.Error("请先上传至少一张图片。")
    if not prompt or not prompt.strip():
        raise gr.Error("请输入提示词。")

    try:
        outputs = run_agent(image_paths, prompt.strip(), int(num_outputs))
    except ImageToolError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"生成失败：{exc}") from exc

    if not outputs:
        raise gr.Error("没有生成任何图片，请调整提示词后重试。")

    return outputs


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Nano Banana 2 Lite 图片合成智能体") as demo:
        gr.Markdown(
            "# Nano Banana 2 Lite 图片合成智能体\n"
            "上传一组图片，输入提示词，由 LangChain 智能体调用 "
            "`gemini-3.1-flash-lite-image` 合成新图片。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(
                    label="上传图片（最多 14 张）",
                    file_count="multiple",
                    file_types=["image"],
                )
                prompt = gr.Textbox(
                    label="提示词",
                    placeholder="例如：把这些图片合成一张拼贴海报",
                    lines=3,
                )
                num_outputs = gr.Number(
                    label="输出张数",
                    value=1,
                    minimum=1,
                    maximum=8,
                    step=1,
                    precision=0,
                )
                run_btn = gr.Button("开始合成", variant="primary")
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="合成结果",
                    columns=2,
                    height="auto",
                )

        run_btn.click(
            fn=generate,
            inputs=[files, prompt, num_outputs],
            outputs=gallery,
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
