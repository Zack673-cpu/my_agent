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
from src.planner import TASK_CHOICES, run_agent
from src.prompt_template import TEMPLATE_CHOICES, build_prompt

load_dotenv()

# gr.Radio 的选项：(展示名, 标识)
_RADIO_CHOICES = [(label, value) for value, label in MODEL_CHOICES]

# 提示词模板下拉：(展示名, 标识)
_TEMPLATE_RADIO_CHOICES = [(label, value) for value, label in TEMPLATE_CHOICES]

# 智能体任务类型：(展示名, 标识)
_TASK_RADIO_CHOICES = [(label, value) for value, label in TASK_CHOICES]


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


def fill_template(template_key: str) -> str:
    """按所选模板返回提示词文本，供“填充模板”按钮写入提示词框。"""
    return build_prompt(template_key)


def generate_agent(files, user_request: str, task_type: str, model: str):
    """智能体模式：看图 → 自动写提示词 → 出图 → 自检 → 自动重试。"""
    image_paths = _to_paths(files)

    if not image_paths:
        raise gr.Error("请先上传至少一张图片。")
    if len(image_paths) > MAX_INPUT_IMAGES:
        raise gr.Error(f"最多支持 {MAX_INPUT_IMAGES} 张输入图片。")
    if not user_request or not user_request.strip():
        raise gr.Error("请用一句话描述你想要的效果。")

    try:
        outputs, thoughts, status = run_agent(
            image_paths, task_type, user_request.strip(), model=model
        )
    except ImageToolError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"智能体运行失败：{exc}") from exc

    if not outputs:
        raise gr.Error("没有生成任何图片，请调整需求后重试。")

    return outputs, thoughts, status


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen-Image 图片编辑智能体") as demo:
        gr.Markdown(
            "# Qwen-Image 图片编辑智能体\n"
            "- **\U0001F916 智能体模式**：只需上传图片 + 一句白话需求，"
            "由 `qwen3.7-plus` 自动看图、写提示词、出图、自检并自动重试；\n"
            "- **\U0001F6E0\uFE0F 手动模式**：自己填提示词、选模型，适合精细控制。"
        )

        with gr.Tab("\U0001F916 智能体模式"):
            gr.Markdown(
                "上传图片（人物参考图 + 可选背景图，最多 "
                f"{MAX_INPUT_IMAGES} 张），用一句话描述你想要的效果，"
                "剩下的交给智能体。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    a_files = gr.File(
                        label=f"上传图片（最多 {MAX_INPUT_IMAGES} 张）",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    a_task = gr.Radio(
                        label="任务类型",
                        choices=_TASK_RADIO_CHOICES,
                        value=_TASK_RADIO_CHOICES[0][1],
                    )
                    a_request = gr.Textbox(
                        label="一句话需求（白话即可）",
                        placeholder=(
                            "例：让孩子躲在大叶片后面探出半个身子看小鸭；"
                            "或：把他做成皮克斯风卡通形象站在花园里"
                        ),
                        lines=3,
                    )
                    a_model = gr.Radio(
                        label="出图模型",
                        choices=_RADIO_CHOICES,
                        value=MODEL_AUTO,
                    )
                    a_run_btn = gr.Button("\U0001F916 启动智能体", variant="primary")
                with gr.Column(scale=1):
                    a_gallery = gr.Gallery(
                        label="生成结果",
                        columns=2,
                        height="auto",
                    )
                    a_status = gr.Textbox(label="状态", interactive=False)
                    a_thoughts = gr.Markdown(label="智能体思考过程")

            a_run_btn.click(
                fn=generate_agent,
                inputs=[a_files, a_request, a_task, a_model],
                outputs=[a_gallery, a_thoughts, a_status],
            )

        with gr.Tab("\U0001F6E0\uFE0F 手动模式"):
            gr.Markdown(
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
                    template_choice = gr.Dropdown(
                        label="内置提示词模板（真人合成到绘本）",
                        choices=_TEMPLATE_RADIO_CHOICES,
                        value="",
                    )
                    fill_btn = gr.Button("填充模板到提示词")
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

            fill_btn.click(
                fn=fill_template,
                inputs=[template_choice],
                outputs=[prompt],
            )
    return demo


if __name__ == "__main__":
    build_ui().launch()
