"""智能体编排层：让 LLM 自主完成「看图 → 写提示词 → 出图 → 自检 → 重试」。

与 ``agent.py`` 的区别：
- ``agent.py`` 只负责「按选定模型出图 + 失败降级」，是被动的工具调用。
- ``planner.py`` 用多模态模型 ``qwen3.7-plus`` 作为大脑，主动完成：

    1. 感知（perceive）：看懂上传的图片，提取人物最显著特征、背景可用元素与光线。
    2. 规划（plan_prompt）：根据「看到的内容 + 用户白话需求」自动写出专业融合提示词。
    3. 执行（run_edit）：调用图像编辑模型出图。
    4. 反思（critique）：再看结果，判断是否换脸/拼贴/遮挡不当/特征丢失，
       不合格则给出修改建议并自动重试。

这条「看图—生成—看结果—修正」的闭环，是本项目体现「智能体」的核心。
"""

from __future__ import annotations

import json
import re

from .agent import run_edit
from .image_tool import ImageToolError, understand_with_dashscope

# 两种任务类型
TASK_REALISTIC = "realistic_fusion"
TASK_CARTOON = "cartoonize"

# 供前端使用的可选项：(标识, 展示名)
TASK_CHOICES: list[tuple[str, str]] = [
    (TASK_REALISTIC, "真人融合（被遮挡 / 做动作 / 局部露出）"),
    (TASK_CARTOON, "卡通化（保留真人最显著特征）"),
]

# 每次任务最多自动重试次数（首次 + 重试，总尝试 = 1 + MAX_RETRIES）
DEFAULT_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# 各步骤的系统提示词（智能体的“行为准则”）
# ---------------------------------------------------------------------------
_PERCEIVE_SYSTEM = (
    "你是资深的图像分析师，擅长从照片中提取人物身份特征与场景信息，"
    "为后续的图像合成 / 卡通化提供精确依据。只输出 JSON，不要解释。"
)

_PERCEIVE_USER = """\
请分析我上传的图片（按上传顺序编号，从 1 开始）。对每张图判断它更适合作为
【人物参考图】还是【背景图】，并输出严格的 JSON（简体中文），结构如下：

{
  "images": [
    {
      "index": 1,
      "role": "person 或 background",
      "distinctive_features": "若为人物：用一句话列出最具辨识度、卡通化后也必须保留的特征，如：方脸、大眼睛双眼皮、齐刘海、肤色偏白、左脸颊有痣。若为背景：留空字符串",
      "description": "若为人物：脸型/五官/发型发色/肤色/年龄段/服装款式与颜色。若为背景：场景内容、主要元素、可用于自然遮挡人物的前景物体"
    }
  ],
  "light_direction": "整体光线方向，如：左上方顺光",
  "possible_occluders": "背景中可用来自然遮挡人物的前景物体，如：大叶片、树干、栏杆；没有则留空"
}

只输出 JSON 本身，不要加 ```json 代码块标记或任何额外文字。"""


def _strip_json(text: str) -> str:
    """去掉模型可能输出的 ```json 代码块包裹，返回纯 JSON 文本。"""
    t = text.strip()
    if t.startswith("```"):
        # 去掉首行 ``` 或 ```json，以及结尾 ```
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _safe_json(text: str) -> dict:
    """尽力把模型输出解析成 dict，失败时返回空 dict。"""
    try:
        data = json.loads(_strip_json(text))
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def perceive(image_paths: list[str]) -> dict:
    """看图并返回结构化感知结果（人物特征 / 背景元素 / 光线 / 遮挡物）。"""
    raw = understand_with_dashscope(
        image_paths, _PERCEIVE_USER, system=_PERCEIVE_SYSTEM
    )
    data = _safe_json(raw)
    data["_raw"] = raw
    return data


def _perception_brief(perception: dict) -> str:
    """把感知结果压成一段可读文本，喂给后续的提示词规划步骤。"""
    lines: list[str] = []
    for img in perception.get("images", []) or []:
        role = img.get("role", "")
        idx = img.get("index", "?")
        feats = img.get("distinctive_features", "")
        desc = img.get("description", "")
        lines.append(f"- 第{idx}张（{role}）：{desc}")
        if feats:
            lines.append(f"  · 最显著特征：{feats}")
    if perception.get("light_direction"):
        lines.append(f"- 光线方向：{perception['light_direction']}")
    if perception.get("possible_occluders"):
        lines.append(f"- 可用遮挡物：{perception['possible_occluders']}")
    return "\n".join(lines) if lines else perception.get("_raw", "")


def _plan_system(task_type: str) -> str:
    if task_type == TASK_CARTOON:
        return (
            "你是图像编辑提示词工程专家，擅长为「真人卡通化 + 背景融合」任务撰写"
            "中文提示词。核心要求：卡通化的同时，必须强约束保留真人最具辨识度的特征，"
            "让人一眼认出是本人。直接输出可用的提示词全文，不要解释、不要加标题。"
        )
    return (
        "你是图像编辑提示词工程专家，擅长为「真人自然融入场景」任务撰写中文提示词。"
        "核心要求：锁定背景不改、严格保留真人五官不换脸、光影自然融合、边缘无拼贴白边，"
        "并合理安排遮挡与动作。直接输出可用的提示词全文，不要解释、不要加标题。"
    )


def _plan_user(task_type: str, user_request: str, brief: str) -> str:
    from .prompt_template import CARTOON_TEMPLATE, COMMON_TEMPLATE

    template = CARTOON_TEMPLATE if task_type == TASK_CARTOON else COMMON_TEMPLATE
    return f"""\
【用户的白话需求】
{user_request}

【看图得到的信息】
{brief}

【提示词结构模板（请遵循其分节结构，把其中 <...> 占位替换为具体内容）】
{template}

请据此产出一段完整、可直接用于图像编辑模型的中文提示词。要求：
- 把模板里所有 <...> 占位都替换成结合上面信息的具体描述；
- 忠实体现用户白话需求中的动作 / 遮挡 / 场景意图；
- 保留模板中对「特征保留 / 背景锁定 / 光影融合 / 边缘精修」的强约束语句；
- 只输出提示词正文本身，不要任何前后缀说明。"""


def plan_prompt(task_type: str, user_request: str, perception: dict) -> str:
    """根据任务类型 + 用户需求 + 感知结果，自动生成专业提示词。"""
    brief = _perception_brief(perception)
    text = understand_with_dashscope(
        [],
        _plan_user(task_type, user_request, brief),
        system=_plan_system(task_type),
    )
    return _strip_json(text) if text.strip().startswith("```") else text.strip()


_CRITIQUE_SYSTEM = (
    "你是严格的图像质检员，负责对比原图与生成结果，判断合成质量是否达标。"
    "只输出 JSON，不要解释。"
)


def _critique_user(task_type: str, user_request: str) -> str:
    if task_type == TASK_CARTOON:
        checklist = (
            "1) 卡通形象是否明显保留了真人最显著特征（脸型/眼睛/发型等），一眼能认出是本人；"
            "2) 人物与背景是否风格统一、光影一致、无拼贴白边；"
            "3) 是否符合用户描述的场景与动作。"
        )
    else:
        checklist = (
            "1) 是否保持了真人的五官脸型、没有换脸或美化成网红脸；"
            "2) 背景是否被保留未被重绘；"
            "3) 遮挡与动作是否自然、符合用户描述、未遮挡面部关键部位；"
            "4) 人物边缘是否干净、无白边黑边拼贴感、光影是否融合。"
        )
    return f"""\
第 1 张是【原始人物参考图】，第 2 张是【生成结果】。用户的需求是：{user_request}

请对照以下要点检查生成结果：
{checklist}

输出严格 JSON（简体中文），不要加代码块标记：
{{
  "passed": true 或 false,
  "issues": ["逐条列出发现的问题，没有则为空数组"],
  "suggestion": "若未通过，给出对提示词的具体修改建议（应补充或强调什么），通过则留空"
}}"""


def critique(
    task_type: str,
    user_request: str,
    person_image: str,
    result_image: str,
) -> dict:
    """看生成结果并判定是否合格，返回 {passed, issues, suggestion}。"""
    raw = understand_with_dashscope(
        [person_image, result_image],
        _critique_user(task_type, user_request),
        system=_CRITIQUE_SYSTEM,
    )
    data = _safe_json(raw)
    if "passed" not in data:
        # 解析失败时保守放行，避免无谓重试，并把原始文本带出用于展示
        data = {"passed": True, "issues": [], "suggestion": "", "_raw": raw}
    return data


def _pick_person_image(image_paths: list[str], perception: dict) -> str:
    """从感知结果里挑出人物参考图，失败则退回第一张。"""
    for img in perception.get("images", []) or []:
        if img.get("role") == "person":
            idx = img.get("index")
            if isinstance(idx, int) and 1 <= idx <= len(image_paths):
                return image_paths[idx - 1]
    return image_paths[0]


def run_agent(
    image_paths: list[str],
    task_type: str,
    user_request: str,
    model: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[list[str], str, str]:
    """智能体主流程，返回（最终图片路径列表, 思考过程 Markdown, 状态说明）。

    流程：感知 → 规划提示词 → 出图 → 质检 → 不合格则带建议重试。
    思考过程 Markdown 会记录每一步的推理，用于前端展示智能体的自主决策。
    """
    if not image_paths:
        raise ImageToolError("没有可用的输入图片，请先上传图片。")
    if not user_request or not user_request.strip():
        raise ImageToolError("请用一句话描述你想要的效果。")

    log: list[str] = []
    task_label = dict(TASK_CHOICES).get(task_type, task_type)
    log.append(f"### 🤖 智能体开始工作（任务：{task_label}）")

    # 1. 感知
    log.append("**第 1 步 · 👁️ 感知**：正在看图，提取人物特征与场景信息……")
    perception = perceive(image_paths)
    brief = _perception_brief(perception)
    log.append(f"感知结果：\n{brief}")

    person_image = _pick_person_image(image_paths, perception)

    # 2. 规划提示词
    log.append("**第 2 步 · 🧠 规划**：结合需求自动撰写专业提示词……")
    prompt = plan_prompt(task_type, user_request.strip(), perception)
    log.append(f"生成的提示词：\n```\n{prompt}\n```")

    # 3~4. 出图 + 反思 + 重试
    final_paths: list[str] = []
    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        log.append(f"**第 3 步 · 🎨 出图（第 {attempt} 次尝试）**：调用图像模型生成……")
        paths, gen_status = run_edit(image_paths, prompt, num_outputs=1, model=model)
        log.append(f"出图完成（{gen_status}）。")
        final_paths = paths

        if attempt > max_retries:
            log.append("已达最大尝试次数，返回当前结果。")
            break

        # 反思
        log.append("**第 4 步 · 🔍 自检**：对照原图检查生成质量……")
        review = critique(task_type, user_request.strip(), person_image, paths[0])
        if review.get("passed"):
            log.append("✅ 自检通过：结果符合要求。")
            break

        issues = review.get("issues") or []
        suggestion = review.get("suggestion") or ""
        issue_text = "；".join(issues) if issues else "存在质量问题"
        log.append(f"⚠️ 自检未通过：{issue_text}")
        if suggestion:
            log.append(f"改进建议：{suggestion}")
            # 把建议追加进提示词后重试
            prompt = f"{prompt}\n\n【重点修正】{suggestion}"
        log.append("→ 智能体决定调整提示词并重试。")

    status = f"智能体已完成，共尝试 {attempt} 次。"
    return final_paths, "\n\n".join(log), status
