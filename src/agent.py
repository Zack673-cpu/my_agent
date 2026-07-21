"""LangChain 智能体：编排 Nano Banana 2 Lite 图片合成工具。"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .image_tool import build_combine_tool

load_dotenv()

# 智能体推理用的文本模型
REASONING_MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = (
    "你是一个图片合成助手。用户已经上传了一组图片，并给出提示词。"
    "你必须调用 combine_images 工具，把用户提示词作为 prompt 传入，"
    "以将这些图片合成/编辑为新的图片。不要编造图片内容，一切合成由工具完成。"
    "工具返回生成图片的本地路径后，用简洁的中文告知用户已完成。"
)

# 从工具输出中解析图片路径
_PATH_RE = re.compile(r"(?im)^(.*\.(?:png|jpe?g|webp))\s*$")


def _build_agent(image_paths: list[str], num_outputs: int) -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model=REASONING_MODEL,
        temperature=0.2,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )
    tool = build_combine_tool(image_paths, num_outputs=num_outputs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, [tool], prompt)
    return AgentExecutor(
        agent=agent,
        tools=[tool],
        verbose=False,
        return_intermediate_steps=True,
    )


def _collect_paths(result: dict) -> list[str]:
    """从智能体的中间步骤与最终输出里收集生成的图片路径。"""
    paths: list[str] = []
    for _action, observation in result.get("intermediate_steps", []) or []:
        if isinstance(observation, str):
            paths.extend(_PATH_RE.findall(observation))
    if not paths:
        output = result.get("output", "")
        if isinstance(output, str):
            paths.extend(_PATH_RE.findall(output))
    # 去重并保留顺序
    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def run_agent(
    image_paths: list[str],
    user_prompt: str,
    num_outputs: int = 1,
) -> list[str]:
    """运行智能体，返回生成图片的本地路径列表。"""
    executor = _build_agent(image_paths, num_outputs)
    result = executor.invoke({"input": user_prompt})
    return _collect_paths(result)
