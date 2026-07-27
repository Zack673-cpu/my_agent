"""轻量工作流引擎：用「节点 + 上下文 + 条件流转 + 事件流」显式表达工作流。

核心概念：
- WorkflowContext：在节点之间流转的共享数据。
- Step：一个工作流节点，run(ctx) 返回下一节点 key（条件分支 / 循环回边）、
  END（结束）或 None（按注册顺序走下一个节点）。
- StepEvent：节点执行事件（running / done / retry），供前端实时渲染状态条。
- Workflow：按图驱动节点执行的引擎，run(ctx) 是生成器，逐事件产出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

# 节点返回 END 表示工作流结束
END = "__end__"

# 防止条件回边写错导致死循环的兜底上限
_MAX_TOTAL_STEPS = 30


@dataclass
class WorkflowContext:
    """节点间流转的共享上下文。"""

    image_paths: list[str] = field(default_factory=list)
    task_type: str = ""
    user_request: str = ""
    model: str = ""
    max_retries: int = 2

    # 各节点写入的中间结果
    perception: dict = field(default_factory=dict)
    person_image: str = ""
    prompt: str = ""
    outputs: list[str] = field(default_factory=list)
    review: dict = field(default_factory=dict)
    attempt: int = 0
    gen_status: str = ""

    # Markdown 详细日志（供前端折叠面板展示）
    logs: list[str] = field(default_factory=list)

    def log(self, text: str) -> None:
        self.logs.append(text)

    def thoughts(self) -> str:
        return "\n\n".join(self.logs)


@dataclass
class StepEvent:
    """节点执行事件，供前端实时渲染状态条。"""

    step_key: str
    status: str  # running / done / retry
    summary: str = ""


# 节点执行函数：读写 ctx，返回 (下一节点 key 或 None/END, 摘要文本)
StepFn = Callable[[WorkflowContext], tuple[Optional[str], str]]


@dataclass
class Step:
    """工作流节点。"""

    key: str
    label: str
    run: StepFn


class Workflow:
    """按注册顺序 + 节点返回的跳转 key 驱动执行的轻量引擎。"""

    def __init__(self, steps: list[Step]):
        self.steps = steps
        self._index = {step.key: i for i, step in enumerate(steps)}

    def run(self, ctx: WorkflowContext) -> Iterator[StepEvent]:
        """执行工作流，逐个产出 StepEvent。"""
        pos = 0
        executed = 0
        while 0 <= pos < len(self.steps):
            if executed >= _MAX_TOTAL_STEPS:
                raise RuntimeError("工作流执行步数超出上限，疑似节点回边形成死循环。")
            step = self.steps[pos]
            executed += 1

            yield StepEvent(step.key, "running")
            next_key, summary = step.run(ctx)

            if next_key is not None and next_key != END:
                if next_key not in self._index:
                    raise RuntimeError(f"节点 {step.key} 返回了未注册的跳转目标：{next_key}")
                # 跳回之前的节点视为重试回边
                is_back_edge = self._index[next_key] <= pos
                yield StepEvent(step.key, "retry" if is_back_edge else "done", summary)
                pos = self._index[next_key]
                continue

            yield StepEvent(step.key, "done", summary)
            if next_key == END:
                break
            pos += 1
