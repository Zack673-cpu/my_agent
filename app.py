"""FastAPI 后端：为「绘本智能体」单页前端提供接口。

前端是 static/ 下的纯原生单页应用，整个体验分四个阶段：

1. 欢迎页：选模型 → 点启动。
2. 读取素材：POST /api/scan 读取桌面的「背景图」「人物」两个文件夹。
3. 生成控制台：GET /api/generate（SSE）对每张背景图跑
   「感知 → 规划 → 出图」的精简智能体流程，逐步骤实时推送。
4. 儿童绘本：前端把生成结果装进 3D 翻页绘本展示。

后端不改动 src/ 下的任何模块，只做复用与编排。
"""

from __future__ import annotations

import json
import os
import re
import threading
import webbrowser
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from src.agent import MODEL_AUTO, MODEL_BAILIAN, MODEL_SILICONFLOW
from src.image_tool import ImageToolError
from src.planner import TASK_CARTOON, TASK_REALISTIC, run_agent_stream

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# 桌面素材文件夹的约定名称
BG_FOLDER_NAME = "背景图"
PERSON_FOLDER_NAME = "人物"

# 识别为图片的扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# 合法的模型 / 任务取值（防止前端传来脏数据）
VALID_MODELS = {MODEL_AUTO, MODEL_BAILIAN, MODEL_SILICONFLOW}
VALID_TASKS = {TASK_REALISTIC, TASK_CARTOON}

# 文本框留空时，按任务类型自动补的默认白话需求
DEFAULT_REQUESTS = {
    TASK_REALISTIC: "让人物自然地融入这张背景，姿态动作贴合场景氛围，光影协调统一。",
    TASK_CARTOON: "把人物卡通化后自然融入这张背景，必须保留人物最显著的特征，一眼能认出是本人。",
}

# 中文数字 → 阿拉伯数字，用于「背景图（一）（二）（三）」的排序
_CN_NUMS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

app = FastAPI(title="绘本智能体")


@app.middleware("http")
async def no_cache_frontend(request: Request, call_next):
    """给页面和静态资源加 no-cache：浏览器每次使用前必须向服务器确认文件是否有更新，
    改完前端代码后普通刷新即可生效，不再需要 Ctrl+F5（未改动时走 304，开销很小）。"""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

# 扫描结果注册表：文件 id -> 本地绝对路径。
# 前端只拿到 id，通过 /api/file/{fid} 取图，避免把本地路径直接暴露成任意文件读取。
_files: dict[str, str] = {}


# ---------------------------------------------------------------------------
# 桌面素材扫描
# ---------------------------------------------------------------------------
def _desktop_candidates() -> list[Path]:
    """列出可能的桌面路径：本地桌面与 OneDrive 桌面都兼容。"""
    home = Path.home()
    candidates = [
        home / "Desktop",
        home / "桌面",
        home / "OneDrive" / "Desktop",
        home / "OneDrive" / "桌面",
    ]
    onedrive = os.environ.get("OneDrive")
    if onedrive:
        candidates += [Path(onedrive) / "Desktop", Path(onedrive) / "桌面"]
    # 去重并只保留真实存在的目录
    seen: set[str] = set()
    result: list[Path] = []
    for c in candidates:
        key = str(c).lower()
        if key not in seen and c.is_dir():
            seen.add(key)
            result.append(c)
    return result


def _find_material_dir(folder_name: str) -> Path | None:
    """在各候选桌面下找到指定名称的素材文件夹。"""
    for desktop in _desktop_candidates():
        target = desktop / folder_name
        if target.is_dir():
            return target
    return None


def _list_images(folder: Path) -> list[Path]:
    """列出文件夹里的所有图片文件。"""
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def _order_key(name: str) -> tuple:
    """背景图排序键：优先按文件名里的数字（阿拉伯或中文）排序。"""
    m = re.search(r"\d+", name)
    if m:
        return (0, int(m.group()), name)
    for ch in name:
        if ch in _CN_NUMS:
            return (0, _CN_NUMS[ch], name)
    return (1, 0, name)


@app.post("/api/scan")
def scan() -> JSONResponse:
    """读取桌面素材：背景图列表（排序后）+ 人物参照图。"""
    bg_dir = _find_material_dir(BG_FOLDER_NAME)
    person_dir = _find_material_dir(PERSON_FOLDER_NAME)
    if bg_dir is None or person_dir is None:
        missing = [
            name for name, d in
            [(BG_FOLDER_NAME, bg_dir), (PERSON_FOLDER_NAME, person_dir)]
            if d is None
        ]
        return JSONResponse(
            {"error": f"在桌面上找不到文件夹：{'、'.join(missing)}。"
                      "请确认桌面上存在「背景图」和「人物」两个文件夹。"},
            status_code=404,
        )

    backgrounds = sorted(_list_images(bg_dir), key=lambda p: _order_key(p.stem))
    persons = _list_images(person_dir)
    if not backgrounds:
        return JSONResponse(
            {"error": f"「{BG_FOLDER_NAME}」文件夹里没有找到任何图片。"},
            status_code=404,
        )
    if not persons:
        return JSONResponse(
            {"error": f"「{PERSON_FOLDER_NAME}」文件夹里没有找到任何图片。"},
            status_code=404,
        )
    # 人物图优先取文件名包含「人物」的那张，保证 planner 的文件名角色绑定生效
    person = next((p for p in persons if "人物" in p.stem), persons[0])

    _files.clear()
    bg_items = []
    for i, p in enumerate(backgrounds, start=1):
        fid = f"bg{i}"
        _files[fid] = str(p)
        bg_items.append({"id": fid, "name": p.stem, "url": f"/api/file/{fid}"})
    _files["person"] = str(person)

    return JSONResponse({
        "backgrounds": bg_items,
        "person": {"id": "person", "name": person.stem, "url": "/api/file/person"},
    })


@app.get("/api/file/{fid}")
def get_file(fid: str):
    """按注册表 id 返回图片文件（素材图或生成结果）。"""
    path = _files.get(fid)
    if not path or not Path(path).is_file():
        return JSONResponse({"error": "文件不存在或已失效。"}, status_code=404)
    return FileResponse(path)


# ---------------------------------------------------------------------------
# 生成（SSE 流式推送步骤事件）
# ---------------------------------------------------------------------------
def _sse(data: dict) -> str:
    """把一个事件对象编码成 SSE 帧。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/generate")
def generate(bg: str, task: str, model: str = MODEL_AUTO, request: str = ""):
    """对「一张背景图 + 人物图」跑精简智能体流程，逐步骤 SSE 推送。

    max_retries=0 时工作流天然只走「感知 → 规划 → 出图」三步，
    出图后直接结束、不做自检重试。
    """

    def stream():
        bg_path = _files.get(bg)
        person_path = _files.get("person")
        if not bg_path or not person_path:
            yield _sse({"type": "error",
                        "message": "素材已失效，请回到首页重新启动读取。"})
            return
        if task not in VALID_TASKS:
            yield _sse({"type": "error", "message": f"未知的任务类型：{task}"})
            return
        if model not in VALID_MODELS:
            yield _sse({"type": "error", "message": f"未知的模型选项：{model}"})
            return

        user_request = request.strip() or DEFAULT_REQUESTS[task]
        try:
            ctx = None
            for event, ctx in run_agent_stream(
                [bg_path, person_path], task, user_request, model, max_retries=0
            ):
                yield _sse({
                    "type": "step",
                    "key": event.step_key,
                    "status": event.status,
                    "summary": event.summary,
                })
            if ctx is None or not ctx.outputs:
                yield _sse({"type": "error", "message": "模型没有返回图片。"})
                return
            # 把生成结果也登记进注册表，前端凭 id 取图
            out_path = ctx.outputs[0]
            fid = f"out_{bg}_{len(_files)}"
            _files[fid] = out_path
            yield _sse({
                "type": "result",
                "url": f"/api/file/{fid}",
                "status": ctx.gen_status,
            })
        except ImageToolError as exc:
            yield _sse({"type": "error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"发生未知错误：{exc}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# 静态资源与首页
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """返回单页应用入口。"""
    return FileResponse(STATIC_DIR / "index.html")


# 浏览器会自发探测 /favicon.ico，直接返回一个内联 SVG 图标，避免 404 日志
FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
    "<text y='.9em' font-size='90'>\U0001F4D6</text></svg>"
)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """返回标签页小图标。"""
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


if __name__ == "__main__":
    # 服务就绪大约需要一两秒，延迟打开浏览器，避免页面比服务先一步请求失败
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    # timeout_graceful_shutdown：Ctrl+C 后最多等 3 秒，超时强制断开残留连接，
    # 避免浏览器还挂着 keep-alive/SSE 连接时关不掉、卡在 "Shutting down"
    uvicorn.run(app, host="127.0.0.1", port=7860, timeout_graceful_shutdown=3)
