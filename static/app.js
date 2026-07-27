/* ==========================================================================
   AI 绘本工坊 · 前端逻辑
   四个阶段：欢迎页 → 读取素材 → 生成控制台（SSE 时间线）→ 3D 翻页绘本
   ========================================================================== */

"use strict";

/* ---------- 全局状态 ---------- */
const state = {
  model: "auto",          // 欢迎页选定后整个批次沿用
  backgrounds: [],        // [{id, name, url}]
  person: null,           // {id, name, url}
  results: [],            // 与 backgrounds 对齐的生成结果 url（失败/跳过为 null）
  current: 0,             // 当前处理到第几张（从 0 起）
  es: null,               // 当前的 EventSource
};

const MODEL_LABELS = {
  auto: "自动调度",
  bailian: "阿里百炼",
  siliconflow: "硅基流动",
};

const STEP_DEFS = [
  { key: "perceive", label: "感知", tag: "PERCEIVE", desc: "看图提取人物特征与场景信息" },
  { key: "plan", label: "规划", tag: "PLAN", desc: "自动撰写专业提示词" },
  { key: "generate", label: "出图", tag: "GENERATE", desc: "调用图像模型生成画面" },
];

const $ = (id) => document.getElementById(id);

/* ---------- 阶段切换 ---------- */
function showStage(id) {
  document.querySelectorAll(".stage").forEach((s) => s.classList.remove("active"));
  const el = $(id);
  el.classList.remove("active");
  void el.offsetWidth; // 强制重排，让入场动画每次都能重播
  el.classList.add("active");
  window.scrollTo({ top: 0 });
}

/* ==========================================================================
   阶段一 · 欢迎页
   ========================================================================== */
$("modelSwitch").addEventListener("click", (e) => {
  const btn = e.target.closest(".model-btn");
  if (!btn) return;
  document.querySelectorAll(".model-btn").forEach((b) => b.classList.remove("on"));
  btn.classList.add("on");
  state.model = btn.dataset.model;
});

$("btnLaunch").addEventListener("click", startScan);
$("btnBackHome").addEventListener("click", () => showStage("stage-welcome"));

/* ==========================================================================
   阶段二 · 读取素材
   ========================================================================== */
async function startScan() {
  showStage("stage-loading");
  $("loadingView").classList.remove("hidden");
  $("loadingError").classList.add("hidden");

  // 至少停留一小段时间，避免加载页一闪而过
  const minWait = new Promise((r) => setTimeout(r, 900));
  try {
    const resp = await fetch("/api/scan", { method: "POST" });
    const data = await resp.json();
    await minWait;
    if (!resp.ok || data.error) {
      return showScanError(data.error || "读取素材失败，请稍后重试。");
    }
    state.backgrounds = data.backgrounds;
    state.person = data.person;
    state.results = new Array(data.backgrounds.length).fill(null);
    state.current = 0;
    enterConsole();
  } catch (err) {
    await minWait;
    showScanError("无法连接后端服务：" + err.message);
  }
}

function showScanError(msg) {
  $("loadingView").classList.add("hidden");
  $("loadingError").classList.remove("hidden");
  $("loadingErrorText").textContent = msg;
}

/* ==========================================================================
   阶段三 · 生成控制台
   ========================================================================== */
function enterConsole() {
  $("timeline").innerHTML = "";
  $("chipModel").textContent = MODEL_LABELS[state.model] || state.model;
  updateProgress();
  showStage("stage-console");
  showAskCard();
}

function updateProgress() {
  const shown = Math.min(state.current + 1, state.backgrounds.length);
  $("chipProgress").textContent = `第 ${shown} / ${state.backgrounds.length} 张`;
}

/* ---- 任务类型询问卡片 ---- */
function showAskCard() {
  const i = state.current;
  const bg = state.backgrounds[i];
  $("askTitle").textContent = `第 ${i + 1} 张 · 选择任务类型`;
  $("askSub").textContent = `背景：${bg.name} × 人物：${state.person.name}`;
  // 展示本次待传的两张素材，方便用户确认
  $("askBgImg").src = bg.url;
  $("askBgName").textContent = bg.name;
  $("askPersonImg").src = state.person.url;
  $("askPersonName").textContent = state.person.name;
  $("askExtra").value = "";
  $("askCard").classList.remove("hidden");
  updateProgress();
  $("askCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

$("btnTaskReal").addEventListener("click", () => chooseTask("realistic_fusion", "真人融合"));
$("btnTaskToon").addEventListener("click", () => chooseTask("cartoonize", "卡通形象"));

function chooseTask(task, taskLabel) {
  $("askCard").classList.add("hidden");
  const extra = $("askExtra").value.trim();
  runJob(state.current, task, taskLabel, extra);
}

/* ---- 创建一张图的任务卡片（含三步时间线）---- */
function buildJobCard(index, taskLabel) {
  const bg = state.backgrounds[index];
  const card = document.createElement("div");
  card.className = "job-card";
  card.innerHTML = `
    <div class="job-head">
      <span class="job-title">第 ${index + 1} 张 · ${bg.name}</span>
      <span class="job-badge">${taskLabel} · 运行中</span>
    </div>
    <div class="steps">
      ${STEP_DEFS.map((s) => `
        <div class="step" data-step="${s.key}">
          <div class="step-node"></div>
          <div class="step-body">
            <div class="step-label">${s.label}<small>${s.tag}</small></div>
            <div class="step-summary">${s.desc}</div>
          </div>
        </div>`).join("")}
    </div>`;
  $("timeline").appendChild(card);
  card.scrollIntoView({ behavior: "smooth", block: "end" });
  return card;
}

/* ---- 执行一张图的生成（SSE）---- */
function runJob(index, task, taskLabel, extra) {
  const card = buildJobCard(index, taskLabel);
  const badge = card.querySelector(".job-badge");
  const params = new URLSearchParams({
    bg: state.backgrounds[index].id,
    task,
    model: state.model,
    request: extra,
  });

  let settled = false; // 收到 result / error 后置真，防止 EventSource 自动重连重复扣费
  const es = new EventSource(`/api/generate?${params}`);
  state.es = es;

  const finish = () => { settled = true; es.close(); state.es = null; };

  es.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }

    if (msg.type === "step") {
      const stepEl = card.querySelector(`.step[data-step="${msg.key}"]`);
      if (!stepEl) return;
      stepEl.classList.remove("running", "done");
      stepEl.classList.add(msg.status === "running" ? "running" : "done");
      stepEl.querySelector(".step-node").textContent =
        msg.status === "running" ? "" : "✓";
      if (msg.summary) {
        stepEl.querySelector(".step-summary").textContent = msg.summary;
      }
      return;
    }

    if (msg.type === "result") {
      finish();
      state.results[index] = msg.url;
      badge.textContent = `${taskLabel} · 完成`;
      badge.classList.add("ok");
      const box = document.createElement("div");
      box.className = "job-result";
      box.innerHTML = `
        <img class="job-thumb" src="${msg.url}" alt="生成结果" />
        <p class="job-result-note">已暂存第 ${index + 1} 张。${msg.status || ""}</p>`;
      card.appendChild(box);
      nextJob();
      return;
    }

    if (msg.type === "error") {
      finish();
      showJobError(card, badge, index, msg.message);
    }
  };

  es.onerror = () => {
    if (settled) return;
    finish();
    showJobError(card, badge, index, "与后端的连接中断，请检查服务是否仍在运行。");
  };
}

/* ---- 单张失败：允许重试这一张或跳过 ---- */
function showJobError(card, badge, index, message) {
  badge.textContent = "失败";
  badge.classList.add("bad");
  const box = document.createElement("div");
  box.className = "job-error";
  box.innerHTML = `
    <div>这一张生成失败：${message}</div>
    <div class="job-error-actions">
      <button class="btn-mini" data-act="retry">重试这一张</button>
      <button class="btn-mini" data-act="skip">跳过</button>
    </div>`;
  card.appendChild(box);
  box.addEventListener("click", (e) => {
    const act = e.target.dataset && e.target.dataset.act;
    if (act === "retry") {
      card.remove();       // 移除失败卡片，重新询问同一张
      showAskCard();
    } else if (act === "skip") {
      box.querySelector(".job-error-actions").remove();
      nextJob();
    }
  });
}

/* ---- 推进到下一张 / 收尾 ---- */
function nextJob() {
  state.current += 1;
  if (state.current < state.backgrounds.length) {
    showAskCard();
  } else {
    setTimeout(enterBook, 900); // 稍作停顿再进入绘本，节奏更从容
  }
}

/* ==========================================================================
   阶段四 · 儿童绘本（3D 翻页）
   ========================================================================== */
function pagePhotoHTML(url, pageNo) {
  return `
    <div class="photo-frame"><img src="${url}" alt="第 ${pageNo} 页" /></div>
    <div class="photo-caption">· 第 ${pageNo} 页 ·</div>`;
}

function buildBook() {
  const book = $("book");
  book.innerHTML = "";
  book.classList.remove("opened", "closed-end");

  // 只取真实生成成功的图，按张数动态排版，不留多余空白页：
  // 偶数张 → 内页纸 N/2 张，封底纸正面是 THE END 衬页；
  // 奇数张 → 最后一张照片直接排在封底纸正面，刚好铺满。
  const photos = state.results.filter(Boolean);
  const odd = photos.length % 2 === 1;
  const innerCount = Math.floor(photos.length / 2);

  const sheets = [
    {
      front: { cls: "page-cover", html: `
        <img class="page-art" src="/static/assets/cover.png" alt="封面"
             onerror="this.remove()" />
        <div class="cover-title">我的奇妙绘本</div>
        <div class="cover-sub">AI STORYBOOK</div>` },
      back: { cls: "page-lining", html: `<div class="lining-note">MY STORYBOOK</div>` },
    },
  ];
  for (let i = 0; i < innerCount; i++) {
    sheets.push({
      front: { cls: "page-photo", html: pagePhotoHTML(photos[2 * i], 2 * i + 1) },
      back: { cls: "page-photo", html: pagePhotoHTML(photos[2 * i + 1], 2 * i + 2) },
    });
  }
  sheets.push({
    front: odd
      ? { cls: "page-photo",
          html: pagePhotoHTML(photos[photos.length - 1], photos.length) }
      : { cls: "page-lining", html: `<div class="lining-note">THE END</div>` },
    back: { cls: "page-cover", html: `
        <img class="page-art" src="/static/assets/back_cover.png" alt="封底"
             onerror="this.remove()" />
        <div class="cover-sub">GOOD NIGHT · SEE YOU NEXT STORY</div>` },
  });

  const total = sheets.length;
  let flipped = 0; // 已翻过去的纸张数
  const Z_GAP = 1.2; // 每张纸的微小厚度（px），避免 3D 共面导致的画面穿透

  const shadow = document.createElement("div");
  shadow.className = "book-shadow";
  book.appendChild(shadow);

  const sheetEls = sheets.map((s, i) => {
    const el = document.createElement("div");
    el.className = "sheet";
    el.innerHTML = `
      <div class="page page-front ${s.front.cls}">${s.front.html}</div>
      <div class="page page-back ${s.back.cls}">${s.back.html}</div>`;
    book.appendChild(el);
    return el;
  });

  // 用 rotateY + translateZ 表达每张纸的翻转与叠放顺序：
  // 未翻时靠前的纸 translateZ 更大（在上）；翻过去后旋转把局部 z 轴反向，
  // 同一份偏移自然形成左侧“后翻的在上”的正确叠序。
  function applyTransforms() {
    sheetEls.forEach((el, i) => {
      const deg = el.classList.contains("flipped") ? -180 : 0;
      el.style.transform =
        `rotateY(${deg}deg) translateZ(${(total - i) * Z_GAP}px)`;
    });
  }

  function updateBookPose() {
    book.classList.toggle("opened", flipped > 0 && flipped < total);
    book.classList.toggle("closed-end", flipped === total);
  }

  // preserve-3d 下浏览器对书页的命中测试不可靠（点封面可能命中底层纸），
  // 改用两个悬浮在最前方的透明点击区：右区向后翻，左区向前翻。
  const zoneLeft = document.createElement("div");
  zoneLeft.className = "flip-zone zone-left";
  const zoneRight = document.createElement("div");
  zoneRight.className = "flip-zone zone-right";
  book.append(zoneLeft, zoneRight);

  function updateZones() {
    zoneRight.classList.toggle("idle", flipped >= total);
    zoneLeft.classList.toggle("idle", flipped <= 0);
  }

  zoneRight.addEventListener("click", () => {
    if (flipped >= total) return;
    sheetEls[flipped].classList.add("flipped");
    flipped += 1;
    applyTransforms();
    updateBookPose();
    updateZones();
  });

  zoneLeft.addEventListener("click", () => {
    if (flipped <= 0) return;
    flipped -= 1;
    sheetEls[flipped].classList.remove("flipped");
    applyTransforms();
    updateBookPose();
    updateZones();
  });

  applyTransforms();
  updateZones();
}

function enterBook() {
  buildBook();
  showStage("stage-book");
}

/* ---- 重新生成：清空本轮内容，回到欢迎页 ---- */
$("btnRegen").addEventListener("click", () => {
  if (state.es) { state.es.close(); state.es = null; }
  state.backgrounds = [];
  state.person = null;
  state.results = [];
  state.current = 0;
  $("timeline").innerHTML = "";
  $("book").innerHTML = "";
  showStage("stage-welcome");
});
