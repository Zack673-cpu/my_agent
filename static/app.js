/* ==========================================================================
   AI 绘本工坊 · 前端逻辑
   四个阶段：欢迎页 → 读取素材 → 生成控制台（SSE 时间线）→ 3D 翻页绘本
   界面附加：顶部 HUD 步进、机器人互动（眼动/彩蛋）、成书暖光切换
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
  {
    key: "perceive", label: "感知", tag: "PERCEIVE", desc: "看图提取人物特征与场景信息",
    icon: `<svg class="step-icon" viewBox="0 0 18 18" aria-hidden="true"><path d="M2 9c3.4-4.4 10.6-4.4 14 0-3.4 4.4-10.6 4.4-14 0Z"/><circle cx="9" cy="9" r="2.4"/></svg>`,
  },
  {
    key: "plan", label: "规划", tag: "PLAN", desc: "自动撰写专业提示词",
    icon: `<svg class="step-icon" viewBox="0 0 18 18" aria-hidden="true"><path d="m3.2 14.8 1.3-4.3 7.5-7.5a2 2 0 0 1 2.8 2.8l-7.5 7.5-4.1 1.5Z"/><path d="m10.4 4.6 2.8 2.8"/></svg>`,
  },
  {
    key: "generate", label: "出图", tag: "GENERATE", desc: "调用图像模型生成画面",
    icon: `<svg class="step-icon" viewBox="0 0 18 18" aria-hidden="true"><rect x="2.4" y="3.6" width="13.2" height="10.8" rx="2"/><circle cx="6.8" cy="8" r="1.4"/><path d="m4 13.4 3.4-3.2 2.4 2.2 2.6-2.6 2.6 2.6"/></svg>`,
  },
];

/* 步骤完成时画在菱形节点里的自绘对勾 */
const CHECK_SVG = `<svg class="node-check" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.6l4.2 4.2L19 7.4"/></svg>`;

const $ = (id) => document.getElementById(id);

/* ---------- 阶段切换 ---------- */
const STAGE_ORDER = ["stage-welcome", "stage-loading", "stage-console", "stage-book"];

function showStage(id) {
  document.querySelectorAll(".stage").forEach((s) => s.classList.remove("active"));
  const el = $(id);
  el.classList.remove("active");
  void el.offsetWidth; // 强制重排，让入场动画每次都能重播
  el.classList.add("active");
  // 成书阶段：整个世界从深空实验室切换到暖光书房
  document.body.classList.toggle("book-mode", id === "stage-book");
  updateHudSteps(id);
  window.scrollTo({ top: 0 });
}

/* 顶部 HUD 步进：当前步骤点亮，已走过的步骤弱亮 */
function updateHudSteps(activeId) {
  const idx = STAGE_ORDER.indexOf(activeId);
  document.querySelectorAll("#hudSteps .hstep").forEach((el, i) => {
    el.classList.toggle("on", i === idx);
    el.classList.toggle("done", i < idx);
    if (i === idx) el.setAttribute("aria-current", "step");
    else el.removeAttribute("aria-current");
  });
}

updateHudSteps("stage-welcome"); // 首屏：点亮第 1 步

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

/* ---- 机器人互动：瞳孔跟随鼠标（仅悬停设备）、点击跳一下（隐藏彩蛋） ---- */
(function initHeroBot() {
  const hero = $("heroBot");
  if (!hero) return;
  const pupils = hero.querySelectorAll(".bot-pupil");

  hero.addEventListener("click", () => {
    hero.classList.remove("hop");
    void hero.offsetWidth; // 重排后再加类，连点也能重播
    hero.classList.add("hop");
  });
  hero.addEventListener("animationend", (e) => {
    if (e.animationName === "bot-hop") hero.classList.remove("hop");
  });

  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hover = window.matchMedia("(hover: hover)").matches;
  if (reduce || !hover || !pupils.length) return;

  let raf = null;
  window.addEventListener("pointermove", (e) => {
    if (raf) return; // 每帧最多更新一次，避免频繁触发布局计算
    raf = requestAnimationFrame(() => {
      raf = null;
      const rect = hero.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height * 0.32; // 面罩大致位置
      const dx = e.clientX - cx, dy = e.clientY - cy;
      const dist = Math.hypot(dx, dy) || 1;
      const r = Math.min(7, dist / 24);
      const tx = ((dx / dist) * r).toFixed(2);
      const ty = ((dy / dist) * r).toFixed(2);
      pupils.forEach((p) => { p.style.transform = `translate(${tx}px, ${ty}px)`; });
    });
  }, { passive: true });
})();

/* ==========================================================================
   阶段二 · 读取素材
   ========================================================================== */
/* 扫描期间滚动的终端日志文案（纯装饰，与后端请求并行） */
const SCAN_LOG_LINES = [
  "打开桌面 · 寻找「背景图」文件夹",
  "打开桌面 · 寻找「人物」文件夹",
  "登记素材清单",
];

async function startScan() {
  showStage("stage-loading");
  $("loadingView").classList.remove("hidden");
  $("loadingError").classList.add("hidden");

  // 终端日志：每 700ms 追加一行，最多保留 3 行
  const logEl = $("scanLog");
  logEl.innerHTML = "";
  let logIdx = 0;
  const pushLog = () => {
    const row = document.createElement("li");
    row.textContent = SCAN_LOG_LINES[logIdx % SCAN_LOG_LINES.length];
    logIdx += 1;
    logEl.appendChild(row);
    if (logEl.children.length > 3) logEl.firstElementChild.remove();
  };
  pushLog();
  const logTimer = setInterval(pushLog, 700);

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
  } finally {
    clearInterval(logTimer);
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
  $("bookCta").classList.add("hidden");
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
  card.className = "job-card running";
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
            <div class="step-label">${s.icon}${s.label}<small>${s.tag}</small></div>
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
      // 完成时在菱形节点里画一个自绘对勾，运行中清空
      stepEl.querySelector(".step-node").innerHTML =
        msg.status === "running" ? "" : CHECK_SVG;
      if (msg.summary) {
        stepEl.querySelector(".step-summary").textContent = msg.summary;
      }
      return;
    }

    if (msg.type === "result") {
      finish();
      card.classList.replace("running", "done");
      state.results[index] = msg.url;
      badge.textContent = `${taskLabel} · 完成`;
      badge.classList.add("ok");
      const box = document.createElement("div");
      box.className = "job-result";
      box.innerHTML = `
        <img class="job-thumb" src="${msg.url}" alt="第 ${index + 1} 张生成结果"
             role="button" tabindex="0" title="点击放大查看"
             aria-label="放大查看第 ${index + 1} 张生成结果" />
        <p class="job-result-note">已暂存第 ${index + 1} 张 · 点击图片可放大查看。${msg.status || ""}</p>`;
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
  card.classList.remove("running");
  card.classList.add("failed");
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
    showBookCta(); // 最后一张生成完不再自动进绘本，让用户先看清结果
  }
}

/* ---- 全部处理完毕：展示装订入口，由用户点按钮再生成绘本 ---- */
function showBookCta() {
  const ok = state.results.filter(Boolean).length;
  $("bookCtaCount").textContent = ok > 0
    ? `已完成 ${ok} 张 · 点击下方按钮装订成绘本`
    : "还没有生成成功的图片";
  $("btnMakeBook").disabled = ok === 0;
  $("bookCta").classList.remove("hidden");
  $("bookCta").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

$("btnMakeBook").addEventListener("click", () => {
  $("bookCta").classList.add("hidden");
  enterBook();
});

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

  // 键盘可达：翻页区可用 Enter / 空格触发
  [[zoneLeft, "向前翻一页"], [zoneRight, "向后翻一页"]].forEach(([zone, label]) => {
    zone.setAttribute("role", "button");
    zone.setAttribute("aria-label", label);
    zone.tabIndex = 0;
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        zone.click();
      }
    });
  });

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
  $("bookCta").classList.add("hidden");
  showStage("stage-welcome");
});

/* ==========================================================================
   图片放大查看：点生成结果缩略图打开，点空白处 / 按 Esc 关闭
   ========================================================================== */
(function initLightbox() {
  const box = $("lightbox");
  const img = $("lightboxImg");
  if (!box || !img) return;

  const close = () => {
    box.classList.add("hidden");
    img.removeAttribute("src");
    document.body.style.overflow = "";
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (e) => {
    if (e.key === "Escape") close();
  };
  const open = (src, alt) => {
    img.src = src;
    img.alt = alt || "生成结果放大图";
    box.classList.remove("hidden");
    document.body.style.overflow = "hidden"; // 打开时禁止背景滚动
    document.addEventListener("keydown", onKey);
  };

  // 点空白处关闭，点图片本身不关闭
  box.addEventListener("click", (e) => {
    if (e.target === img) return;
    close();
  });
  $("lightboxClose").addEventListener("click", close);

  // 事件委托：所有生成结果缩略图都可点击放大（键盘 Enter / 空格同样可用）
  document.addEventListener("click", (e) => {
    const thumb = e.target.closest(".job-thumb");
    if (thumb) open(thumb.src, thumb.alt);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const thumb = e.target.closest && e.target.closest(".job-thumb");
    if (!thumb) return;
    e.preventDefault();
    open(thumb.src, thumb.alt);
  });
})();

/* ==========================================================================
   全局背景特效 · 星链粒子（Canvas 手绘，零依赖）
   粒子缓慢漂浮，距离足够近时互相连线；光标靠近时也会与粒子连成网络。
   ========================================================================== */
(function initStarfield() {
  const canvas = $("fxCanvas");
  if (!canvas) return;
  // 尊重系统「减少动态效果」设置：直接不启动
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const ctx = canvas.getContext("2d");
  const DPR = Math.min(window.devicePixelRatio || 1, 1.5);
  // 粒子配色：亮金 / 奶油金 / 铜金，保证在暖黑背景上清晰可辨
  const COLORS = ["255, 214, 130", "255, 236, 190", "240, 168, 46"];
  const LINK_DIST = 138;   // 粒子互连距离
  const CURSOR_DIST = 160; // 光标连线距离
  const cursor = { x: -9999, y: -9999 };
  let W = 0, H = 0, particles = [];

  function spawn() {
    // 屏幕越大粒子越多，上限 96 个，保证满帧流畅；密度抬高后星座网络始终清晰
    const count = Math.min(96, Math.round((W * H) / 16000));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      r: 1.2 + Math.random() * 1.3,
      c: COLORS[(Math.random() * COLORS.length) | 0],
      a: 0.55 + Math.random() * 0.4,
    }));
  }

  function resize() {
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    spawn();
  }

  function frame() {
    ctx.clearRect(0, 0, W, H);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      // 飘出边缘后从另一侧绕回
      if (p.x < -20) p.x = W + 20; else if (p.x > W + 20) p.x = -20;
      if (p.y < -20) p.y = H + 20; else if (p.y > H + 20) p.y = -20;
    }
    // 粒子之间连线
    ctx.lineWidth = 1;
    for (let i = 0; i < particles.length; i++) {
      const a = particles[i];
      for (let j = i + 1; j < particles.length; j++) {
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > LINK_DIST * LINK_DIST) continue;
        const t = 1 - Math.sqrt(d2) / LINK_DIST;
        ctx.strokeStyle = `rgba(240, 168, 46, ${(t * 0.22).toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
    // 粒子本体，以及与光标之间的连线
    for (const p of particles) {
      const dx = p.x - cursor.x;
      const dy = p.y - cursor.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < CURSOR_DIST * CURSOR_DIST) {
        const t = 1 - Math.sqrt(d2) / CURSOR_DIST;
        ctx.strokeStyle = `rgba(255, 214, 130, ${(t * 0.38).toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(cursor.x, cursor.y);
        ctx.stroke();
      }
      // 光晕（大而淡）+ 亮核（小而亮），让星座在暖黑背景上更清晰
      ctx.fillStyle = `rgba(${p.c}, ${(p.a * 0.22).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 2.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `rgba(${p.c}, ${p.a.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
    requestAnimationFrame(frame);
  }

  window.addEventListener("mousemove", (e) => {
    cursor.x = e.clientX;
    cursor.y = e.clientY;
  });
  document.addEventListener("mouseleave", () => {
    cursor.x = -9999;
    cursor.y = -9999;
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 160);
  });

  resize();
  requestAnimationFrame(frame);
})();
