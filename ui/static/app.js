const thread = document.getElementById("thread");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const newChatBtn = document.getElementById("newChatBtn");
const menuBtn = document.getElementById("menuBtn");
const sidebar = document.querySelector(".sidebar");
const statusLine = document.getElementById("statusLine");
const pillReady = document.getElementById("pillReady");
const scrollFab = document.getElementById("scrollFab");
const scrollBtn = document.getElementById("scrollBtn");
const accessBox = document.getElementById("accessBox");
const demoTokenInput = document.getElementById("demoToken");
const saveTokenBtn = document.getElementById("saveTokenBtn");
const accessMsg = document.getElementById("accessMsg");
const limitsBox = document.getElementById("limitsBox");
const limitsText = document.getElementById("limitsText");

const STORAGE_KEY = "rag_na_biblia.thread.v1";
const ASSISTANT_NAME = "Agente Bíblia RAG";
const DEMO_TOKEN_KEY = "rag_na_biblia.demo_token.v1";

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(text) {
  const escaped = escapeHtml(text ?? "");
  const parts = escaped.split(/```/);
  let out = "";

  for (let i = 0; i < parts.length; i++) {
    const chunk = parts[i];
    if (i % 2 === 1) {
      const lines = chunk.split("\n");
      let lang = "";
      if (lines.length > 1 && /^[a-z0-9#+.\-]{1,20}$/i.test(lines[0].trim())) {
        lang = lines[0].trim();
        lines.shift();
      }
      const code = lines.join("\n").replace(/\n$/, "");
      out += `<pre><code data-lang="${lang}">${code}</code></pre>`;
      continue;
    }
    let html = chunk;
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    html = html.replace(/\n/g, "<br>");
    out += html;
  }
  return out;
}

function container() {
  let el = thread.querySelector(".container");
  if (!el) {
    el = document.createElement("div");
    el.className = "container";
    thread.appendChild(el);
  }
  return el;
}

function addMessage(role, text, { meta = "", typing = false, cached = false } = {}) {
  const msg = document.createElement("div");
  msg.className = `msg ${role === "user" ? "user" : "assistant"}`;

  const row = document.createElement("div");
  row.className = "row";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "V" : "A";
  avatar.setAttribute("aria-hidden", "true");

  const body = document.createElement("div");

  const metaEl = document.createElement("div");
  metaEl.className = "meta";
  const left = document.createElement("div");
  left.innerHTML = `<strong>${role === "user" ? "Você" : ASSISTANT_NAME}</strong>${meta ? ` • ${escapeHtml(meta)}` : ""}${cached ? " • cache" : ""}`;

  const actions = document.createElement("div");
  actions.className = "actions";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "iconbtn";
  copyBtn.textContent = "Copiar";
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(String(text ?? ""));
      copyBtn.textContent = "Copiado";
      setTimeout(() => (copyBtn.textContent = "Copiar"), 900);
    } catch {
      copyBtn.textContent = "Falhou";
      setTimeout(() => (copyBtn.textContent = "Copiar"), 900);
    }
  });

  if (role !== "user" && !typing) actions.appendChild(copyBtn);

  metaEl.appendChild(left);
  metaEl.appendChild(actions);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (typing) {
    bubble.innerHTML = `<span class="typing">Pensando <span class="dots"><span></span><span></span><span></span></span></span>`;
  } else {
    bubble.innerHTML = renderMarkdown(text);
  }

  body.appendChild(metaEl);
  body.appendChild(bubble);

  row.appendChild(avatar);
  row.appendChild(body);
  msg.appendChild(row);

  container().appendChild(msg);
  return { msg, bubble };
}

function scrollToBottom() {
  thread.scrollTop = thread.scrollHeight;
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  input.disabled = busy;
  input.placeholder = busy ? "Aguardando resposta…" : "Pergunte algo…";
}

function autosize() {
  input.style.height = "0px";
  const h = Math.min(input.scrollHeight, 180);
  input.style.height = `${Math.max(h, 46)}px`;
}

function saveThread() {
  const items = [...container().querySelectorAll(".msg")].map((m) => {
    const role = m.classList.contains("user") ? "user" : "assistant";
    const bubble = m.querySelector(".bubble");
    return { role, text: bubble ? bubble.innerText : "" };
  });
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

function loadThread() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return false;
  try {
    const items = JSON.parse(raw);
    if (!Array.isArray(items) || items.length === 0) return false;
    for (const it of items) {
      if (!it?.role) continue;
      addMessage(it.role === "user" ? "user" : "assistant", String(it.text ?? ""));
    }
    return true;
  } catch {
    return false;
  }
}

function resetThread() {
  container().innerHTML = "";
  addMessage(
    "assistant",
    "Faça uma pergunta sobre a Bíblia. Eu respondo como um agente RAG (Chroma + Cohere Embeddings + Groq) e cito versículos quando possível."
  );
  scrollToBottom();
  saveThread();
  input.focus();
}

async function ask(message) {
  const token = localStorage.getItem(DEMO_TOKEN_KEY) || "";
  const res = await fetch("/api/answer", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { "X-Demo-Token": token } : {}),
    },
    body: JSON.stringify({ message }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data?.detail ? ` (${data.detail})` : "";
    const err = new Error((data?.error || "erro") + detail);
    err.data = data;
    err.status = res.status;
    throw err;
  }
  return { answer: data.answer ?? "", cached: !!data.cached };
}

function setReadyPill({ text, kind }) {
  pillReady.textContent = text;
  pillReady.classList.remove("pill-ok", "pill-warn", "pill-bad");
  if (kind) pillReady.classList.add(kind);
}

function setStatus({ ready, reasons }) {
  if (ready) {
    statusLine.textContent = "Agente pronto para perguntas";
    setReadyPill({ text: "Agente pronto", kind: "pill-ok" });
    return;
  }

  statusLine.textContent = "Agente precisa de configuração";
  setReadyPill({ text: "Configurar", kind: "pill-warn" });

  if (Array.isArray(reasons) && reasons.length) {
    const mapping = {
      missing_chroma_db: "Rode a ingestão (ingestion/ingest.py).",
      missing_cohere_api_key: "Falta COHERE_API_KEY no .env.",
      missing_groq_api_key: "Falta GROQ_API_KEY no .env.",
    };
    const hint = reasons.map((r) => mapping[r] || r).join(" ");
    statusLine.textContent = `Agente precisa de configuração • ${hint}`;
  }
}

function initAccessBox(health) {
  accessMsg.textContent = "";
  const requiresToken = !!health?.demo?.requires_token;
  accessBox.hidden = !requiresToken;
  if (!requiresToken) return;

  const stored = localStorage.getItem(DEMO_TOKEN_KEY) || "";
  demoTokenInput.value = stored;
  if (stored) accessMsg.textContent = "Chave salva neste navegador.";
}

function initLimitsBox(health) {
  const demo = health?.demo;
  const enabled = !!demo?.enabled;
  limitsBox.hidden = !enabled;
  if (!enabled) return;

  const parts = [];
  if (Number.isFinite(demo?.per_ip_per_day) && demo.per_ip_per_day > 0) parts.push(`por IP: ${demo.per_ip_per_day}/dia`);
  if (Number.isFinite(demo?.total_per_day) && demo.total_per_day > 0) parts.push(`total: ${demo.total_per_day}/dia`);
  if (Number.isFinite(demo?.cooldown_s) && demo.cooldown_s > 0) parts.push(`intervalo: ${demo.cooldown_s}s`);
  if (Number.isFinite(demo?.max_chars) && demo.max_chars > 0) parts.push(`máx. ${demo.max_chars} caracteres por pergunta`);

  limitsText.textContent =
    parts.length > 0
      ? `Para proteger a cota das APIs, esta demo tem limites (${parts.join(" • ")}).`
      : "Para proteger a cota das APIs, esta demo tem limites.";
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    setStatus(data);
    initAccessBox(data);
    initLimitsBox(data);
  } catch {
    statusLine.textContent = "Servidor offline";
    setReadyPill({ text: "Offline", kind: "pill-bad" });
  }
}

// UI events
document.querySelectorAll(".chip").forEach((b) => {
  b.addEventListener("click", () => {
    input.value = b.getAttribute("data-prompt") || "";
    autosize();
    input.focus();
    if (sidebar) sidebar.classList.remove("open");
  });
});

menuBtn?.addEventListener("click", () => sidebar?.classList.toggle("open"));
thread.addEventListener("click", () => sidebar?.classList.remove("open"));

newChatBtn.addEventListener("click", () => resetThread());
clearBtn.addEventListener("click", () => resetThread());

saveTokenBtn?.addEventListener("click", () => {
  const token = String(demoTokenInput?.value || "").trim();
  if (!token) {
    localStorage.removeItem(DEMO_TOKEN_KEY);
    accessMsg.textContent = "Chave removida.";
    return;
  }
  localStorage.setItem(DEMO_TOKEN_KEY, token);
  accessMsg.textContent = "Chave salva.";
});

input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

thread.addEventListener("scroll", () => {
  const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 220;
  scrollFab.hidden = nearBottom;
});

scrollBtn.addEventListener("click", () => scrollToBottom());

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage("user", message);
  input.value = "";
  autosize();
  saveThread();

  setBusy(true);
  const typing = addMessage("assistant", "", { typing: true, meta: "respondendo…" });

  try {
    const result = await ask(message);
    typing.bubble.innerHTML = renderMarkdown(result.answer || "(sem resposta)");
    typing.msg.querySelector(".meta strong").textContent = ASSISTANT_NAME;
    if (result.cached) {
      const left = typing.msg.querySelector(".meta > div");
      if (left) left.innerHTML = `<strong>${ASSISTANT_NAME}</strong> • cache`;
    }
    // habilita copiar
    const copyBtn = typing.msg.querySelector(".actions .iconbtn");
    if (copyBtn) copyBtn.style.display = "inline-flex";
    saveThread();
  } catch (err) {
    const msg = String(err?.message || err);
    const data = err?.data || {};
    const status = err?.status || 0;
    if (msg.includes("unauthorized")) {
      accessMsg.textContent = "Chave inválida. Atualize em Acesso.";
      if (accessBox) accessBox.hidden = false;
    }

    if (status === 429 || msg.includes("rate_limited") || msg.includes("cooldown")) {
      const retry = data?.retry_after_s ? ` Aguarde ${data.retry_after_s}s e tente de novo.` : "";
      typing.bubble.textContent =
        "Limite atingido nesta demo para proteger a cota das APIs." + retry;
    } else if (msg.includes("message_too_long")) {
      const max = data?.max_chars ? ` (máx. ${data.max_chars} caracteres)` : "";
      typing.bubble.textContent = "Pergunta muito longa" + max + ".";
    } else {
      typing.bubble.textContent = `Erro ao responder: ${msg}`;
    }
  } finally {
    setBusy(false);
    input.focus();
    scrollToBottom();
  }
});

// Boot
window.addEventListener("load", async () => {
  autosize();
  const loaded = loadThread();
  if (!loaded) resetThread();
  await refreshHealth();
  setInterval(refreshHealth, 15000);
  input.focus();
  scrollToBottom();
});
