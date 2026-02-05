const thread = document.getElementById("thread");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const newChatBtn = document.getElementById("newChatBtn");
const menuBtn = document.getElementById("menuBtn");
const sidebar = document.querySelector(".sidebar");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statusHelp = document.getElementById("statusHelp");
const statusLine = document.getElementById("statusLine");
const pillReady = document.getElementById("pillReady");
const scrollFab = document.getElementById("scrollFab");
const scrollBtn = document.getElementById("scrollBtn");

const STORAGE_KEY = "rag_na_biblia.thread.v1";

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
  avatar.textContent = role === "user" ? "V" : "R";
  avatar.setAttribute("aria-hidden", "true");

  const body = document.createElement("div");

  const metaEl = document.createElement("div");
  metaEl.className = "meta";
  const left = document.createElement("div");
  left.innerHTML = `<strong>${role === "user" ? "Você" : "Rag na Bíblia"}</strong>${meta ? ` • ${escapeHtml(meta)}` : ""}${cached ? " • cache" : ""}`;

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
  addMessage("assistant", "Faça uma pergunta sobre a Bíblia. Eu respondo usando RAG (Chroma + LLM).");
  scrollToBottom();
  saveThread();
  input.focus();
}

async function ask(message) {
  const res = await fetch("/api/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data?.detail ? ` (${data.detail})` : "";
    throw new Error((data?.error || "erro") + detail);
  }
  return { answer: data.answer ?? "", cached: !!data.cached };
}

function setStatus({ ready, reasons, config }) {
  statusHelp.textContent = "";

  if (ready) {
    statusDot.className = "dot dot-ok";
    statusText.textContent = "Pronto";
    statusLine.textContent = "Pronto para perguntas";
    pillReady.textContent = "Pronto";
  } else {
    statusDot.className = "dot dot-bad";
    statusText.textContent = "Não pronto";
    statusLine.textContent = "Requer configuração";
    pillReady.textContent = "Atenção";
    if (Array.isArray(reasons) && reasons.length) {
      const mapping = {
        missing_chroma_db: "Banco Chroma não encontrado. Rode a ingestão.",
        missing_google_api_key: "Falta GOOGLE_API_KEY no ambiente (.env).",
      };
      statusHelp.textContent = reasons.map((r) => mapping[r] || r).join(" ");
    }
  }
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    setStatus(data);
  } catch {
    statusDot.className = "dot dot-warn";
    statusText.textContent = "Sem conexão";
    statusHelp.textContent = "Servidor não respondeu em /api/health.";
    statusLine.textContent = "Offline";
    pillReady.textContent = "Offline";
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
    typing.msg.querySelector(".meta strong").textContent = "Rag na Bíblia";
    if (result.cached) {
      const left = typing.msg.querySelector(".meta > div");
      if (left) left.innerHTML = `<strong>Rag na Bíblia</strong> • cache`;
    }
    // habilita copiar
    const copyBtn = typing.msg.querySelector(".actions .iconbtn");
    if (copyBtn) copyBtn.style.display = "inline-flex";
    saveThread();
  } catch (err) {
    typing.bubble.textContent = `Erro ao responder: ${err?.message || err}`;
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
