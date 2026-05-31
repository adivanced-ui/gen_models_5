"""
04_rag_app.py — RAG как веб-приложение
═══════════════════════════════════════════════════════════════════
Финальный шаг практики: превращаем RAG-пайплайн в полноценное
веб-приложение со встроенным фронтендом.

Что есть:
  • Drag-and-drop загрузка .txt файлов в браузере.
  • Чанкинг и индексирование на бэкенде, прогресс в UI.
  • Чат-интерфейс с историей вопросов.
  • Прозрачность: для каждого ответа видно КАКИЕ фрагменты
    документа модель использовала (раскрывающийся блок).
  • Список загруженных источников сбоку.

Endpoints:
  GET  /            — HTML+JS фронтенд
  POST /upload      — загрузка файла, чанкинг, индексирование
  POST /ask         — JSON {question} → JSON {answer, sources}
  GET  /sources     — список загруженных документов
  POST /clear       — очистить базу

Требования:
  - LM Studio запущен (порт 1234)
  - Загружены: embedding-модель + LLM
  - pip install flask openai chromadb

Запуск: python 04_rag_app.py  →  http://localhost:5011
"""

import os
import re
from datetime import date
import chromadb
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

# ********************* КОНФИГУРАЦИЯ *********************

LM_STUDIO_URL   = "http://127.0.0.1:1234/v1"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
LLM_MODEL       = "google/gemma-4-e4b"

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_app_db")
COLLECTION = "rag_app"

CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_PARAGRAPHS = 1
TOP_K         = 3

MAX_TOKENS  = 600
TEMPERATURE = 0.3

PORT = 5011


# ********************* ИНИЦИАЛИЗАЦИЯ *********************

app    = Flask(__name__)
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection(
    name=COLLECTION, metadata={"hnsw:space": "cosine"}
)


# ********************* ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ *********************

def chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_paragraphs: int = CHUNK_OVERLAP_PARAGRAPHS,
) -> list[dict]:
    """Чанкинг по абзацам"""
    chunks = []
    pages = text.replace("\r\n", "\n").replace("\r", "\n").split("\f")

    for page_number, page_text in enumerate(pages, start=1):
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", page_text)
            if paragraph.strip()
        ]
        i = 0
        while i < len(paragraphs):
            start = i
            current = [paragraphs[i]]
            i += 1

            while i < len(paragraphs):
                candidate = current + [paragraphs[i]]
                if len("\n\n".join(candidate)) > max_chars:
                    break
                current.append(paragraphs[i])
                i += 1

            chunk = "\n\n".join(current).strip()
            chunks.append({
                "text": chunk,
                "page": page_number,
                "paragraph_start": start + 1,
                "paragraph_end": i,
                "length": len(chunk),
            })

            if overlap_paragraphs > 0 and i < len(paragraphs):
                i = max(i - overlap_paragraphs, start + 1)

    return chunks


def format_pages(pages: set[int]) -> str:
    """Компактно показывает страницы."""
    if not pages:
        return "—"

    sorted_pages = sorted(pages)
    ranges = []
    start = prev = sorted_pages[0]
    for page in sorted_pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = page
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Батчевый эмбеддинг через LM Studio."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_one(text: str) -> list[float]:
    """Эмбеддинг одного текста."""
    return embed_batch([text])[0]


# ********************* МАРШРУТЫ API *********************

@app.route("/")
def index():
    """Отдаём HTML-страницу со встроенным фронтендом."""
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    """
    Загрузка файла. Принимаем multipart/form-data с полем 'file'.
    Возвращаем JSON: {source, chunks, total_in_db}
    """
    if "file" not in request.files:
        return jsonify({"error": "Нет файла"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith((".txt", ".md")):
        return jsonify({"error": "Поддерживаются только .txt и .md"}), 400

    # Читаем содержимое — ожидаем UTF-8
    try:
        text = f.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "Файл должен быть в кодировке UTF-8"}), 400

    source = f.filename

    # Чанкинг по абзацам. 
    chunk_records = chunk_text(text)
    if not chunk_records:
        return jsonify({"error": "Файл пустой"}), 400
    chunks = [record["text"] for record in chunk_records]
    indexed_date = date.today().isoformat()

    # Эмбеддинги (батчем — быстрее)
    embeddings = embed_batch(chunks)

    # Сохраняем в Chroma. Перед добавлением — удаляем старые записи этого
    # источника (если файл загружают повторно, чтобы не было дублей).
    existing = collection.get(where={"source": source})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = [{
        "source": source,
        "chunk_index": i,
        "length": record["length"],
        "page": record["page"],
        "date": indexed_date,
        "paragraph_start": record["paragraph_start"],
        "paragraph_end": record["paragraph_end"],
    } for i, record in enumerate(chunk_records)]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return jsonify({
        "source": source,
        "chunks": len(chunks),
        "date": indexed_date,
        "total_in_db": collection.count(),
    })


PROMPT_TEMPLATE = """Ты — ассистент, отвечающий на вопросы СТРОГО на основе предоставленного контекста.
Если в контексте нет ответа — честно скажи «В предоставленных документах ответа нет».
Не выдумывай факты. Не используй внешние знания.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""


@app.route("/ask", methods=["POST"])
def ask():
    """
    Ответ на вопрос. Принимаем JSON {question}, возвращаем
    {answer, sources: [{source, chunk_index, similarity, text}]}
    """
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Пустой вопрос"}), 400

    if collection.count() == 0:
        return jsonify({"error": "База пуста — загрузите хотя бы один документ"}), 400

    # 1. Эмбеддинг вопроса
    q_vec = embed_one(question)

    # 2. Поиск top-K в Chroma
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=TOP_K,
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # 3. Сборка контекста для промпта
    context_parts = []
    sources_for_ui = []
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        similarity = 1.0 - dist     # cosine: чем меньше distance, тем больше сходство
        page = meta.get("page", "—")
        indexed_date = meta.get("date", "—")
        context_parts.append(
            f"[Источник: {meta['source']}, страница {page}, дата {indexed_date}, "
            f"фрагмент #{meta['chunk_index']}]\n{doc}"
        )
        sources_for_ui.append({
            "rank": rank,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "page": page,
            "date": indexed_date,
            "similarity": round(similarity, 3),
            "text": doc,
        })
    context = "\n\n---\n\n".join(context_parts)

    # 4. Промпт + LLM
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    msg = resp.choices[0].message
    answer = msg.content or getattr(msg, "reasoning_content", None) or "[пустой ответ]"

    return jsonify({"answer": answer, "sources": sources_for_ui})


@app.route("/sources", methods=["GET"])
def sources():
    """Список загруженных документов с числом чанков."""
    n = collection.count()
    if n == 0:
        return jsonify({"sources": [], "total": 0})

    # peek забирает все записи без поиска (limit=n)
    sample = collection.peek(limit=n)
    stats = {}
    for meta in sample["metadatas"]:
        source = meta["source"]
        item = stats.setdefault(source, {
            "name": source,
            "chunks": 0,
            "pages": set(),
            "dates": set(),
        })
        item["chunks"] += 1
        if meta.get("page") is not None:
            try:
                item["pages"].add(int(meta["page"]))
            except (TypeError, ValueError):
                pass
        if meta.get("date"):
            item["dates"].add(str(meta["date"]))

    sources_payload = []
    for item in stats.values():
        sources_payload.append({
            "name": item["name"],
            "chunks": item["chunks"],
            "pages": format_pages(item["pages"]),
            "date": ", ".join(sorted(item["dates"])) if item["dates"] else "—",
        })

    return jsonify({
        "sources": sorted(sources_payload, key=lambda x: x["name"].lower()),
        "total": n,
    })


@app.route("/clear", methods=["POST"])
def clear():
    """Очистка БД — пересоздаём коллекцию."""
    global collection
    chroma.delete_collection(COLLECTION)
    collection = chroma.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    return jsonify({"status": "ok"})


# ********************* HTML / JS ФРОНТЕНД *********************

HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>RAG App — лекция 11</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {
    --bg:#f1f5f9; --card:#fff; --border:#e2e8f0;
    --text:#1e293b; --muted:#64748b;
    --primary:#3b82f6; --green:#22c55e; --purple:#8b5cf6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  header { background: var(--card); border-bottom: 1px solid var(--border); padding: 14px 24px;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 700; flex: 1; }
  .badge { font-size: 12px; color: var(--muted); padding: 3px 10px; border-radius: 20px; background: #f1f5f9; }

  .layout { max-width: 1100px; margin: 0 auto; padding: 18px 16px;
            display: grid; grid-template-columns: 1fr 280px; gap: 16px; }

  .panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .panel h2 { font-size: 13px; font-weight: 700; color: var(--muted);
              text-transform: uppercase; letter-spacing: .4px; margin-bottom: 10px; }

  /* Зона drag-and-drop */
  .dropzone { border: 2px dashed var(--border); border-radius: 10px; padding: 22px;
              text-align: center; cursor: pointer; transition: all .2s; background: #fafafa; }
  .dropzone:hover, .dropzone.over { border-color: var(--primary); background: #eff6ff; }
  .dropzone p { color: var(--muted); font-size: 13px; }
  .dropzone strong { color: var(--text); }

  /* Чат */
  #chat { display: flex; flex-direction: column; gap: 10px; min-height: 200px; max-height: 460px;
          overflow-y: auto; padding: 4px; }
  .msg { padding: 10px 14px; border-radius: 10px; line-height: 1.6; font-size: 14px; }
  .msg.user { align-self: flex-end; background: var(--primary); color: #fff; max-width: 78%; }
  .msg.bot  { align-self: flex-start; background: #f8fafc; border: 1px solid var(--border);
              max-width: 88%; white-space: pre-wrap; }
  .msg.error { align-self: center; background: #fee2e2; color: #b91c1c;
               border: 1px solid #fca5a5; font-size: 13px; }

  /* Раскрывающийся блок источников */
  .sources { margin-top: 8px; border-top: 1px dashed var(--border); padding-top: 8px; }
  .sources-toggle { font-size: 12px; color: var(--purple); cursor: pointer; user-select: none; }
  .sources-toggle:hover { text-decoration: underline; }
  .sources-list { display: none; margin-top: 8px; flex-direction: column; gap: 6px; }
  .sources-list.open { display: flex; }
  .source-item { background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 6px;
                 padding: 8px 10px; font-size: 12px; }
  .source-item .meta { color: var(--purple); font-weight: 700; margin-bottom: 4px; }
  .source-item .text { color: var(--text); line-height: 1.55; max-height: 80px; overflow-y: auto; }

  /* Форма ввода */
  .ask-row { display: flex; gap: 8px; margin-top: 12px; }
  .ask-row input { flex: 1; border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px;
                   font-size: 14px; font-family: inherit; outline: none; }
  .ask-row input:focus { border-color: var(--primary); }
  .btn { padding: 9px 18px; border: none; border-radius: 8px; font-size: 13px;
         font-weight: 600; cursor: pointer; transition: opacity .2s; }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn-primary { background: var(--primary); color: #fff; }
  .btn-clear   { background: #fee2e2; color: #b91c1c; }
  .btn:hover:not(:disabled) { opacity: .85; }

  /* Боковая панель */
  .src-list { display: flex; flex-direction: column; gap: 6px; }
  .src-item { background: #f8fafc; border: 1px solid var(--border); border-radius: 6px;
              padding: 8px 10px; font-size: 12px; }
  .src-item b { color: var(--text); }
  .src-item span { color: var(--muted); font-size: 11px; }
  .empty { color: var(--muted); font-size: 12px; font-style: italic; padding: 6px; }

  .toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
           background: #1e293b; color: #fff; padding: 9px 18px; border-radius: 8px;
           font-size: 13px; opacity: 0; transition: opacity .3s; pointer-events: none; }
  .toast.show { opacity: 1; }

  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border);
             border-top-color: var(--primary); border-radius: 50%;
             animation: spin .7s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  @media (max-width: 800px) { .layout { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<header>
  <span style="font-size: 22px;">📚</span>
  <h1>RAG App</h1>
  <span class="badge">Лекция 11 — локальные модели + ChromaDB</span>
</header>

<div class="layout">
  <!-- ─── Главная колонка ─────────────────────────── -->
  <main style="display: flex; flex-direction: column; gap: 14px;">

    <!-- Загрузка документов -->
    <div class="panel">
      <h2>📄 Загрузить документ</h2>
      <div class="dropzone" id="dropzone">
        <p><strong>Перетащите .txt или .md файл сюда</strong><br>
           или <a href="#" id="pick">выберите вручную</a></p>
        <input type="file" id="file-input" accept=".txt,.md" hidden>
      </div>
    </div>

    <!-- Чат -->
    <div class="panel">
      <h2>💬 Чат с документами</h2>
      <div id="chat">
        <div class="msg bot">Загрузите документ и задайте вопрос. Я отвечу только по содержимому загруженных файлов.</div>
      </div>
      <div class="ask-row">
        <input type="text" id="question" placeholder="Введите вопрос..." />
        <button class="btn btn-primary" id="ask-btn">Спросить</button>
      </div>
    </div>
  </main>

  <!-- ─── Боковая колонка: источники ──────────────── -->
  <aside style="display: flex; flex-direction: column; gap: 14px;">
    <div class="panel">
      <h2>📚 Загружено</h2>
      <div id="sources-list" class="src-list"></div>
      <button class="btn btn-clear" id="clear-btn"
              style="margin-top: 12px; width: 100%; font-size: 12px;">
        🗑 Очистить базу
      </button>
    </div>
  </aside>
</div>

<div class="toast" id="toast"></div>

<script>
// ── Состояние ────────────────────────────────────────────
const dropzone   = document.getElementById('dropzone');
const fileInput  = document.getElementById('file-input');
const chat       = document.getElementById('chat');
const question   = document.getElementById('question');
const askBtn     = document.getElementById('ask-btn');
const srcList    = document.getElementById('sources-list');

// ── Утилиты ──────────────────────────────────────────────
function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

function addMsg(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

// ── Drag-and-drop ─────────────────────────────────────────
dropzone.addEventListener('click', () => fileInput.click());
document.getElementById('pick').addEventListener('click', e => { e.preventDefault(); fileInput.click(); });
fileInput.addEventListener('change', e => { if (e.target.files[0]) uploadFile(e.target.files[0]); });

['dragenter', 'dragover'].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('over'); })
);
['dragleave', 'drop'].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('over'); })
);
dropzone.addEventListener('drop', e => { if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); });

// ── Загрузка файла ────────────────────────────────────────
async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);

  const note = addMsg(`⏳ Индексирую "${file.name}"...`, 'bot');
  try {
    const r = await fetch('/upload', { method: 'POST', body: form });
    const d = await r.json();
    if (!r.ok || d.error) {
      note.className = 'msg error';
      note.textContent = '⚠️ ' + (d.error || r.status);
      return;
    }
    note.textContent = `✅ "${d.source}" — ${d.chunks} чанков добавлено. Дата: ${d.date}. Всего в базе: ${d.total_in_db}.`;
    refreshSources();
  } catch (e) {
    note.className = 'msg error';
    note.textContent = '⚠️ ' + e.message;
  }
}

// ── Список источников в боковой панели ────────────────────
async function refreshSources() {
  try {
    const r = await fetch('/sources');
    const d = await r.json();
    if (!d.sources.length) {
      srcList.innerHTML = '<div class="empty">Документов пока нет</div>';
      return;
    }
    srcList.innerHTML = d.sources.map(s =>
      `<div class="src-item"><b>${escHtml(s.name)}</b><br><span>${s.chunks} чанков • стр. ${escHtml(s.pages)} • дата ${escHtml(s.date)}</span></div>`
    ).join('');
  } catch (e) { /* тихо игнорируем */ }
}

// ── Отправка вопроса ──────────────────────────────────────
async function sendQuestion() {
  const q = question.value.trim();
  if (!q) return;

  addMsg(q, 'user');
  question.value = '';
  askBtn.disabled = true;

  const thinking = addMsg('', 'bot');
  thinking.innerHTML = '<span class="spinner"></span>Думаю...';

  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const d = await r.json();
    if (!r.ok || d.error) {
      thinking.className = 'msg error';
      thinking.textContent = '⚠️ ' + (d.error || r.status);
      return;
    }
    // Подменяем содержимое: ответ + раскрывающийся блок источников
    thinking.innerHTML = '';
    const ansDiv = document.createElement('div');
    ansDiv.textContent = d.answer;
    thinking.appendChild(ansDiv);

    if (d.sources && d.sources.length) {
      const wrap = document.createElement('div');
      wrap.className = 'sources';
      const toggle = document.createElement('div');
      toggle.className = 'sources-toggle';
      toggle.textContent = `▶ Показать источники (${d.sources.length})`;
      const list = document.createElement('div');
      list.className = 'sources-list';
      d.sources.forEach(s => {
        const item = document.createElement('div');
        item.className = 'source-item';
        item.innerHTML = `<div class="meta">#${s.rank} • ${escHtml(s.source)} • стр. ${escHtml(s.page)} • дата ${escHtml(s.date)} • чанк ${s.chunk_index} • cos=${s.similarity}</div>
                          <div class="text">${escHtml(s.text)}</div>`;
        list.appendChild(item);
      });
      toggle.addEventListener('click', () => {
        const open = list.classList.toggle('open');
        toggle.textContent = (open ? '▼ Скрыть' : '▶ Показать') + ` источники (${d.sources.length})`;
      });
      wrap.appendChild(toggle);
      wrap.appendChild(list);
      thinking.appendChild(wrap);
    }
  } catch (e) {
    thinking.className = 'msg error';
    thinking.textContent = '⚠️ ' + e.message;
  } finally {
    askBtn.disabled = false;
    question.focus();
  }
}

askBtn.addEventListener('click', sendQuestion);
question.addEventListener('keydown', e => { if (e.key === 'Enter') sendQuestion(); });

// ── Очистка базы ──────────────────────────────────────────
document.getElementById('clear-btn').addEventListener('click', async () => {
  if (!confirm('Удалить все загруженные документы?')) return;
  await fetch('/clear', { method: 'POST' });
  refreshSources();
  showToast('База очищена');
});

// ── Инициализация ─────────────────────────────────────────
refreshSources();
</script>
</body>
</html>"""


# ********************* ТОЧКА ВХОДА *********************

if __name__ == "__main__":
    print("=" * 70)
    print(f"  📚 RAG App: http://localhost:{PORT}")
    print(f"  Папка БД: {CHROMA_DIR}")
    print(f"  Записей в БД: {collection.count()}")
    print(f"\n  Требования:")
    print(f"    1. LM Studio запущен (порт 1234)")
    print(f"    2. Загружены: {EMBEDDING_MODEL}")
    print(f"                  {LLM_MODEL}")
    print(f"    3. pip install flask openai chromadb")
    print("=" * 70 + "\n")
    app.run(debug=True, port=PORT)
