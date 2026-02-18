"""
Streamlit-интерфейс для аналитического сервиса Синергия.
Два агента: RAG-агент (внутренние документы) и Websearch-агент (аналитика вузов).

Оптимизация «Начать поиск» (по логам):
- Раньше: RAG (search ~29 s + generate ~23 s) + Websearch + Future — всё подряд.
  Websearch при 500/524 от Artemox ждал до 72–645 s, из-за этого долгое ожидание.
- Сделано: после «Начать поиск» выполняется только RAG; ответ показывается сразу.
  Websearch и Future запускаются при открытии соответствующих вкладок (лениво).
- Retriever (BM25 + FAISS + SentenceTransformer) подгружается в фоне при старте приложения,
  чтобы первый поиск не тратил ~20 s на холодную загрузку.
"""
import concurrent.futures
import logging
import os
import threading
import time
import uuid

import streamlit as st

# Таймаут для Websearch-агента (сек); при превышении показываем «Агент пока недоступен»
WEBSEARCH_TIMEOUT = 60
FUTURE_AGENT_TIMEOUT = 90
FINAL_STRATEGY_TIMEOUT = 60
POLL_INTERVAL = 2  # интервал опроса фоновых агентов (сек)


def _run_rag_task(search_query: str, primary_query: str, original_query: str):
    """Выполняет RAG (поиск + генерация) в потоке. Возвращает (answer, docs, top_sources, error)."""
    try:
        ret = get_retriever()
        docs = ret.search(search_query, primary_query=primary_query)
        if not docs:
            return (None, [], [], None)
        answer = generate(original_query, docs)
        return (answer, docs, ret.get_top_sources(), None)
    except Exception as e:
        log.warning("RAG task failed: %s", e)
        return (None, [], [], str(e))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Предзагрузка retriever в фоне, чтобы первый «Начать поиск» не ждал холодный старт (~20 s)
_preload_started = False
def _preload_retriever():
    global _preload_started
    if _preload_started:
        return
    _preload_started = True
    def _run():
        try:
            from retriever import get_retriever
            get_retriever()
            log.info("Retriever preloaded (BM25 + FAISS + embedding model)")
        except Exception as e:
            log.warning("Retriever preload failed: %s", e)
    threading.Thread(target=_run, daemon=True).start()
_preload_retriever()

from classifier import FIELDS, FIELDS_RU, classify, params_to_keywords
from generator import generate
from query_enricher import enrich_query
from retriever import get_retriever
from websearch_agent import web_search
from future_agent import future_chat
from final_strategy_agent import build_final_strategy

# Подставить ключ из st.secrets, если нет в env
if "OPENROUTER_API_KEY" not in os.environ:
    try:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
if "ARTEMOX_API_KEY" not in os.environ:
    try:
        os.environ["ARTEMOX_API_KEY"] = st.secrets["ARTEMOX_API_KEY"]
    except Exception:
        pass

st.set_page_config(page_title="Цифровой ассистент Синергии", layout="centered")

# --- Styling: Synergy palette (red/white/black) ---
st.markdown(
    """
    <style>
    :root {
        --synergy-red: #d71920;
        --synergy-black: #111111;
        --synergy-gray: #f4f4f4;
    }
    .stApp {
        background-color: #ffffff;
        color: var(--synergy-black);
    }
    .main h1, .main h2, .main h3 {
        color: var(--synergy-black);
    }
    section[data-testid="stSidebar"] {
        width: 360px !important;
        min-width: 360px !important;
    }
    .synergy-title {
        background: #ffffff;
        color: var(--synergy-red);
        padding: 26px 30px;
        border-radius: 10px;
        border: 2px solid rgba(215, 25, 32, 0.22);
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        font-weight: 700;
        font-size: 38px;
        letter-spacing: 0.4px;
        margin-bottom: 14px;
        margin-top: 0;
    }
    .synergy-caption {
        margin-top: 6px;
        margin-bottom: 22px;
        color: #2b2b2b;
        font-size: 17px;
        line-height: 1.45;
    }
    .main .block-container {
        padding-top: 18px;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 18px;
    }
    .stButton > button {
        background-color: var(--synergy-red) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.45rem 1rem !important;
        white-space: nowrap !important;
        width: auto !important;
        min-width: 6rem !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #eeeeee !important;
        color: var(--synergy-black) !important;
        border: 1px solid #d7d7d7 !important;
        width: auto !important;
        min-width: 6rem !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #e3e3e3 !important;
        color: var(--synergy-black) !important;
    }
    .stButton > button:hover {
        background-color: #b9151a !important;
        color: #ffffff !important;
    }
    .stTextInput > div > div > input,
    .stTextArea textarea {
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 18px;
        letter-spacing: 0.2px;
        padding: 10px 14px !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--synergy-red) !important;
        border-bottom: 3px solid var(--synergy-red) !important;
    }
    .stAlert {
        border-left: 4px solid var(--synergy-red);
    }
    .synergy-note {
        background: var(--synergy-gray);
        border: 1px solid #e0e0e0;
        border-left: 4px solid var(--synergy-black);
        padding: 10px 12px;
        border-radius: 8px;
        color: var(--synergy-black);
    }
    .synergy-separator {
        height: 1px;
        background: #e6e6e6;
        margin: 10px 0 18px 0;
        border: 0;
    }
    .metric-wrapper {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: stretch;
        margin: 8px 0 12px 0;
    }
    .metric-bars {
        flex: 2 1 260px;
    }
    .metric-bar-row {
        margin-bottom: 6px;
    }
    .metric-bar-label {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 4px;
        letter-spacing: 0.2px;
    }
    .metric-bar-track {
        background: #f3f3f3;
        border-radius: 999px;
        overflow: hidden;
        height: 8px;
    }
    .metric-bar-fill {
        height: 8px;
    }
    .metric-bar-effect { background: #2ecc71; }   /* зелёный — эффект */
    .metric-bar-cost   { background: #f39c12; }   /* янтарный — затраты */
    .metric-bar-time   { background: #3498db; }   /* синий — время */
    .metric-bar-risk   { background: #e74c3c; }   /* красный — риск */
    .metric-opt-box {
        flex: 0 0 150px;
        border: 2px solid var(--synergy-red);
        border-radius: 999px;
        padding: 8px 14px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: #fff5f5;
    }
    .metric-opt-title {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 2px;
        color: var(--synergy-black);
    }
    .metric-opt-score {
        font-size: 22px;
        font-weight: 800;
    }
    .metric-opt-score {
        color: var(--synergy-red);
    }
    .swot-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        background: #ffffff;
    }
    .swot-table th, .swot-table td {
        padding: 10px 12px;
        vertical-align: top;
        border-bottom: 1px solid #f0f0f0;
    }
    .swot-table td {
        white-space: pre-line;
    }
    .swot-table tr:last-child th, .swot-table tr:last-child td {
        border-bottom: 0;
    }
    .swot-tag {
        font-weight: 800;
        width: 68px;
        white-space: nowrap;
    }
    .swot-s { color: #1a7f37; background: #eef9f1; }
    .swot-w { color: #b54708; background: #fff4e5; }
    .swot-o { color: #0b4aa2; background: #eaf2ff; }
    .swot-t { color: #b42318; background: #ffeceb; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="synergy-title">Цифровой ассистент руководства корпорации Синергия</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="synergy-caption">Единая аналитическая среда, объединяющая внутренние данные, '
    'внешние кейсы вузов и прогнозы для поддержки управленческих решений.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="synergy-separator"></div>', unsafe_allow_html=True)

# --- Sidebar: описание системы и агентов ---
with st.sidebar:
    st.markdown("### О системе")
    st.write(
        "Сервис помогает принимать решения на основе "
        "внутренних документов, анализа конкурентов и прогнозных сценариев."
    )
    st.markdown("### Блоки анализа")
    st.markdown("**🟥 Наш прошлый опыт**")
    st.write("Внутренняя аналитика: документы, стенограммы, брифинги.")
    st.markdown("**🟥 Анализ конкурентов**")
    st.write("Внешние кейсы других вузов РФ и СНГ.")
    st.markdown("**🟥 Предложения и гипотезы**")
    st.write("Идеи и варианты развития на 1–3 года.")

# Инициализация session_id для websearch и других агентов
if "websearch_session_id" not in st.session_state:
    st.session_state["websearch_session_id"] = str(uuid.uuid4())
if "future_session_id" not in st.session_state:
    st.session_state["future_session_id"] = str(uuid.uuid4())
if "final_session_id" not in st.session_state:
    st.session_state["final_session_id"] = str(uuid.uuid4())

# Инициализация состояния
if "enriched_query" not in st.session_state:
    st.session_state["enriched_query"] = None
if "query_approved" not in st.session_state:
    st.session_state["query_approved"] = False
if "websearch_unavailable" not in st.session_state:
    st.session_state["websearch_unavailable"] = False
if "future_unavailable" not in st.session_state:
    st.session_state["future_unavailable"] = False
if "final_unavailable" not in st.session_state:
    st.session_state["final_unavailable"] = False

# Опрос фоновых агентов (Websearch, Future) — результат выводится по готовности
# Примечание: final_strategy теперь выполняется синхронно, не опрашиваем
def _poll_pending_agents():
    rerun_needed = False
    for key, result_key, unavailable_key, timeout in [
        ("_pending_websearch_future", "websearch_result", "websearch_unavailable", WEBSEARCH_TIMEOUT),
        ("_pending_future_future", "future_result", "future_unavailable", FUTURE_AGENT_TIMEOUT),
    ]:
        fut = st.session_state.get(key)
        if fut is None:
            continue
        start = st.session_state.get(key + "_start", 0)
        if time.time() - start > timeout + 5:
            st.session_state[result_key] = None
            st.session_state[unavailable_key] = True
            del st.session_state[key]
            if key + "_start" in st.session_state:
                del st.session_state[key + "_start"]
            log.warning("%s: снято по таймауту", key)
            continue
        try:
            res = fut.result(timeout=0)
            st.session_state[result_key] = res
            st.session_state[unavailable_key] = False
            del st.session_state[key]
            if key + "_start" in st.session_state:
                del st.session_state[key + "_start"]
            log.info("%s: готов", key)
        except concurrent.futures.TimeoutError:
            rerun_needed = True
        except Exception as e:
            st.session_state[result_key] = None
            st.session_state[unavailable_key] = True
            del st.session_state[key]
            if key + "_start" in st.session_state:
                del st.session_state[key + "_start"]
            log.warning("%s failed: %s", key, e)
    if not rerun_needed and "_agent_executor" in st.session_state:
        try:
            st.session_state["_agent_executor"].shutdown(wait=False)
        except Exception:
            pass
        del st.session_state["_agent_executor"]
    return rerun_needed

_poll_rerun = _poll_pending_agents()

# Убрали фоновое выполнение - стратегии будут считаться синхронно при нажатии "Запустить советника"

# ---- Глобальный ввод и обогащение запроса (для всех вкладок) ----
query = st.text_input(
    "Введите запрос:",
    placeholder="Например: сотрудничество со Сбером в 2025?",
    key="rag_query",
)

# Логика цветов кнопок: красной должна быть актуальная кнопка
_current_q = (query or "").strip()
_has_enriched = bool(st.session_state.get("enriched_query"))
_original_matches = st.session_state.get("original_query") == _current_q
_enrich_ready_for_current = _has_enriched and _original_matches

# "Обработать запрос" красная, если запрос не обогащён (по умолчанию красная при первом заходе)
_should_show_process_primary = not _enrich_ready_for_current

launch_btn = st.button(
    "Обработать запрос",
    key="rag_launch",
    type="primary" if _should_show_process_primary else "secondary",
)

if launch_btn and query:
    st.session_state["original_query"] = query.strip()
    st.session_state["query_approved"] = False
    with st.spinner("Обогащение запроса…"):
        try:
            t0 = time.perf_counter()
            enriched_query = enrich_query(query.strip())
            log.info("Enrich query (Запуск советника): %.2f s", time.perf_counter() - t0)
            st.session_state["enriched_query"] = enriched_query
        except Exception as e:
            st.error(f"Ошибка при обогащении запроса: {e}")
            st.session_state["enriched_query"] = query.strip()
    # После обогащения делаем rerun, чтобы обновить цвета кнопок
    st.rerun()

if st.session_state.get("enriched_query") and st.session_state.get("original_query") == (query or "").strip():
    st.markdown("---")
    st.markdown("### 📝 Обогащенный запрос")
    st.caption("Вы можете отредактировать запрос перед запуском анализа:")

    edited_query_direct = st.text_area(
        "Обогащенный запрос",
        value=st.session_state.get("enriched_query", ""),
        key="edited_enriched_query_direct",
        height=100,
        label_visibility="collapsed",
    )
    if edited_query_direct != st.session_state.get("enriched_query"):
        st.session_state["enriched_query"] = edited_query_direct

    # "Запустить советника" красная, когда есть обогащенный запрос для текущего запроса
    _should_show_launch_primary = _enrich_ready_for_current
    propose_btn = st.button(
        "Запустить советника",
        type="primary" if _should_show_launch_primary else "secondary",
        key="propose_all_agents",
    )
    if propose_btn:
        st.session_state["enriched_query"] = edited_query_direct.strip()
        st.session_state["websearch_unavailable"] = False
        st.session_state["future_unavailable"] = False
        st.session_state["final_strategy_result"] = None  # Сбрасываем предыдущие стратегии
        try:
            q = st.session_state["original_query"]
            eq = st.session_state["enriched_query"]
            sid_web = st.session_state["websearch_session_id"]
            sid_fut = st.session_state["future_session_id"]

            with st.spinner("Запуск всех агентов и формирование стратегий…"):
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=3)
                f_rag = ex.submit(_run_rag_task, q, q, q)
                f_web = ex.submit(web_search, session_id=sid_web, user_query=eq)
                f_fut = ex.submit(future_chat, session_id=sid_fut, user_query=eq)
                
                # Ждём все 3 результата
                t0 = time.perf_counter()
                rag_result = f_rag.result(timeout=120)
                log.info("RAG: готов за %.2f s", time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                web_result = f_web.result(timeout=WEBSEARCH_TIMEOUT)
                log.info("Websearch: готов за %.2f s", time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                future_result = f_fut.result(timeout=FUTURE_AGENT_TIMEOUT)
                log.info("Future: готов за %.2f s", time.perf_counter() - t0)

            if rag_result and rag_result[3]:
                st.error(f"Ошибка RAG: {rag_result[3]}")
            elif rag_result and rag_result[0]:
                # Сохраняем RAG результаты в session_state
                st.session_state["last_answer"] = rag_result[0]
                st.session_state["last_docs"] = rag_result[1]
                st.session_state["top_sources"] = rag_result[2]
                log.info("RAG results saved: answer length=%d, docs=%d, sources=%d", 
                        len(rag_result[0]) if rag_result[0] else 0,
                        len(rag_result[1]) if rag_result[1] else 0,
                        len(rag_result[2]) if rag_result[2] else 0)
            else:
                st.info("По запросу ничего не найдено.")
                # Очищаем старые данные, если ничего не найдено
                st.session_state["last_answer"] = None
                st.session_state["last_docs"] = None
                st.session_state["top_sources"] = None

            st.session_state["websearch_result"] = web_result
            st.session_state["future_result"] = future_result

            # Формируем итоговые стратегии синхронно
            if rag_result and rag_result[0] and web_result and future_result:
                with st.spinner("Формируем итоговые стратегии…"):
                    try:
                        raw_web = getattr(web_result, "raw", {}) or {}
                        web_summary = raw_web.get("summary", "") if isinstance(raw_web, dict) else ""
                        web_bullets = raw_web.get("bullets", []) if isinstance(raw_web, dict) else []
                        if not isinstance(web_bullets, list):
                            web_bullets = []

                        t0 = time.perf_counter()
                        final_result = build_final_strategy(
                            rag_summary=rag_result[0],
                            web_summary=web_summary,
                            web_bullets=web_bullets,
                            future_text=getattr(future_result, "answer_text", "") or "",
                        )
                        log.info("Final-strategy agent: готов за %.2f s", time.perf_counter() - t0)
                        # Сохраняем результат в session_state ПЕРЕД rerun
                        st.session_state["final_strategy_result"] = final_result
                        st.session_state["show_swot_map"] = {}
                        log.info("Final strategy saved to session_state: %s", type(final_result))
                    except Exception as e:
                        st.error(f"Ошибка при формировании стратегий: {e}")
                        log.warning("Final strategy failed: %s", e)
                        import traceback
                        log.warning("Traceback: %s", traceback.format_exc())

            try:
                ex.shutdown(wait=False)
            except Exception:
                pass
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка: {e}")
            log.warning("Agent execution failed: %s", e)

# ---- Итоговые стратегии и SWOT на главной ----
# Стратегии выводятся ПЕРВЫМИ, перед вкладками
final_strategy = st.session_state.get("final_strategy_result")
if final_strategy:
    result = final_strategy
    import re

    text = result.main_text or ""
    swot_all = result.swot_text or ""
    
    log.info("Final strategy found: text length=%d, swot length=%d", len(text), len(swot_all))
    
    # Всегда показываем заголовок, если есть стратегии
    st.markdown("---")
    st.subheader("Итоговые отранжированные стратегии")
    
    # Выводим стратегии всегда, если они есть
    if text:
        blocks = re.split(r"\n(?=###\s*Стратегия\s*\d+:)", text)
        header = blocks[0].strip() if blocks else ""
        lines = header.splitlines()
        keep = []
        for line in lines:
            s = line.strip()
            if s.startswith("Ранжирование") or s.startswith("1\ufe0f\u20e3") or s.startswith("2\ufe0f\u20e3") or s.startswith("3\ufe0f\u20e3"):
                break
            keep.append(line)
        header = "\n".join(keep).strip()
        if header:
            st.markdown(header)

    swot_by_idx: dict[int, dict[str, list[str]]] = {}
    if swot_all:
        parts = re.split(r"\n(?=###\s*Стратегия\s*\d+:)", swot_all)
        for p in parts:
            m_idx = re.match(r"###\s*Стратегия\s*(\d+):", p.strip())
            if not m_idx:
                continue
            idx = int(m_idx.group(1))
            swot_by_idx[idx] = {"S": [], "W": [], "O": [], "T": []}
            for key in ["S", "W", "O", "T"]:
                m = re.search(rf"{key}\s*:\s*(.*?)(?=\n[A-Z]\s*:|\Z)", p, flags=re.DOTALL)
                if m:
                    lines = []
                    for line in m.group(1).splitlines():
                        line = line.strip()
                        if line.startswith("-"):
                            lines.append(line.lstrip("-").strip())
                    swot_by_idx[idx][key] = lines[:5]

    def _extract_scores(block: str) -> dict[str, str]:
        scores = {}
        for label in ["Затратность", "Рисковость", "Время", "Эффект", "Оптимальность"]:
            m = re.search(rf"{label}\s*=\s*(\d+)", block)
            if not m:
                m = re.search(rf"{label}\s*:\s*(\d+)", block)
            if m:
                scores[label] = m.group(1)
        return scores

    def _render_pills(scores: dict):
        if not scores:
            return

        def _clamp(val: str) -> int:
            try:
                v = int(val)
            except Exception:
                return 0
            return max(0, min(v, 10))

        bars_order = [
            ("Эффект", "effect"),
            ("Затратность", "cost"),
            ("Время", "time"),
            ("Рисковость", "risk"),
        ]

        parts = ['<div class="metric-wrapper">']

        # Левая часть — гистограммы
        bars = ['<div class="metric-bars">']
        for label, key in bars_order:
            if label not in scores:
                continue
            v = _clamp(scores[label])
            width = v * 10  # 0–100%
            bars.append(
                '<div class="metric-bar-row">'
                f'<div class="metric-bar-label">{label}: {v}/10</div>'
                '<div class="metric-bar-track">'
                f'<div class="metric-bar-fill metric-bar-{key}" style="width:{width}%;"></div>'
                '</div>'
                '</div>'
            )
        bars.append("</div>")  # .metric-bars
        parts.append("".join(bars))

        # Правая часть — оптимальность
        opt_val = scores.get("Оптимальность")
        if opt_val is not None:
            ov = _clamp(opt_val)
            parts.append(
                '<div class="metric-opt-box">'
                '<div class="metric-opt-title">Оптимальность</div>'
                f'<div class="metric-opt-score">{ov}/10</div>'
                '</div>'
            )

        parts.append("</div>")  # .metric-wrapper
        st.markdown("".join(parts), unsafe_allow_html=True)

    def _render_swot_table(swot: dict[str, list[str]]):
        def _clean(s: str) -> str:
            s = re.sub(r"<br\s*/?>", " ", s, flags=re.IGNORECASE)
            s = re.sub(r"<[^>]+>", "", s)
            s = s.replace("•", "").strip()
            return s.strip() or "—"

        def _escape(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def _li(items: list[str]) -> str:
            if not items:
                return "—"
            cleaned = [_escape(_clean(i)) for i in items]
            return "\n".join(cleaned)

        html = f"""
        <table class="swot-table">
          <tr>
            <th class="swot-tag swot-s">🟢 S</th>
            <td>{_li(swot.get("S", []))}</td>
          </tr>
          <tr>
            <th class="swot-tag swot-w">🟠 W</th>
            <td>{_li(swot.get("W", []))}</td>
          </tr>
          <tr>
            <th class="swot-tag swot-o">🔵 O</th>
            <td>{_li(swot.get("O", []))}</td>
          </tr>
          <tr>
            <th class="swot-tag swot-t">🔴 T</th>
            <td>{_li(swot.get("T", []))}</td>
          </tr>
        </table>
        """
        st.markdown(html, unsafe_allow_html=True)

    # Обрабатываем блоки стратегий только если есть текст
    if text:
        blocks_main = blocks[1:] if len(blocks) > 1 else []
        strategy_blocks = []
        for i, b in enumerate(blocks_main, 1):
            b = b.strip()
            if not b or not re.match(r"^###\s*Стратегия\s*\d+:", b):
                continue
            opt = _extract_scores(b).get("Оптимальность", "0")
            try:
                opt_int = int(opt)
            except ValueError:
                opt_int = 0
            strategy_blocks.append((opt_int, i, b))

        strategy_blocks.sort(key=lambda x: (-x[0], x[1]))
        cup_chars = ("\U0001f947", "\U0001f948", "\U0001f949")

        def _drop_ranking_block(text: str) -> str:
            lines = text.splitlines()
            keep = []
            for line in lines:
                s = line.strip()
                if s.startswith("Ранжирование") or s.startswith("1\ufe0f\u20e3") or s.startswith("2\ufe0f\u20e3") or s.startswith("3\ufe0f\u20e3"):
                    break
                keep.append(line)
            return "\n".join(keep).strip()

        def _drop_scores_and_rules(text: str) -> str:
            """Убирает строку с оценками (Оценки 0-10: ...) и горизонтальные разделители (---)."""
            lines = text.splitlines()
            keep = []
            for line in lines:
                s = line.strip()
                if "Оценки" in s and ("Затратность" in s or "Оптимальность" in s or re.search(r"\d+\s*;\s*\d+", s)):
                    continue
                if re.match(r"^[-*_]{2,}\s*$", s):
                    continue
                keep.append(line)
            return "\n".join(keep).strip()

        for rank, (opt_int, i, b) in enumerate(strategy_blocks, 1):
            title_line = b.splitlines()[0].strip()
            title_rest = re.sub(r"^#+\s*", "", title_line).strip()

            rank_badge = ""
            if rank <= 3:
                rank_badge = f'<span style="margin-right:8px;">{cup_chars[rank - 1]}</span>'

            st.markdown(
                f"""
                <h3 style="display:flex;align-items:center;gap:6px;">
                    {rank_badge}
                    <span>{title_rest}</span>
                </h3>
                """,
                unsafe_allow_html=True,
            )

            scores = _extract_scores(b)
            _render_pills(scores)

            b_no_scores = re.sub(r"^Оценки.*?$", "", b, flags=re.MULTILINE).strip()
            desc_raw = "\n".join(b_no_scores.splitlines()[1:]).strip()
            desc = _drop_ranking_block(desc_raw)
            desc = _drop_scores_and_rules(desc)
            if desc:
                st.markdown(desc)

            if "show_swot_map" not in st.session_state:
                st.session_state["show_swot_map"] = {}
            shown = bool(st.session_state["show_swot_map"].get(i, False))
            btn = "Показать SWOT" if not shown else "Скрыть SWOT"
            if st.button(btn, type="primary" if not shown else "secondary", key=f"swot_btn_{i}"):
                st.session_state["show_swot_map"][i] = not shown
                st.rerun()

            if st.session_state["show_swot_map"].get(i, False):
                sw = swot_by_idx.get(i, {"S": [], "W": [], "O": [], "T": []})
                _render_swot_table(sw)

            st.markdown("<br>", unsafe_allow_html=True)
# Если стратегий нет - просто ничего не показываем, вкладки будут ниже

# Вкладки для переключения между блоками анализа (показываются всегда, но детали доступны после запуска)
tab1, tab2, tab3 = st.tabs([
    "📚 Наш прошлый опыт",
    "🔍 Анализ конкурентов",
    "🚀 Предложения и гипотезы",
])

# =========================
# ВКЛАДКА 1: НАШ ПРОШЛЫЙ ОПЫТ (RAG)
# =========================
with tab1:
    st.subheader("Наш прошлый опыт: аналитика внутренних документов")

    # ---- Ответ RAG ----
    if st.session_state.get("last_answer"):
        st.markdown("### Ответ по внутренним документам")
        raw = st.session_state["last_answer"]
        import re
        cleaned = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        st.markdown(cleaned)

    # ---- Источники RAG ----
    if st.session_state.get("top_sources"):
        st.markdown("### Источники")
        top_sources = st.session_state["top_sources"]
        if top_sources:
            for src in top_sources:
                source_text = src.get("file", "Неизвестный файл")
                if src.get("date"):
                    source_text += f" ({src.get('date')})"
                st.markdown(f"• {source_text}")
        else:
            st.caption("Нет результатов")

# =========================
# ВКЛАДКА 2: АНАЛИЗ КОНКУРЕНТОВ (WEBSEARCH)
# =========================
with tab2:
    st.subheader("Анализ конкурентов: аналогичные ситуации у других университетов")
    
    # Если есть результат от автоматического запуска или сохраненный результат
    if st.session_state.get("websearch_result"):
        result = st.session_state["websearch_result"]
        
        # Показываем результаты в читаемом виде
        st.markdown("### 📊 Результаты анализа")
        
        # Получаем данные из raw
        raw_data = result.raw
        summary = ""
        bullets = []
        parsed_payload = None

        # Если raw_data - строка, пытаемся распарсить как JSON
        if isinstance(raw_data, str):
            import json
            try:
                parsed_payload = json.loads(raw_data)
            except Exception:
                parsed_payload = None
        elif isinstance(raw_data, dict):
            parsed_payload = raw_data

        if isinstance(parsed_payload, dict):
            summary = parsed_payload.get("summary", "") or ""
            bullets = parsed_payload.get("bullets", []) or []

        # Если summary выглядит как JSON, пробуем распарсить ещё раз
        if isinstance(summary, str):
            summary_candidate = summary.strip()
            if "```" in summary_candidate:
                summary_candidate = summary_candidate.replace("```json", "").replace("```", "").strip()
            if summary_candidate.startswith("{"):
                import json
                try:
                    nested = json.loads(summary_candidate)
                    summary = nested.get("summary", "") or ""
                    bullets = nested.get("bullets", []) or bullets
                except Exception:
                    pass

        # Если summary и bullets пустые, пробуем использовать answer_text как JSON
        if (not summary and not bullets) and isinstance(result.answer_text, str):
            import json
            try:
                nested = json.loads(result.answer_text)
                summary = nested.get("summary", "") or summary
                bullets = nested.get("bullets", []) or bullets
            except Exception:
                pass
        
        # Показываем summary
        if summary:
            summary_clean = summary.strip()
            if "```" in summary_clean:
                summary_clean = summary_clean.replace("```json", "").replace("```", "").strip()
            if summary_clean.startswith('"') and summary_clean.endswith('"'):
                summary_clean = summary_clean[1:-1]
            st.markdown(summary_clean)
        
        # Показываем bullets
        if bullets:
            if summary:
                st.markdown("")  # Отступ после summary
            st.markdown("**Ключевые факты:**")
            for bullet in bullets:
                bullet_text = str(bullet).strip()
                if bullet_text.startswith('"') and bullet_text.endswith('"'):
                    bullet_text = bullet_text[1:-1]
                st.markdown(f"• {bullet_text}")

        if not summary and not bullets:
            st.info("Не удалось извлечь текстовый ответ. Попробуйте повторить поиск.")
        
        # Источники
        if result.sources:
            st.markdown("---")
            st.markdown("### 📚 Источники")
            for i, src in enumerate(result.sources, 1):
                title = src.get("title", "Источник")
                url = src.get("url", "")
                date = src.get("date", "")
                
                if date:
                    st.markdown(f"**{i}.** {title} *(опубликовано: {date})*")
                else:
                    st.markdown(f"**{i}.** {title}")
                
                if url:
                    st.markdown(f"🔗 [{url}]({url})")
                st.markdown("")
    
    elif st.session_state.get("_pending_websearch_future"):
        st.markdown(
            '<div class="synergy-note">Websearch‑агент выполняется. Результат появится автоматически по готовности.</div>',
            unsafe_allow_html=True,
        )

    elif st.session_state.get("websearch_unavailable"):
        st.markdown(
            '<div class="synergy-note">Агент пока недоступен. Websearch не успел ответить за отведённое время. '
            'Попробуйте позже или нажмите «Начать поиск» в RAG-агенте ещё раз.</div>',
            unsafe_allow_html=True,
        )

    elif st.session_state.get("enriched_query"):
        st.markdown(
            '<div class="synergy-note">Сначала нажмите «Обработать запрос», затем «Запустить советника» — после этого здесь появится анализ аналогичных ситуаций у других вузов.</div>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            '<div class="synergy-note">Сначала нажмите «Обработать запрос», затем «Запустить советника» — после этого здесь появится анализ аналогичных ситуаций у других вузов.</div>',
            unsafe_allow_html=True,
        )

# =========================
# ВКЛАДКА 3: ПРЕДЛОЖЕНИЯ И ГИПОТЕЗЫ (FUTURE)
# =========================
with tab3:
    st.subheader("Предложения и гипотезы на будущее (1–3 года)")

    if st.session_state.get("future_result"):
        result = st.session_state["future_result"]

        st.markdown("### 💡 Варианты развития")
        import re
        raw = result.answer_text or ""
        cleaned = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        st.markdown(cleaned)

    elif st.session_state.get("_pending_future_future"):
        st.markdown(
            '<div class="synergy-note">Future‑агент выполняется. Результат появится автоматически по готовности.</div>',
            unsafe_allow_html=True,
        )

    elif st.session_state.get("future_unavailable"):
        st.markdown(
            '<div class="synergy-note">Агент пока недоступен. Future-agent не успел ответить за отведённое время. '
            'Попробуйте позже или нажмите «Начать поиск» в RAG-агенте ещё раз.</div>',
            unsafe_allow_html=True,
        )

    elif st.session_state.get("enriched_query"):
        st.markdown(
            '<div class="synergy-note">Сначала нажмите «Обработать запрос», затем «Запустить советника» — после этого здесь появятся прогнозные предложения.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="synergy-note">Сначала нажмите «Обработать запрос», затем «Запустить советника» — после этого здесь появятся прогнозы.</div>',
            unsafe_allow_html=True,
        )


# Подсказка по ключу
if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ARTEMOX_API_KEY")):
    st.sidebar.warning(
        "API ключ не задан. Укажите OPENROUTER_API_KEY или ARTEMOX_API_KEY в окружении "
        "или в `.streamlit/secrets.toml`."
    )

# Опрос фоновых агентов: если Websearch или Future ещё в работе — обновить страницу через POLL_INTERVAL
if _poll_rerun:
    time.sleep(POLL_INTERVAL)
    st.rerun()
