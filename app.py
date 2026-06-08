"""
Bottle web UI for the DragonflyDB conversational agent — multi-user, per-user memory isolation.

Memory is namespaced per user: ("user", <user_id>, "long_term").
Multiple browser tabs can run concurrently, each with a different user+session context.
Identity is carried in the URL — no cookies required.

Routes:
  GET  /                                       user picker
  POST /user/new                               validate user_id → /u/<user_id>/
  GET  /u/<user_id>/                           session picker for this user
  POST /u/<user_id>/session/new                new session → chat
  POST /u/<user_id>/session/select             resume session → chat
  GET  /u/<user_id>/chat/<session_id>          chat page
  POST /u/<user_id>/chat/<session_id>/message  JSON: send message
  POST /u/<user_id>/chat/<session_id>/summarize JSON: summarize memories
  POST /u/<user_id>/chat/<session_id>/end      JSON: end session

Usage:
    python app.py [-H host] [-p port] [-s password] [-u username]
                  [--threshold float] [--ttl int] [--web-port int]
"""

import asyncio
import json
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import re
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Optional
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server
import socketserver

import bottle
from bottle import request, response, redirect
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from redisvl.utils.vectorize import HFTextVectorizer
from typing_extensions import Annotated, TypedDict

from agent import (
    DragonflyRedisStore,
    HFVectorizerEmbeddings,
    _is_catchall,
    _user_ns,
    extract_and_store,
    EMBEDDING_DIMS,
    EXTRACTION_EVERY_N,
    LLM_BASE_URL,
    LLM_MODEL,
)
from semanticcache import (
    DragonflySemanticCache,
    build_redis_url,
    CACHE_INDEX_NAME,
    EMBEDDING_MODEL,
)

_USER_ID_RE = re.compile(r'^[\w\-]{1,64}$')

# ── activity log ContextVar ───────────────────────────────────────────────────
_activity_log: ContextVar[list] = ContextVar("activity_log", default=[])


def _log(type_: str, msg: str, detail: str = "") -> dict:
    return {"type": type_, "msg": msg, "detail": detail}


# ── background asyncio event loop ─────────────────────────────────────────────
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True, name="aio-worker").start()


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result(timeout=600)


# ── global shared state ───────────────────────────────────────────────────────
_vectorizer: Optional[HFTextVectorizer] = None
_cache: Optional[DragonflySemanticCache] = None
_llm: Optional[ChatOpenAI] = None
_checkpointer = None
_store = None

# Per-user compiled LangGraph instances; built lazily on first use per user_id.
_compiled_graphs: dict[str, object] = {}
_compiled_graphs_lock = threading.Lock()

# In-process message + turn tracking (keyed by session_id).
_session_messages: dict[str, list[BaseMessage]] = {}
_session_turns: dict[str, int] = {}


# ── AgentState ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    memories: list[str]


# ── graph factory ─────────────────────────────────────────────────────────────

def build_web_graph(llm: ChatOpenAI, cache: DragonflySemanticCache, user_id: str) -> StateGraph:
    ns = _user_ns(user_id)

    async def retrieve_memories(state: AgentState, *, store: BaseStore) -> dict:
        log = _activity_log.get()
        if not state["messages"]:
            return {"memories": []}
        query = state["messages"][-1].content
        log.append(_log("MEMORY_SEARCH", f'Querying long-term store: "{query[:70]}"'))
        if _is_catchall(query):
            results = await store.asearch(ns, limit=20)
            log.append(_log("MEMORY_FOUND", f"Catch-all recall: {len(results)} memories retrieved"))
        else:
            results = await store.asearch(ns, query=query, limit=3)
            log.append(_log("MEMORY_FOUND", f"{len(results)} relevant memories retrieved"))
        for r in results:
            log.append(_log("MEMORY_ITEM", r.value.get("text", "")[:120], f"key: {r.key}"))
        return {"memories": [r.value["text"] for r in results if r.value.get("text")]}

    async def chat(state: AgentState) -> dict:
        log = _activity_log.get()
        user_msg = state["messages"][-1].content

        log.append(_log("CACHE_CHECK", f'Checking semantic cache: "{user_msg[:70]}"'))
        hits = cache.check(prompt=user_msg)
        if hits:
            dist = hits[0].get("vector_distance", "?")
            dist_str = f"{float(dist):.4f}" if dist != "?" else "?"
            log.append(_log("CACHE_HIT", f"Cache hit — cosine distance: {dist_str}", hits[0]["response"][:120]))
            return {"messages": [AIMessage(content=hits[0]["response"])]}

        log.append(_log("CACHE_MISS", "No cache hit — invoking LLM"))
        n_mem = len(state.get("memories", []))
        n_msg = len(state["messages"])
        system_parts = ["You are a helpful assistant."]
        if state.get("memories"):
            system_parts.append(
                "\nRelevant context from past sessions:\n"
                + "\n".join(f"- {m}" for m in state["memories"])
            )
        log.append(_log("LLM_CALL", f"Invoking {LLM_MODEL}",
                        f"{n_msg} messages in thread · {n_mem} memories in system prompt"))
        resp = await llm.ainvoke(
            [SystemMessage(content="\n".join(system_parts))] + list(state["messages"])
        )
        log.append(_log("LLM_RESPONSE", f"Response received ({len(resp.content)} chars)"))
        cache.store(prompt=user_msg, response=resp.content)
        log.append(_log("CACHE_STORE", "Response stored in semantic cache"))
        return {"messages": [AIMessage(content=resp.content)]}

    g = StateGraph(AgentState)
    g.add_node("retrieve_memories", retrieve_memories)
    g.add_node("chat", chat)
    g.add_edge(START, "retrieve_memories")
    g.add_edge("retrieve_memories", "chat")
    g.add_edge("chat", END)
    return g


def _get_compiled(user_id: str):
    with _compiled_graphs_lock:
        if user_id not in _compiled_graphs:
            _compiled_graphs[user_id] = build_web_graph(_llm, _cache, user_id).compile(
                checkpointer=_checkpointer, store=_store
            )
        return _compiled_graphs[user_id]


# ── startup ───────────────────────────────────────────────────────────────────

async def _startup(redis_url: str, ttl: int) -> None:
    global _checkpointer, _store
    os.environ.setdefault("REDIS_URL", redis_url)
    ttl_cfg = {"default_ttl": ttl, "refresh_on_read": True}
    index_cfg = {"embed": HFVectorizerEmbeddings(_vectorizer), "dims": EMBEDDING_DIMS, "fields": ["text"]}
    cp_mgr = AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl_cfg)
    _checkpointer = await cp_mgr.__aenter__()
    await _checkpointer.asetup()
    st_mgr = DragonflyRedisStore.from_conn_string(redis_url, index=index_cfg)
    _store = await st_mgr.__aenter__()
    _store.setup()


# ── async helpers ─────────────────────────────────────────────────────────────

async def _load_history(session_id: str) -> list[BaseMessage]:
    try:
        cfg = {"configurable": {"thread_id": session_id}}
        t = await _checkpointer.aget_tuple(cfg)
        if t and t.checkpoint:
            return t.checkpoint.get("channel_values", {}).get("messages", [])
    except Exception:
        pass
    return []


async def _list_users() -> list[str]:
    try:
        ns_list = await _store.alist_namespaces(prefix=("user",), max_depth=2)
    except Exception:
        return []
    users = {ns[1] for ns in ns_list if len(ns) >= 2}
    return sorted(users)


async def _list_sessions(user_id: str) -> list[dict]:
    ns = _user_ns(user_id)
    try:
        all_items = await _store.asearch(ns, limit=1000)
    except Exception:
        return []
    sessions: dict[str, dict] = {}
    for item in all_items:
        sid = item.value.get("session_id")
        if not sid:
            continue
        if sid not in sessions:
            sessions[sid] = {"session_id": sid, "facts": 0, "prefs": 0, "summary": ""}
        key = item.key or ""
        if "fact" in key:
            sessions[sid]["facts"] += 1
        elif "pref" in key:
            sessions[sid]["prefs"] += 1
        elif ("curated_summary" in key or "summary" in key) and not sessions[sid]["summary"]:
            sessions[sid]["summary"] = item.value.get("text", "")[:200]
    return sorted(sessions.values(), key=lambda s: s["session_id"])


async def _invoke(user_id: str, session_id: str, user_input: str) -> tuple[str, list]:
    activity: list = []
    token = _activity_log.set(activity)
    try:
        cfg = {"configurable": {"thread_id": session_id}}
        result = await _get_compiled(user_id).ainvoke(
            {"messages": [HumanMessage(content=user_input)], "memories": []},
            config=cfg,
        )
        reply = result["messages"][-1].content
    finally:
        _activity_log.reset(token)
    return reply, activity


async def _do_summarize(user_id: str, session_id: str) -> dict:
    activity: list = []
    ns = _user_ns(user_id)
    activity.append(_log("SUMMARIZE_SEARCH", f"Gathering memories for session {session_id[:8]}…"))
    try:
        all_items = await _store.asearch(ns, limit=1000)
    except Exception as e:
        activity.append(_log("SUMMARIZE_ERROR", f"Store search failed: {e}"))
        return {"summary": "", "activity": activity}

    session_items = [r for r in all_items if r.value.get("session_id") == session_id]
    facts  = [r.value["text"] for r in session_items if "fact"  in (r.key or "")]
    prefs  = [r.value["text"] for r in session_items if "pref"  in (r.key or "")]
    topics = [r.value["text"] for r in session_items if "topic" in (r.key or "")]
    activity.append(_log("SUMMARIZE_FOUND", f"Retrieved {len(session_items)} entries",
                         f"{len(facts)} facts · {len(prefs)} preferences · {len(topics)} topics"))

    if not session_items:
        activity.append(_log("SUMMARIZE_EMPTY", "No memories found for this session yet"))
        return {"summary": "(No memories stored for this session yet.)", "activity": activity}

    sections = []
    if facts:   sections.append("Facts:\n"             + "\n".join(f"- {f}" for f in facts))
    if prefs:   sections.append("Preferences:\n"       + "\n".join(f"- {p}" for p in prefs))
    if topics:  sections.append("Topics of interest:\n" + "\n".join(f"- {t}" for t in topics))

    activity.append(_log("SUMMARIZE_LLM", f"Sending {len(session_items)} entries to {LLM_MODEL}"))
    resp = await _llm.ainvoke([
        SystemMessage(content=(
            "You are a memory summarization assistant. Given extracted memory entries "
            "about a user, write a clear, concise, readable summary of what is known "
            "about them. Be specific and factual."
        )),
        HumanMessage(content=(
            "Construct a comprehensive summary of what is known about this user "
            "based on the following extracted memories:\n\n" + "\n\n".join(sections)
        )),
    ])
    summary_text = resp.content
    ts = str(int(time.time()))
    key = f"{session_id}_curated_summary_{ts}"
    await _store.aput(ns, key, {"text": summary_text, "session_id": session_id, "ts": ts, "type": "curated_summary"})
    activity.append(_log("SUMMARIZE_STORED", "Curated summary stored", f"key: …{key[-24:]}"))
    activity.append(_log("SUMMARIZE_DONE", "Summary complete"))
    return {"summary": summary_text, "activity": activity}


# ── HTML templates ────────────────────────────────────────────────────────────

_COMMON_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }
button { cursor: pointer; border: none; border-radius: 6px; padding: 8px 16px; font-size: 14px; font-weight: 600; transition: opacity .15s; }
button:hover { opacity: 0.85; }
.btn-primary   { background: #3182ce; color: #fff; }
.btn-secondary { background: #2d3748; color: #e2e8f0; border: 1px solid #4a5568; }
.btn-danger    { background: #e53e3e; color: #fff; }
.btn-purple    { background: #6b46c1; color: #fff; }
a { color: #63b3ed; text-decoration: none; }
a:hover { text-decoration: underline; }
"""

USER_PICKER_HTML = (
    "<!DOCTYPE html><html lang='en'><head>"
    "<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>DragonflyDB Agent</title><style>" + _COMMON_CSS + """
.container { max-width: 680px; margin: 60px auto; padding: 0 20px; }
h1 { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
.subtitle { color: #718096; margin-bottom: 32px; font-size: 14px; }
.card { background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 24px; margin-bottom: 20px; }
.card h2 { font-size: 13px; font-weight: 700; margin-bottom: 16px; color: #718096; text-transform: uppercase; letter-spacing: .07em; }
.new-form { display: flex; gap: 10px; align-items: center; }
input[type=text] { background: #0f1117; border: 1px solid #4a5568; border-radius: 6px; color: #e2e8f0; padding: 8px 12px; font-size: 14px; width: 260px; }
input[type=text]:focus { outline: none; border-color: #3182ce; }
.hint { color: #4a5568; font-size: 12px; margin-top: 10px; }
.user-grid { display: flex; flex-wrap: wrap; gap: 10px; }
.user-btn { background: #2d3748; border: 1px solid #4a5568; border-radius: 8px; padding: 12px 20px; cursor: pointer; font-size: 14px; color: #e2e8f0; font-weight: 600; transition: border-color .15s, background .15s; }
.user-btn:hover { border-color: #63b3ed; background: #1e3a5f; }
.empty { color: #4a5568; font-style: italic; font-size: 14px; }
</style></head><body>
<div class='container'>
  <h1>DragonflyDB Conversational Agent</h1>
  <p class='subtitle'>Multi-user &middot; Per-user memory isolation &middot; Semantic cache + long-term vector store</p>

  <div class='card'>
    <h2>Select or Create User</h2>
    <form method='POST' action='/user/new' class='new-form'>
      <input type='text' name='user_id' placeholder='Enter user ID…' required autocomplete='off' pattern='[\\w\\-]{1,64}'>
      <button type='submit' class='btn-primary'>Continue &rarr;</button>
    </form>
    <p class='hint'>Alphanumeric, hyphens and underscores only. Type an existing ID to resume, or a new one to start fresh.</p>
  </div>

  % if users:
  <div class='card'>
    <h2>Existing Users</h2>
    <div class='user-grid'>
      % for uid in users:
      <form method='POST' action='/user/new'>
        <input type='hidden' name='user_id' value='{{uid}}'>
        <button type='submit' class='user-btn'>{{uid}}</button>
      </form>
      % end
    </div>
  </div>
  % end
</div>
</body></html>
"""
)

SESSION_PICKER_HTML = (
    "<!DOCTYPE html><html lang='en'><head>"
    "<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>Sessions — {{user_id}}</title><style>" + _COMMON_CSS + """
.container { max-width: 900px; margin: 60px auto; padding: 0 20px; }
.breadcrumb { color: #4a5568; font-size: 13px; margin-bottom: 20px; }
h1 { font-size: 24px; font-weight: 700; margin-bottom: 6px; }
.user-badge { display: inline-block; background: #2c5282; color: #90cdf4; padding: 3px 10px; border-radius: 4px; font-size: 13px; font-weight: 600; margin-left: 10px; vertical-align: middle; }
.subtitle { color: #718096; margin-bottom: 32px; font-size: 14px; }
.card { background: #1a202c; border: 1px solid #2d3748; border-radius: 10px; padding: 24px; margin-bottom: 20px; }
.card h2 { font-size: 13px; font-weight: 700; margin-bottom: 16px; color: #718096; text-transform: uppercase; letter-spacing: .07em; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; padding: 8px 12px; color: #718096; font-weight: 600; border-bottom: 1px solid #2d3748; font-size: 12px; }
td { padding: 10px 12px; border-bottom: 1px solid #1e2533; vertical-align: top; }
tr:last-child td { border-bottom: none; }
.sid { font-family: monospace; font-size: 12px; color: #63b3ed; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-right: 4px; }
.badge-fact { background: #1a365d; color: #63b3ed; }
.badge-pref { background: #1c4532; color: #68d391; }
.summary-text { color: #a0aec0; font-size: 13px; max-width: 340px; line-height: 1.4; }
.new-form { display: flex; gap: 14px; align-items: center; }
.empty { color: #4a5568; font-style: italic; padding: 20px 0; text-align: center; font-size: 14px; }
</style></head><body>
<div class='container'>
  <div class='breadcrumb'><a href='/'>&#8592; All Users</a></div>
  <h1>Sessions <span class='user-badge'>{{user_id}}</span></h1>
  <p class='subtitle'>Memory is isolated to this user &mdash; other users cannot see these sessions.</p>

  <div class='card'>
    <h2>Start New Session</h2>
    <form method='POST' action='/u/{{user_id}}/session/new' class='new-form'>
      <button type='submit' class='btn-primary'>+ New Session</button>
      <span style='color:#718096;font-size:13px'>Creates a fresh session with no prior messages</span>
    </form>
  </div>

  <div class='card'>
    <h2>Resume Existing Session</h2>
    % if sessions:
    <table>
      <thead><tr><th>Session ID</th><th>Memories</th><th>Latest Summary</th><th></th></tr></thead>
      <tbody>
        % for s in sessions:
        <tr>
          <td><span class='sid'>{{s['session_id'][:8]}}&hellip;{{s['session_id'][-4:]}}</span></td>
          <td>
            <span class='badge badge-fact'>{{s['facts']}} facts</span>
            <span class='badge badge-pref'>{{s['prefs']}} prefs</span>
          </td>
          <td><div class='summary-text'>{{s['summary'] or '&mdash;'}}</div></td>
          <td>
            <form method='POST' action='/u/{{user_id}}/session/select'>
              <input type='hidden' name='session_id' value='{{s["session_id"]}}'>
              <button type='submit' class='btn-secondary'>Resume &rarr;</button>
            </form>
          </td>
        </tr>
        % end
      </tbody>
    </table>
    % else:
    <p class='empty'>No sessions with stored memories yet. Start a new session above.</p>
    % end
  </div>
</div>
</body></html>
"""
)

CHAT_PAGE_HTML = (
    "<!DOCTYPE html><html lang='en'><head>"
    "<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>{{user_id}} / {{session_id[:8]}} &mdash; Agent</title><style>" + _COMMON_CSS + """
body { display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
header { background: #1a202c; border-bottom: 1px solid #2d3748; padding: 12px 20px; display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
header h1 { font-size: 15px; font-weight: 700; }
.user-badge { background: #2c5282; color: #90cdf4; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 700; }
.sid-badge  { font-family: monospace; font-size: 11px; color: #63b3ed; background: #1a365d; padding: 3px 8px; border-radius: 4px; }
.header-actions { margin-left: auto; display: flex; gap: 8px; align-items: center; }
.back-link { color: #718096; font-size: 13px; }
.main { display: flex; flex: 1; overflow: hidden; }
.conv-pane { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
.messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
.msg { max-width: 78%; }
.msg.user      { align-self: flex-end; }
.msg.assistant { align-self: flex-start; }
.msg.system-note { align-self: center; max-width: 92%; }
.bubble { padding: 11px 15px; border-radius: 12px; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
.msg.user .bubble      { background: #2b6cb0; color: #fff; border-bottom-right-radius: 3px; }
.msg.assistant .bubble { background: #1a202c; border: 1px solid #2d3748; color: #e2e8f0; border-bottom-left-radius: 3px; }
.msg.system-note .bubble { background: #1c4532; border: 1px solid #276749; color: #9ae6b4; font-size: 13px; }
.msg-label { font-size: 11px; color: #4a5568; margin-bottom: 3px; }
.msg.user .msg-label { text-align: right; }
.input-area { padding: 14px 18px; background: #1a202c; border-top: 1px solid #2d3748; display: flex; gap: 10px; align-items: flex-end; flex-shrink: 0; }
textarea { flex: 1; background: #0f1117; border: 1px solid #2d3748; border-radius: 8px; color: #e2e8f0; padding: 10px 14px; font-size: 14px; resize: none; min-height: 44px; max-height: 140px; line-height: 1.5; font-family: inherit; }
textarea:focus { outline: none; border-color: #3182ce; }
#spinner { color: #718096; font-size: 13px; display: none; }
#spinner.on { display: inline; }
.activity-pane { width: 340px; flex-shrink: 0; border-left: 1px solid #2d3748; display: flex; flex-direction: column; overflow: hidden; background: #0d1117; }
.activity-header { padding: 11px 14px; font-size: 11px; font-weight: 700; color: #4a5568; text-transform: uppercase; letter-spacing: .07em; border-bottom: 1px solid #2d3748; flex-shrink: 0; }
.activity-list { flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 5px; }
.ae { background: #1a202c; border-radius: 5px; padding: 7px 10px; font-size: 12px; border-left: 3px solid #4a5568; }
.ae.CACHE_CHECK   { border-color: #4299e1; }
.ae.CACHE_HIT     { border-color: #68d391; }
.ae.CACHE_MISS    { border-color: #fc8181; }
.ae.CACHE_STORE   { border-color: #4299e1; }
.ae.MEMORY_SEARCH { border-color: #48bb78; }
.ae.MEMORY_FOUND  { border-color: #68d391; }
.ae.MEMORY_ITEM   { border-color: #276749; }
.ae.LLM_CALL      { border-color: #ed8936; }
.ae.LLM_RESPONSE  { border-color: #f6ad55; }
.ae.EXTRACT_TRIGGER { border-color: #9f7aea; }
.ae.SUMMARIZE_SEARCH,.ae.SUMMARIZE_FOUND,.ae.SUMMARIZE_LLM,
.ae.SUMMARIZE_STORED,.ae.SUMMARIZE_DONE,.ae.SUMMARIZE_EMPTY,
.ae.SUMMARIZE_ERROR { border-color: #b794f4; }
.ae-type   { font-weight: 700; color: #718096; font-size: 10px; letter-spacing: .05em; }
.ae-msg    { color: #e2e8f0; margin-top: 2px; }
.ae-detail { color: #718096; margin-top: 3px; font-style: italic; font-size: 11px; }
.ae-empty  { color: #4a5568; font-style: italic; font-size: 13px; text-align: center; padding: 20px; }
</style></head><body>

<header>
  <h1>DragonflyDB Agent</h1>
  <span class='user-badge'>{{user_id}}</span>
  <span class='sid-badge'>{{session_id}}</span>
  <div class='header-actions'>
    <a class='back-link' href='/u/{{user_id}}/'>&#8592; Sessions</a>
    <button class='btn-purple' onclick='doSummarize()'>Summarize</button>
    <button class='btn-danger'  onclick='endSession()'>End Session</button>
  </div>
</header>

<div class='main'>
  <div class='conv-pane'>
    <div class='messages' id='messages'>
      % for msg in history:
      <div class='msg {{msg["role"]}}'>
        <div class='msg-label'>{{msg["role"].capitalize()}}</div>
        <div class='bubble'>{{msg["content"]}}</div>
      </div>
      % end
      % if not history:
      <div style='color:#4a5568;text-align:center;margin-top:60px;font-size:14px'>Session started. Send a message to begin.</div>
      % end
    </div>
    <div class='input-area'>
      <textarea id='inp' placeholder='Type a message… (Enter to send, Shift+Enter for newline)' rows='1'></textarea>
      <button class='btn-primary' onclick='sendMessage()'>Send</button>
      <span id='spinner'>&#9203;</span>
    </div>
  </div>

  <div class='activity-pane'>
    <div class='activity-header'>Memory &amp; LLM Activity</div>
    <div class='activity-list' id='activity-list'>
      <div class='ae-empty'>Activity will appear here as you chat.</div>
    </div>
  </div>
</div>

<script>
const USER_ID = {{!json.dumps(user_id)}};
const SID     = {{!json.dumps(session_id)}};
const BASE    = '/u/' + USER_ID + '/chat/' + SID;

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function renderActivity(entries) {
  const el = document.getElementById('activity-list');
  if (!entries || !entries.length) { el.innerHTML = "<div class='ae-empty'>No activity recorded.</div>"; return; }
  el.innerHTML = entries.map(e =>
    `<div class='ae ${esc(e.type)}'><div class='ae-type'>${esc(e.type)}</div><div class='ae-msg'>${esc(e.msg)}</div>${e.detail ? `<div class='ae-detail'>${esc(e.detail)}</div>` : ''}</div>`
  ).join('');
  el.scrollTop = el.scrollHeight;
}
function addMsg(role, content) {
  const msgs = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = `<div class='msg-label'>${role.charAt(0).toUpperCase()+role.slice(1)}</div><div class='bubble'>${esc(content)}</div>`;
  msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight;
}
function addNote(content) {
  const msgs = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg system-note';
  d.innerHTML = `<div class='bubble'>${esc(content)}</div>`;
  msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight;
}
function busy(on) { document.getElementById('spinner').className = on ? 'on' : ''; }

async function sendMessage() {
  const inp = document.getElementById('inp');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = ''; inp.style.height = '';
  addMsg('user', text); busy(true);
  try {
    const r = await fetch(BASE + '/message', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({input:text})});
    const d = await r.json();
    addMsg('assistant', d.reply);
    renderActivity(d.activity);
    if (d.extraction_triggered) addNote('Background memory extraction scheduled (every ' + d.extraction_every + ' turns)');
  } catch(e) { addMsg('assistant', 'Error: ' + e.message); }
  finally { busy(false); }
}
async function doSummarize() {
  addNote('Running "Summarize Information"…'); busy(true);
  try {
    const r = await fetch(BASE + '/summarize', {method:'POST'});
    const d = await r.json();
    renderActivity(d.activity);
    if (d.summary) addNote('Curated Summary:\\n\\n' + d.summary);
  } catch(e) { addNote('Summarize failed: ' + e.message); }
  finally { busy(false); }
}
function endSession() {
  if (!confirm('End this session? Final memory extraction will run and you will return to the session picker.')) return;
  busy(true);
  fetch(BASE + '/end', {method:'POST'}).finally(() => { window.location.href = '/u/' + USER_ID + '/'; });
}
document.getElementById('inp').addEventListener('input', function() {
  this.style.height = 'auto'; this.style.height = Math.min(this.scrollHeight, 140) + 'px';
});
document.getElementById('inp').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
window.addEventListener('load', () => { const m = document.getElementById('messages'); m.scrollTop = m.scrollHeight; });
</script>
</body></html>
"""
)


# ── Bottle routes ─────────────────────────────────────────────────────────────

app = bottle.Bottle()


@app.route("/", method="GET")
def index():
    users = run_async(_list_users())
    return bottle.template(USER_PICKER_HTML, users=users)


@app.route("/user/new", method="POST")
def user_new():
    user_id = (request.forms.get("user_id") or "").strip()
    if not user_id or not _USER_ID_RE.match(user_id):
        redirect("/")
    redirect(f"/u/{user_id}/")


@app.route("/u/<user_id>/", method="GET")
def session_picker(user_id):
    sessions = run_async(_list_sessions(user_id))
    return bottle.template(SESSION_PICKER_HTML, user_id=user_id, sessions=sessions)


@app.route("/u/<user_id>/session/new", method="POST")
def session_new(user_id):
    redirect(f"/u/{user_id}/chat/{uuid.uuid4()}")


@app.route("/u/<user_id>/session/select", method="POST")
def session_select(user_id):
    session_id = (request.forms.get("session_id") or "").strip()
    redirect(f"/u/{user_id}/chat/{session_id}" if session_id else f"/u/{user_id}/")


@app.route("/u/<user_id>/chat/<session_id>", method="GET")
def chat_page(user_id, session_id):
    if session_id not in _session_messages:
        prior = run_async(_load_history(session_id))
        _session_messages[session_id] = list(prior)
    history = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in _session_messages[session_id]
    ]
    return bottle.template(CHAT_PAGE_HTML, user_id=user_id, session_id=session_id, history=history, json=json)


@app.route("/u/<user_id>/chat/<session_id>/message", method="POST")
def chat_message(user_id, session_id):
    response.content_type = "application/json"
    body = request.json or {}
    user_input = (body.get("input") or "").strip()
    if not user_input:
        return json.dumps({"error": "empty input"})

    reply, activity = run_async(_invoke(user_id, session_id, user_input))

    _session_messages.setdefault(session_id, [])
    _session_messages[session_id] += [HumanMessage(content=user_input), AIMessage(content=reply)]
    _session_turns[session_id] = _session_turns.get(session_id, 0) + 1
    turn = _session_turns[session_id]
    extraction_triggered = False

    if turn % EXTRACTION_EVERY_N == 0:
        asyncio.run_coroutine_threadsafe(
            extract_and_store(list(_session_messages[session_id]), _store, _llm, session_id, user_id),
            _loop,
        )
        activity.append(_log(
            "EXTRACT_TRIGGER",
            f"Background memory extraction scheduled (turn {turn})",
            f"Extracting from last {min(10, len(_session_messages[session_id]))} messages",
        ))
        extraction_triggered = True

    return json.dumps({
        "reply": reply,
        "activity": activity,
        "turn": turn,
        "extraction_triggered": extraction_triggered,
        "extraction_every": EXTRACTION_EVERY_N,
    })


@app.route("/u/<user_id>/chat/<session_id>/summarize", method="POST")
def chat_summarize(user_id, session_id):
    response.content_type = "application/json"
    return json.dumps(run_async(_do_summarize(user_id, session_id)))


@app.route("/u/<user_id>/chat/<session_id>/end", method="POST")
def chat_end(user_id, session_id):
    messages = _session_messages.get(session_id, [])
    if messages:
        run_async(extract_and_store(messages, _store, _llm, session_id, user_id))
    _session_messages.pop(session_id, None)
    _session_turns.pop(session_id, None)
    response.content_type = "application/json"
    return json.dumps({"status": "ok"})


# ── threading WSGI server (stdlib only — no extra deps) ───────────────────────

class _ThreadingWSGIServer(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Bottle web UI for DragonflyDB conversational agent")
    p.add_argument("-H", "--host",     default="localhost", help="DragonflyDB host")
    p.add_argument("-p", "--port",     type=int, default=7900, help="DragonflyDB port")
    p.add_argument("-s", "--password", default=None)
    p.add_argument("-u", "--username", default=None)
    p.add_argument("--threshold",      type=float, default=0.15)
    p.add_argument("--ttl",            type=int,   default=86400, help="Session TTL in seconds")
    p.add_argument("--web-port",       type=int,   default=8080)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    redis_url = build_redis_url(args.host, args.port, args.username, args.password)

    print(f"Loading embedding model {EMBEDDING_MODEL}…")
    _vectorizer = HFTextVectorizer(model=EMBEDDING_MODEL)

    _cache = DragonflySemanticCache(
        name=CACHE_INDEX_NAME,
        vectorizer=_vectorizer,
        redis_url=redis_url,
        distance_threshold=args.threshold,
        overwrite=False,
    )
    _llm = ChatOpenAI(base_url=LLM_BASE_URL, model=LLM_MODEL, api_key="not-needed", temperature=0.25)

    print(f"Connecting to DragonflyDB at {args.host}:{args.port}…")
    run_async(_startup(redis_url, args.ttl))
    print("Ready.")

    print(f"\nOpen http://localhost:{args.web_port}/\n")
    httpd = make_server("0.0.0.0", args.web_port, app,
                        server_class=_ThreadingWSGIServer,
                        handler_class=WSGIRequestHandler)
    httpd.serve_forever()
