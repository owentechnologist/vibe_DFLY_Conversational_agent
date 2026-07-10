# How Memory Works in This Agent

This agent has three layers of memory, each serving a different purpose. Here's how they interact from the perspective of a running conversation.

---

## Layer 1 — Semantic Cache (session-specific, prompt-level deduplication)

Before the LLM is ever called, every user message is checked against a **semantic cache** stored in DragonflyDB (`DragonflySemanticCache`). The cache holds prior prompt/response pairs as vector embeddings. If your new message is semantically close enough to a cached prompt (within the `--threshold` cosine distance, default `0.15`), the cached response is returned immediately — no LLM call happens at all.

This is not "memory" in the conversational sense. It's a cost and latency optimization. The cache is local to a specific session for a user: if two different users ask essentially the same question, the second one will *not* receive the cached response.  If the same user in two different sessions asks the same question they will also NOT get the cached response.

**What it stores:** the prompt text and the LLM's response, indexed as a 768-dimension vector (`sentence-transformers/all-mpnet-base-v2`).

**Where it lives:** DragonflyDB, index name `llm_semantic_cache`.

**When it's consulted:** at the very start of every turn, before the graph runs.

**When it's written:** after every real LLM response, so the next similar question can be served from cache.

---

## Layer 2 — Session Checkpointer (short-term, per-conversation)

Once past the cache, the LangGraph graph runs. The graph uses `AsyncRedisSaver` as its **checkpointer** — this is LangGraph's built-in mechanism for persisting graph state between invocations.

Each conversation is identified by a **session ID** (a UUID). Every time you send a message, LangGraph saves the entire message list for that thread to Dragonfly. When you send the next message in the same session, it loads that checkpoint and appends your new message — giving the LLM the full conversation history without you having to re-send it.

**What it stores:** the full `AgentState` for the thread — every message in the conversation so far.

**Where it lives:** DragonflyDB (Redis). Keys are managed by the `AsyncRedisSaver` and scoped to the `thread_id`.

**TTL:** sessions expire after `--ttl` seconds (default 24 hours, `86400`). The TTL refreshes on each read, so an active conversation stays alive. An idle session expires naturally.

**Resuming a session:** pass `--session <uuid>` when starting the agent. The checkpointer will reload the prior message history and the conversation continues exactly where it left off.

---

## Layer 3 — Long-Term Memory Store (cross-session, extracted facts)

This is where the agent actually learns things about you across sessions. The agent uses `DragonflyRedisStore` (a patched `AsyncRedisStore`) as a **LangGraph store** — a key/value store with semantic search.

Long-term memories are not stored raw. An LLM extraction pass distills the conversation into structured entries: facts, preferences, topics, and a summary. Each item becomes its own document in the store under the namespace `("user", "long_term")`, keyed like `{session_id}_fact_<ts>_0`, `{session_id}_pref_<ts>_0`, etc.

**When extraction runs:**
- **Every X (configurable) LLM response** during a session — an `asyncio` background task fires silently while you keep chatting. You'll see `[Background memory extraction scheduled]` in the console.
- **On exit** — a final synchronous extraction pass runs before the process terminates, ensuring the session is fully committed.

**How memories are used at the start of a turn:**  
The `retrieve_memories` node runs before the `chat` node. It takes your latest message, embeds it, and does a semantic similarity search against the long-term store (top 3 results). Any matching memories are injected into the system prompt as bullet points under "Relevant context from past sessions." The LLM then answers with that context available.

**What it stores:** plain-text snippets, each with `text`, `session_id`, and `ts` fields. The `text` field is also vectorized so semantic search works.

**Where it lives:** DragonflyDB, namespace `user.long_term` (stored as a TAG field for Dragonfly Search compatibility — see `dragonfly-search-compat.md`).

---

## The Data Flow, Turn by Turn

```
You type a message
        │
        ▼
┌───────────────────────┐
│  Semantic Cache check  │ ──── hit? ──▶ return cached response immediately
└───────────────────────┘
        │ miss
        ▼
┌───────────────────────┐
│  LangGraph invocation  │
│  (session checkpoint   │ ◀── loads prior messages from AsyncRedisSaver
│   restored)           │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  retrieve_memories     │ ──── semantic search ──▶ DragonflyRedisStore
│  node                  │     (top 3 relevant facts from past sessions)
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  chat node             │ ──── LLM call with history + retrieved memories
└───────────────────────┘
        │
        ├──▶ cache.store(prompt, response)  [write to semantic cache]
        ├──▶ checkpoint saved               [AsyncRedisSaver writes new state]
        └──▶ every Xth turn (configurable): background extraction ──▶ DragonflyRedisStore
```

---

## Configuration Reference

| Flag | Default | Effect |
|---|---|---|
| `--session <uuid>` | new UUID | Resume a prior session |
| `--ttl <seconds>` | `86400` (24h) | Session checkpoint TTL |
| `--threshold <float>` | `0.15` | Semantic cache similarity threshold (lower = stricter) |
| `--no-background` | off | Disable mid-session background extraction |
| `-H / -p` | `localhost:7900` | DragonflyDB connection (agent) |

The semantic cache and the agent normally share the same cache, but could use separate caches if that proved useful.
