"""
LangGraph agent with two-tier memory (session + long-term) backed by DragonflyDB.

Short-term memory  — AsyncRedisSaver checkpointer, TTL-bounded per session thread
Long-term memory   — DragonflyRedisStore with semantic search; populated by LLM extraction
Extraction runs    — synchronously on exit + background asyncio task every 3rd response

Usage:
    python agent.py [-H host] [-p port] [-s password] [-u username]
                    [--threshold float] [--session <uuid>]
                    [--ttl <seconds>] [--no-background]
"""

import argparse
import asyncio
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore, PutOp
from langgraph.store.redis.aio import (
    AsyncRedisStore,
    AsyncSearchIndex,
    FilterQuery,
    Query,
    REDIS_KEY_SEPARATOR,
    RedisDocument,
    ULID,
    VectorQuery,
    _decode_ns,
    _ensure_string_or_literal,
    _namespace_to_text,
    _row_to_search_item,
    _token_escaper,
    _token_unescaper,
    get_text_at_path,
    tokenize_path,
)
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.query.filter import Tag
from redisvl.exceptions import RedisSearchError
from typing_extensions import TypedDict

from semanticcache import DragonflySemanticCache
from settings import (
    CACHE_DISTANCE_THRESHOLD,
    CACHE_INDEX_NAME,
    CATCHALL_CACHE_SESSION_TTL,
    CATCHALL_CACHE_ALL_TTL,
    DEDUP_THRESHOLD,
    DRAGONFLY_HOST,
    DRAGONFLY_PASSWORD,
    DRAGONFLY_PORT,
    DRAGONFLY_USERNAME,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    EXTRACTION_CHUNK_BUDGET,
    EXTRACTION_EVERY_N,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE_EXTRACTION,
    PROMPT_AI_TOKEN_LIMIT,
    PROMPT_HUMAN_CHAR_LIMIT,
    SESSION_TTL_SECONDS,
    TOKEN_LIMIT,
    TOPK_DAY_TTL,
    TOPK_DECAY,
    TOPK_DEPTH,
    TOPK_K,
    TOPK_MONTH_TTL,
    TOPK_SESSION_KEY,
    TOPK_USER_KEY,
    TOPK_WIDTH,
    TOPK_YEAR_TTL,
    build_redis_url,
)

_initialized_topk_keys: set[str] = set()
EMPTY_RESPONSE_PLACEHOLDER = (
    "it appears I either have nothing to say, or require more tokens to complete my processing of that request"
)

_CATCHALL_PATTERNS = (
    "what do you know about me",
    "what have i told you",
    "remind me what information we have exchanged",
    "what did we talk about",
    "summarize what you know",
    "what do you remember",
    "tell me everything you know",
    "what have we discussed",
)


def _user_ns(user_id: str) -> tuple[str, ...]:
    return ("user", user_id, "long_term")


def _user_session_ns(user_id: str, session_id: str) -> tuple[str, ...]:
    return ("user", user_id, "session", session_id)


_EDIT_PATTERNS = (
    "actually,",
    "that's not right",
    "that's wrong",
    "that's incorrect",
    "you're wrong",
    "let me correct",
    "to clarify,",
    "actually i ",
    "actually my ",
    "i prefer ",
    "my preference is",
    "please remember that",
    "please note that",
    "forget that",
    "i should clarify",
    "to be clear,",
    "please update that",
    "please correct that",
)

SESSION_EDIT_PROMPT = (
    "The user is correcting or stating a preference/fact about themselves. "
    "Extract the corrected fact or preference as a single clear statement. "
    "Return ONLY valid JSON:\n"
    '{"text": "the fact or preference"}'
)


def _is_catchall(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _CATCHALL_PATTERNS)


def _is_edit_intent(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _EDIT_PATTERNS)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _chunk_messages(messages: list[BaseMessage], budget: int = EXTRACTION_CHUNK_BUDGET) -> list[list[BaseMessage]]:
    chunks: list[list[BaseMessage]] = []
    chunk: list[BaseMessage] = []
    count = 0
    for msg in messages:
        t = _estimate_tokens(msg.content)
        if count + t > budget and chunk:
            chunks.append(chunk)
            chunk, count = [], 0
        chunk.append(msg)
        count += t
    if chunk:
        chunks.append(chunk)
    return chunks

def _compress_prompt_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Enforce per-turn size limits before sending history to the LLM.

    - Each AI message is truncated to PROMPT_AI_TOKEN_LIMIT tokens (char estimate).
    - Oldest human turns are dropped (along with their paired AI reply) until the
      total human-message character count fits within PROMPT_HUMAN_CHAR_LIMIT.
      If the newest human message alone exceeds the limit, it is trimmed from the start.
    """
    ai_char_limit = PROMPT_AI_TOKEN_LIMIT * 4

    # Pass 1: cap each AI message
    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and len(msg.content) > ai_char_limit:
            result.append(AIMessage(content=msg.content[:ai_char_limit] + "…"))
        else:
            result.append(msg)

    # Pass 2: drop/trim oldest human turns until human total <= PROMPT_HUMAN_CHAR_LIMIT
    human_total = sum(len(m.content) for m in result if isinstance(m, HumanMessage))
    excess = human_total - PROMPT_HUMAN_CHAR_LIMIT
    i = 0
    while i < len(result) and excess > 0:
        if isinstance(result[i], HumanMessage):
            c = len(result[i].content)
            if c <= excess:
                result.pop(i)
                if i < len(result) and isinstance(result[i], AIMessage):
                    result.pop(i)
                excess -= c
            else:
                result[i] = HumanMessage(content=result[i].content[excess:])
                excess = 0
                i += 1
        else:
            i += 1

    return result


EXTRACTION_PROMPT = (
    "Given this conversation excerpt, extract facts, user preferences, and notable topics. "
    "For each fact and preference assign a scope:\n"
    "  'user'    — stable info that should persist across all future sessions "
    "(e.g. name, job, location, permanent preferences)\n"
    "  'session' — time-sensitive or context-specific, relevant only to this conversation "
    "(e.g. current news concerns, today's tasks, temporary context)\n"
    "Return ONLY valid JSON:\n"
    '{"facts": [{"text": "...", "scope": "user|session"}], '
    '"preferences": [{"text": "...", "scope": "user|session"}], '
    '"topics": ["..."], "summary": "..."}'
)


class DragonflyRedisStore(AsyncRedisStore):
    r"""AsyncRedisStore patched for Dragonfly Search compatibility.

    Dragonfly Search doesn't support backslash-escaped dots in TEXT field exact
    queries (e.g. @prefix:user\.long_term) — same class of issue as the
    VECTOR_RANGE fix in DragonflySemanticCache.

    Fix: use TAG field for prefix, store raw (unescaped) namespace values
    (e.g. "user.long_term"), and query with TAG syntax @prefix:{user\.long_term}
    — Dragonfly handles the backslash-escaped dot correctly inside TAG braces.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client=None,
        index=None,
        connection_args=None,
        ttl=None,
        cluster_mode=None,
        store_prefix: str = "store",
        vector_prefix: str = "store_vectors",
    ) -> None:
        connection_args = {"socket_keepalive": True, **(connection_args or {})}
        super().__init__(
            redis_url,
            redis_client=redis_client,
            index=index,
            connection_args=connection_args,
            ttl=ttl,
            cluster_mode=cluster_mode,
            store_prefix=store_prefix,
            vector_prefix=vector_prefix,
        )
        # Patch store_index: change prefix field from TEXT → TAG
        self.store_index = AsyncSearchIndex.from_dict(
            {
                "index": {
                    "name": self.store_prefix,
                    "prefix": self.store_prefix + REDIS_KEY_SEPARATOR,
                    "storage_type": "json",
                },
                "fields": [
                    {"name": "prefix", "type": "tag"},
                    {"name": "key", "type": "tag"},
                    {"name": "created_at", "type": "numeric"},
                    {"name": "updated_at", "type": "numeric"},
                    {"name": "ttl_minutes", "type": "numeric"},
                    {"name": "expires_at", "type": "numeric"},
                ],
            },
            redis_client=self._redis,
        )
        # Patch vector_index prefix field if semantic search is enabled
        if self.index_config:
            index_dict = dict(self.index_config)
            vector_schema: dict[str, Any] = {
                "index": {
                    "name": self.vector_prefix,
                    "prefix": self.vector_prefix + REDIS_KEY_SEPARATOR,
                    "storage_type": index_dict.get("vector_storage_type", "json"),
                },
                "fields": [
                    {"name": "prefix", "type": "tag"},
                    {"name": "key", "type": "tag"},
                    {"name": "field_name", "type": "tag"},
                    {"name": "embedding", "type": "vector"},
                    {"name": "created_at", "type": "numeric"},
                    {"name": "updated_at", "type": "numeric"},
                    {"name": "ttl_minutes", "type": "numeric"},
                    {"name": "expires_at", "type": "numeric"},
                ],
            }
            for f in vector_schema["fields"]:
                if f["name"] == "embedding":
                    f["attrs"] = {
                        "algorithm": "flat",
                        "datatype": "float32",
                        "dims": self.index_config["dims"],
                        "distance_metric": {
                            "cosine": "COSINE",
                            "inner_product": "IP",
                            "l2": "L2",
                        }[_ensure_string_or_literal(index_dict.get("distance_type", "cosine"))],
                    }
                    if "ann_index_config" in index_dict:
                        f["attrs"].update(index_dict["ann_index_config"])
                    break
            self.vector_index = AsyncSearchIndex.from_dict(vector_schema, redis_client=self._redis)

    def _raw_ns(self, namespace: tuple[str, ...]) -> str:
        """Raw unescaped namespace for storing in documents."""
        return ".".join(namespace)

    def _tag_filter(self, namespace: tuple[str, ...]) -> str:
        """TAG query term for exact namespace match: {escaped_ns}."""
        return f"{{{_namespace_to_text(namespace)}}}"

    async def _aprepare_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> tuple:
        """Prepare PUT queries storing raw (unescaped) namespace in documents."""
        dedupped_ops: dict = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for op in dedupped_ops.values():
            (deletes if op.value is None else inserts).append(op)

        operations: list[RedisDocument] = []
        to_embed: list[tuple[str, str, str, str]] = []

        for op in deletes:
            query = f"@prefix:{self._tag_filter(op.namespace)} @key:{{{_token_escaper.escape(op.key)}}}"
            results = await self.store_index.search(query)
            for doc in results.docs:
                await self._redis.delete(doc.id)

        for op in inserts:
            now = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
            ttl_minutes = expires_at = None
            if op.ttl is not None:
                ttl_minutes = op.ttl
                expires_at = int((datetime.now(timezone.utc) + timedelta(minutes=op.ttl)).timestamp())

            operations.append(RedisDocument(
                prefix=self._raw_ns(op.namespace),   # raw, not escaped
                key=op.key,
                value=op.value,
                created_at=now,
                updated_at=now,
                ttl_minutes=ttl_minutes,
                expires_at=expires_at,
            ))

            if self.index_config and op.index is not False:
                paths = (
                    self.index_config["__tokenized_fields"]
                    if op.index is None
                    else [(ix, tokenize_path(ix)) for ix in op.index]
                )
                for path, tokenized_path in paths:
                    for text in get_text_at_path(op.value, tokenized_path):
                        to_embed.append((self._raw_ns(op.namespace), op.key, path, text))

        embedding_request = ("", to_embed) if to_embed else None
        return operations, embedding_request

    async def _batch_put_ops(self, put_ops: Sequence[tuple[int, PutOp]]) -> None:
        operations, embedding_request = await self._aprepare_batch_PUT_queries(put_ops)

        for _, op in put_ops:
            query = f"@prefix:{self._tag_filter(op.namespace)} @key:{{{_token_escaper.escape(op.key)}}}"
            results = await self.store_index.search(query)
            if self.cluster_mode:
                for doc in results.docs:
                    await self._redis.delete(doc.id)
                if self.index_config:
                    for doc in (await self.vector_index.search(query)).docs:
                        await self._redis.delete(doc.id)
            else:
                pipeline = self._redis.pipeline(transaction=True)
                for doc in results.docs:
                    pipeline.delete(doc.id)
                if self.index_config:
                    for doc in (await self.vector_index.search(query)).docs:
                        pipeline.delete(doc.id)
                if pipeline.command_stack:
                    await pipeline.execute()

        doc_ids: dict[tuple[str, str], str] = {}
        store_docs: list[Any] = []
        store_keys: list[str] = []
        ttl_tracking: dict[str, tuple[list[str], Optional[float]]] = {}

        for _, op in put_ops:
            if op.value is not None:
                doc_id = str(ULID())
                raw_ns = self._raw_ns(op.namespace)
                doc_ids[(raw_ns, op.key)] = doc_id
                if hasattr(op, "ttl") and op.ttl is not None:
                    main_key = f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
                    ttl_tracking[main_key] = ([], op.ttl)

        for doc in operations:
            doc_id = doc_ids[(doc["prefix"], doc["key"])]
            doc.pop("ttl_minutes", None)
            doc.pop("expires_at", None)
            store_docs.append(doc)
            store_keys.append(f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{doc_id}")

        if store_docs:
            if self.cluster_mode:
                for i, item in enumerate(store_docs):
                    await self.store_index.load([item], keys=[store_keys[i]])
            else:
                await self.store_index.load(store_docs, keys=store_keys)

        if embedding_request and self.embeddings:
            _, text_params = embedding_request
            vectors = await self.embeddings.aembed_documents([t for _, _, _, t in text_params])
            vector_docs: list[dict[str, Any]] = []
            vector_keys: list[str] = []
            for (ns, key, path, _), vector in zip(text_params, vectors):
                doc_id = doc_ids[(ns, key)]
                vector_docs.append({
                    "prefix": ns,
                    "key": key,
                    "field_name": path,
                    "embedding": vector.tolist() if hasattr(vector, "tolist") else vector,
                    "created_at": datetime.now(timezone.utc).timestamp(),
                    "updated_at": datetime.now(timezone.utc).timestamp(),
                })
                vk = f"{self.vector_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
                vector_keys.append(vk)
                mk = f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
                if mk in ttl_tracking:
                    ttl_tracking[mk][0].append(vk)

            if vector_docs:
                if self.cluster_mode:
                    for i, item in enumerate(vector_docs):
                        await self.vector_index.load([item], keys=[vector_keys[i]])
                else:
                    await self.vector_index.load(vector_docs, keys=vector_keys)

        for mk, (related, ttl_m) in ttl_tracking.items():
            await self._apply_ttl_to_keys(mk, related, ttl_m)

    def _get_batch_GET_ops_queries(self, get_ops):
        ns_groups: dict = defaultdict(list)
        for idx, op in get_ops:
            ns_groups[op.namespace].append((idx, op.key))

        out = []
        for namespace, items in ns_groups.items():
            _, keys = zip(*items)
            # Use Tag filter with raw namespace — generates @prefix:{user\.long_term}
            prefix_filter = Tag("prefix") == self._raw_ns(namespace)
            filter_str = f"({prefix_filter} "
            filter_str += f"{Tag('key') == list(keys)})" if keys else ")"
            out.append((filter_str, [], namespace, items))
        return out

    def _get_batch_search_queries(self, search_ops):
        queries, embedding_requests = [], []
        for idx, op in search_ops:
            conditions = []
            if op.namespace_prefix:
                escaped = _namespace_to_text(op.namespace_prefix)
                conditions.append(f"@prefix:{{{escaped}*}}")   # TAG wildcard inside braces
            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))
            query = " ".join(conditions) if conditions else "*"
            limit = op.limit if op.limit is not None else 10
            offset = op.offset if op.offset is not None else 0
            queries.append((query, [limit, offset], limit, offset))
        return queries, embedding_requests

    async def _batch_search_ops(self, search_ops, results) -> None:
        queries, embedding_requests = self._get_batch_search_queries(search_ops)
        query_vectors: dict = {}
        if embedding_requests and self.embeddings:
            vectors = await self.embeddings.aembed_documents([q for _, q in embedding_requests])
            query_vectors = dict(zip([i for i, _ in embedding_requests], vectors))

        for (idx, op), (query_str, _, limit, offset) in zip(search_ops, queries):
            if op.query and idx in query_vectors:
                vector = query_vectors[idx]
                escaped = _namespace_to_text(op.namespace_prefix)
                vector_query = VectorQuery(
                    vector=vector.tolist() if hasattr(vector, "tolist") else vector,
                    vector_field_name="embedding",
                    filter_expression=f"@prefix:{{{escaped}*}}",   # TAG wildcard
                    return_fields=["prefix", "key", "vector_distance"],
                    num_results=limit,
                )
                vector_query.paging(offset, limit)
                try:
                    vector_results_docs = await self.vector_index.query(vector_query)
                except RedisSearchError:
                    # Stale pool connection (ECONNRESET); retry once.
                    await asyncio.sleep(0.1)
                    vector_results_docs = await self.vector_index.query(vector_query)

                result_map: dict = {}
                pipeline = self._redis.pipeline(transaction=False)
                for doc in vector_results_docs:
                    doc_id = doc.get("id") if isinstance(doc, dict) else getattr(doc, "id", None)
                    if doc_id:
                        uuid_part = doc_id.split(":")[1]
                        sk = f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{uuid_part}"
                        result_map[sk] = doc
                        pipeline.json().get(sk)
                store_docs_raw = await pipeline.execute()

                items = []
                for sk, store_doc in zip(result_map.keys(), store_docs_raw):
                    if not store_doc:
                        continue
                    vr = result_map[sk]
                    dist = vr.get("vector_distance") if isinstance(vr, dict) else getattr(vr, "vector_distance", 0)
                    score = (1.0 - float(dist)) if dist is not None else 0.0
                    if not isinstance(store_doc, dict):
                        try:
                            store_doc = json.loads(store_doc)
                        except Exception:
                            continue
                    if not isinstance(store_doc, dict):
                        continue
                    store_doc["vector_distance"] = dist
                    if op.filter:
                        value = store_doc.get("value", {})
                        if not all(
                            (value.get(k) in v if isinstance(v, list) else value.get(k) == v)
                            for k, v in op.filter.items()
                        ):
                            continue
                    items.append(_row_to_search_item(_decode_ns(store_doc["prefix"]), store_doc, score=score))
                results[idx] = items

            else:
                res = await self.store_index.search(Query(query_str).paging(offset, limit))
                items = []
                for doc in res.docs:
                    data = json.loads(doc.json)
                    if op.filter:
                        value = data.get("value", {})
                        if not all(
                            (value.get(k) in v if isinstance(v, list) else value.get(k) == v)
                            for k, v in op.filter.items()
                        ):
                            continue
                    items.append(_row_to_search_item(_decode_ns(data["prefix"]), data))
                results[idx] = items

    async def _batch_list_namespaces_ops(self, list_ops, results) -> None:
        for idx, op in list_ops:
            base_query = "*"
            if op.match_conditions:
                conditions = []
                for cond in op.match_conditions:
                    ns_text = _namespace_to_text(cond.path, handle_wildcards=True)
                    if cond.match_type == "prefix":
                        conditions.append(f"@prefix:{{{ns_text}*}}")
                    elif cond.match_type == "suffix":
                        conditions.append(f"@prefix:*{ns_text}")
                if conditions:
                    base_query = " ".join(conditions)

            try:
                raw = await self._redis.execute_command(
                    "FT.AGGREGATE", self.store_prefix,
                    base_query,
                    "LOAD", "1", "prefix",
                    "GROUPBY", "1", "@prefix",
                    "LIMIT", "0", "10000",
                    "DIALECT", "2",
                )
            except Exception:
                results[idx] = []
                continue

            # raw[0] is result count; raw[1:] are rows of alternating field/value pairs
            namespaces: set = set()
            for row in raw[1:]:
                if not isinstance(row, (list, tuple)):
                    continue
                for i in range(0, len(row) - 1, 2):
                    field = row[i]
                    val = row[i + 1]
                    if (field.decode() if isinstance(field, bytes) else field) == "prefix":
                        prefix_str = val.decode() if isinstance(val, bytes) else val
                        ns = tuple(prefix_str.split("."))
                        if op.max_depth is not None:
                            ns = ns[:op.max_depth]
                        namespaces.add(ns)
                        break

            sorted_ns = sorted(namespaces)
            if op.limit or op.offset:
                o = op.offset or 0
                l = op.limit or 10
                sorted_ns = sorted_ns[o:o + l]
            results[idx] = sorted_ns


class HFVectorizerEmbeddings(Embeddings):
    """Adapts redisvl HFTextVectorizer to the LangChain Embeddings interface
    so DragonflyRedisStore can use the same local model as the semantic cache."""

    def __init__(self, vectorizer: HFTextVectorizer) -> None:
        self._v = vectorizer

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._v.embed_many(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._v.embed(text)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    memories: list[str]          # retrieved long-term context, injected fresh each turn
    catchall_scope: Optional[str]  # "session" | "all" | None — set before ainvoke on catchall queries


async def _setup_topk_keys(redis_client) -> None:
    for key in (TOPK_SESSION_KEY, TOPK_USER_KEY):
        if not await redis_client.exists(key):
            try:
                await redis_client.execute_command(
                    "TOPK.RESERVE", key, TOPK_K, TOPK_WIDTH, TOPK_DEPTH, TOPK_DECAY
                )
            except Exception:
                pass
        _initialized_topk_keys.add(key)


async def _ensure_topk_key(redis_client, key: str, ttl: int) -> None:
    if key in _initialized_topk_keys:
        return
    if not await redis_client.exists(key):
        try:
            await redis_client.execute_command(
                "TOPK.RESERVE", key, TOPK_K, TOPK_WIDTH, TOPK_DEPTH, TOPK_DECAY
            )
            await redis_client.expire(key, ttl)
        except Exception:
            pass
    _initialized_topk_keys.add(key)


async def _record_tokens(redis_client, user_id: str, session_id: str, tokens: int) -> None:
    if tokens <= 0:
        return
    now = datetime.now(timezone.utc)
    day_str   = now.strftime("%Y%m%d")
    month_str = now.strftime("%Y%m")
    year_str  = now.strftime("%Y")
    session_entry = f"{user_id}:{session_id}"

    time_keys = [
        (f"topk:tokens:session:day:{day_str}",    session_entry, TOPK_DAY_TTL),
        (f"topk:tokens:user:day:{day_str}",        user_id,       TOPK_DAY_TTL),
        (f"topk:tokens:session:month:{month_str}", session_entry, TOPK_MONTH_TTL),
        (f"topk:tokens:user:month:{month_str}",   user_id,       TOPK_MONTH_TTL),
        (f"topk:tokens:session:year:{year_str}",  session_entry, TOPK_YEAR_TTL),
        (f"topk:tokens:user:year:{year_str}",     user_id,       TOPK_YEAR_TTL),
    ]
    for key, _, ttl in time_keys:
        await _ensure_topk_key(redis_client, key, ttl)

    pipe = redis_client.pipeline(transaction=False)
    pipe.execute_command("TOPK.INCRBY", TOPK_SESSION_KEY, session_entry, tokens)
    pipe.execute_command("TOPK.INCRBY", TOPK_USER_KEY, user_id, tokens)
    for key, entry, _ in time_keys:
        pipe.execute_command("TOPK.INCRBY", key, entry, tokens)
    await pipe.execute()


async def _fetch_memories(
    store: BaseStore,
    ns: tuple,
    session_ns: tuple,
    query: str,
    catchall: bool = False,
) -> tuple[list[tuple[str, str]], int, int]:
    """Search both namespaces session-first, deduplicate by text.

    Returns (items, lt_count, sess_count) where items is a list of (text, key) pairs
    and the counts are pre-dedup sizes for observability logging.
    """
    if catchall:
        lt_results = await store.asearch(ns, limit=20)
        sess_results = await store.asearch(session_ns, limit=20)
    else:
        lt_results = await store.asearch(ns, query=query, limit=3)
        sess_results = await store.asearch(session_ns, query=query, limit=3)
    seen: set[str] = set()
    items: list[tuple[str, str]] = []
    for r in [*sess_results, *lt_results]:
        txt = r.value.get("text", "")
        if txt and txt not in seen:
            seen.add(txt)
            items.append((txt, r.key or ""))
    return items, len(lt_results), len(sess_results)


def build_graph(llm: ChatOpenAI, cache: DragonflySemanticCache, user_id: str, session_id: str, redis_client) -> StateGraph:
    ns = _user_ns(user_id)
    session_ns = _user_session_ns(user_id, session_id)

    async def retrieve_memories(state: AgentState, *, store: BaseStore) -> dict:
        if not state["messages"]:
            return {"memories": []}
        query = state["messages"][-1].content
        scope = state.get("catchall_scope")

        if scope == "session":
            results = await store.asearch(session_ns, limit=20)
            return {"memories": [r.value["text"] for r in results if r.value.get("text")]}

        catchall = scope == "all" or _is_catchall(query)
        items, _lt_count, _ss_count = await _fetch_memories(store, ns, session_ns, query, catchall=catchall)
        return {"memories": [txt for txt, _key in items]}

    async def chat(state: AgentState) -> dict:
        user_msg = state["messages"][-1].content
        scope = state.get("catchall_scope")

        # Resolve cache scope and TTL based on clarified intent.
        if scope == "session":
            cache_session_id = session_id
            cache_ttl = CATCHALL_CACHE_SESSION_TTL
        elif scope == "all":
            cache_session_id = None
            cache_ttl = CATCHALL_CACHE_ALL_TTL
        else:
            cache_session_id = None
            cache_ttl = None

        hits = cache.check(
            prompt=user_msg,
            user_id=user_id,
            session_id=cache_session_id,
        )
        if hits:
            return {"messages": [AIMessage(content=hits[0]["response"])]}

        system_parts = ["You are a concise assistant with an excellent memory. Be brief and succinct. Limit examples to one or two. Avoid repeating yourself."]
        if scope == "session":
            system_parts.append(
                "\nThe user wants a summary of this conversation only. "
                "Summarize the key topics, facts shared, and any preferences expressed so far in this session."
            )
            if state.get("memories"):
                system_parts.append(
                    "\nFacts and preferences explicitly noted this session:\n"
                    + "\n".join(f"- {m}" for m in state["memories"])
                )
        elif scope == "all":
            if state.get("memories"):
                system_parts.append(
                    "\nThe user wants a summary of everything you know about them across all sessions. "
                    "Here are all stored facts and preferences:\n"
                    + "\n".join(f"- {m}" for m in state["memories"])
                )
        elif state.get("memories"):
            system_parts.append(
                "\nRelevant context from past sessions:\n"
                + "\n".join(f"- {m}" for m in state["memories"])
            )

        response = await llm.ainvoke(
            [SystemMessage(content="\n".join(system_parts))]
            + _compress_prompt_messages(list(state["messages"]))
        )
        content = response.content or EMPTY_RESPONSE_PLACEHOLDER
        tokens = (response.usage_metadata or {}).get("total_tokens", 0)
        if not tokens:
            tokens = _estimate_tokens(user_msg) + _estimate_tokens(content)
        await _record_tokens(redis_client, user_id, session_id, tokens)
        if response.content:
            cache.store(
                prompt=user_msg,
                response=response.content,
                user_id=user_id,
                session_id=cache_session_id,
                ttl=cache_ttl,
            )
        return {"messages": [AIMessage(content=content)]}

    g = StateGraph(AgentState)
    g.add_node("retrieve_memories", retrieve_memories)
    g.add_node("chat", chat)
    g.add_edge(START, "retrieve_memories")
    g.add_edge("retrieve_memories", "chat")
    g.add_edge("chat", END)
    return g


async def extract_and_store(
    messages: list[BaseMessage],
    store: BaseStore,
    llm: ChatOpenAI,
    session_id: str,
    user_id: str,
    cache: DragonflySemanticCache,
    session_ttl_minutes: int = 1440,
) -> None:
    if len(messages) < 2:
        return

    ns = _user_ns(user_id)
    session_ns = _user_session_ns(user_id, session_id)

    def _item_scope(item) -> tuple[str, str]:
        """Return (text, scope) from either a {text, scope} dict or a plain string."""
        if isinstance(item, dict):
            scope = item.get("scope", "user")
            return item.get("text", ""), scope if scope in ("user", "session") else "user"
        return str(item), "user"

    async def _extract_chunk(chunk: list[BaseMessage]) -> tuple[list[tuple[str, str, str]], dict]:
        conv = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in chunk
        )
        try:
            resp = await llm.ainvoke([
                SystemMessage(content=EXTRACTION_PROMPT),
                HumanMessage(content=conv),
            ])
            data = json.loads(resp.content)
        except Exception:
            return [], {}
        ts = str(int(time.time()))
        entries: list[tuple[str, str, str]] = []
        for i, item in enumerate(data.get("facts", [])):
            text, scope = _item_scope(item)
            if text:
                entries.append((f"fact_{ts}_{i}", text, scope))
        for i, item in enumerate(data.get("preferences", [])):
            text, scope = _item_scope(item)
            if text:
                entries.append((f"pref_{ts}_{i}", text, scope))
        if data.get("summary"):
            entries.append((f"summary_{ts}", data["summary"], "user"))
        return entries, data

    chunks = _chunk_messages(messages)
    chunk_results = await asyncio.gather(*[_extract_chunk(c) for c in chunks])

    ts = str(int(time.time()))
    cache_parts: list[str] = []

    for entries, data in chunk_results:
        for key, text, scope in entries:
            target_ns = ns if scope == "user" else session_ns
            existing = await store.asearch(target_ns, query=text, limit=1)
            if existing and existing[0].score >= DEDUP_THRESHOLD:
                continue
            put_kwargs: dict = {"ttl": session_ttl_minutes} if scope == "session" else {}
            await store.aput(
                target_ns,
                f"{session_id}_{key}",
                {"text": text, "session_id": session_id, "ts": ts, "scope": scope},
                **put_kwargs,
            )

        parts = []
        if data.get("facts"):
            parts.append("Facts:\n" + "\n".join(
                f"- {f['text'] if isinstance(f, dict) else f}" for f in data["facts"]
            ))
        if data.get("preferences"):
            parts.append("Preferences:\n" + "\n".join(
                f"- {p['text'] if isinstance(p, dict) else p}" for p in data["preferences"]
            ))
        if data.get("summary"):
            parts.append(f"Summary: {data['summary']}")
        cache_parts.extend(parts)

    if cache_parts:
        cache.refresh_catchall_cache(
            session_id, user_id, "\n\n".join(cache_parts), _CATCHALL_PATTERNS,
            ttl=CATCHALL_CACHE_SESSION_TTL,
        )


async def _extract_and_store_session_edit(
    user_msg: str,
    assistant_reply: str,
    store: BaseStore,
    llm: ChatOpenAI,
    session_id: str,
    user_id: str,
    session_ttl_minutes: int,
) -> None:
    session_ns = _user_session_ns(user_id, session_id)
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=SESSION_EDIT_PROMPT),
            HumanMessage(content=f"User: {user_msg}\nAssistant: {assistant_reply}"),
        ])
        data = json.loads(resp.content)
        text = data.get("text", "").strip()
    except Exception:
        return
    if not text:
        return
    ts = str(int(time.time()))
    await store.aput(
        session_ns,
        f"edit_{ts}",
        {"text": text, "session_id": session_id, "ts": ts},
        ttl=session_ttl_minutes,
    )


async def run(args: argparse.Namespace) -> None:
    redis_url = build_redis_url(args.host, args.port, args.username, args.password)
    session_id = args.session or str(uuid.uuid4())
    user_id = args.user
    config = {"configurable": {"thread_id": session_id}}

    vectorizer = HFTextVectorizer(model=EMBEDDING_MODEL)
    cache = DragonflySemanticCache(
        name=CACHE_INDEX_NAME,
        vectorizer=vectorizer,
        redis_url=redis_url,
        distance_threshold=args.threshold,
        overwrite=True,
        filterable_fields=[
            {"name": "user_id", "type": "tag"},
            {"name": "session_id", "type": "tag"},
        ],
    )
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key="not-needed",
        temperature=LLM_TEMPERATURE_EXTRACTION,
        max_tokens=TOKEN_LIMIT,
    )

    session_ttl_minutes = max(1, args.ttl // 60)
    ttl_cfg = {"default_ttl": args.ttl, "refresh_on_read": True}
    index_cfg = {"embed": HFVectorizerEmbeddings(vectorizer), "dims": EMBEDDING_DIMS, "fields": ["text"]}

    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl_cfg) as checkpointer:
        await checkpointer.asetup()
        async with DragonflyRedisStore.from_conn_string(redis_url, index=index_cfg) as store:
            store.setup()
            await _setup_topk_keys(store._redis)

            compiled = build_graph(llm, cache, user_id, session_id, store._redis).compile(checkpointer=checkpointer, store=store)

            print(f"User    : {user_id}")
            print(f"Session : {session_id}")
            print(f"Dragonfly: {args.host}:{args.port}  |  Session TTL: {args.ttl}s")
            print("Type 'END' to quit.\n")

            messages: list[BaseMessage] = []
            turn_count = 0
            pending: list[asyncio.Task] = []

            try:
                while True:
                    user_input = input("You> ").strip()
                    if not user_input or user_input.lower() == "end":
                        break

                    catchall_scope = None
                    if _is_catchall(user_input):
                        print(
                            "\nAssistant: Do you want me to reflect on this session alone, "
                            "or on all the discussions we have had? (session/all)"
                        )
                        scope_raw = input("You> ").strip().lower()
                        catchall_scope = "session" if "session" in scope_raw else "all"

                    result = await compiled.ainvoke(
                        {"messages": [HumanMessage(content=user_input)], "memories": [], "catchall_scope": catchall_scope},
                        config=config,
                    )
                    reply = result["messages"][-1].content
                    messages += [HumanMessage(content=user_input), AIMessage(content=reply)]
                    turn_count += 1

                    print(f"\nAssistant: {reply}\n")

                    if _is_edit_intent(user_input):
                        pending.append(asyncio.create_task(
                            _extract_and_store_session_edit(
                                user_input, reply, store, llm,
                                session_id, user_id, session_ttl_minutes,
                            )
                        ))

                    if not args.no_background and turn_count % EXTRACTION_EVERY_N == 0:
                        pending.append(asyncio.create_task(
                            extract_and_store(list(messages), store, llm, session_id, user_id, cache, session_ttl_minutes)
                        ))
                        print("[Background memory extraction scheduled]")

            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            print("Extracting session memories...")
            await extract_and_store(messages, store, llm, session_id, user_id, cache, session_ttl_minutes)
            print("Memories saved.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangGraph agent with two-tier memory on DragonflyDB")
    p.add_argument("-H", "--host",     default=DRAGONFLY_HOST)
    p.add_argument("-p", "--port",     type=int, default=DRAGONFLY_PORT)
    p.add_argument("-s", "--password", default=DRAGONFLY_PASSWORD)
    p.add_argument("-u", "--username", default=DRAGONFLY_USERNAME)
    p.add_argument("--threshold",      type=float, default=CACHE_DISTANCE_THRESHOLD)
    p.add_argument("--session",        default=None, help="Resume a prior session by ID")
    p.add_argument("--ttl",            type=int, default=SESSION_TTL_SECONDS, help="Session memory TTL in seconds")
    p.add_argument("--user",           default="default", help="Stable user ID for memory namespace isolation")
    p.add_argument("--no-background",  action="store_true", dest="no_background")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
