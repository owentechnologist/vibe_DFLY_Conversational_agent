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
from typing_extensions import TypedDict

from semanticcache import DragonflySemanticCache, build_redis_url

LLM_BASE_URL = "http://localhost:6060/v1"
LLM_MODEL = "qwen3.5-9b-glm5.1-distill-v1"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMS = 768  # all-mpnet-base-v2 output dimension
CACHE_INDEX_NAME = "llm_semantic_cache"
EXTRACTION_EVERY_N = 3

_CATCHALL_PATTERNS = (
    "what do you know about me",
    "what have i told you",
    "remind me what we",
    "what did we talk about",
    "summarize what you know",
    "what do you remember",
    "tell me everything you know",
    "what have we discussed",
)


def _user_ns(user_id: str) -> tuple[str, ...]:
    return ("user", user_id, "long_term")


def _is_catchall(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _CATCHALL_PATTERNS)

EXTRACTION_PROMPT = (
    "Given this conversation excerpt, extract persistent facts, user preferences, "
    "and notable topics. Return ONLY valid JSON:\n"
    '{"facts": ["..."], "preferences": ["..."], "topics": ["..."], "summary": "..."}'
)


class DragonflyRedisStore(AsyncRedisStore):
    """AsyncRedisStore patched for Dragonfly Search compatibility.

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
            query = f"@prefix:{self._tag_filter(op.namespace)} @key:{{{op.key}}}"
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
                        conditions.append(f"@prefix:{{{ns_text}*}}")   # TAG wildcard
                    elif cond.match_type == "suffix":
                        conditions.append(f"@prefix:*{ns_text}")        # TEXT fallback
                if conditions:
                    base_query = " ".join(conditions)

            try:
                res = await self.store_index.search(FilterQuery(filter_expression=base_query, return_fields=["prefix"]))
            except Exception:
                results[idx] = []
                continue

            namespaces: set = set()
            for doc in res.docs:
                if hasattr(doc, "prefix"):
                    ns = tuple(_token_unescaper.unescape(doc.prefix).split("."))
                    if op.max_depth is not None:
                        ns = ns[:op.max_depth]
                    namespaces.add(ns)

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
    memories: list[str]  # retrieved long-term context, injected fresh each turn


def build_graph(llm: ChatOpenAI, cache: DragonflySemanticCache, user_id: str) -> StateGraph:
    ns = _user_ns(user_id)

    async def retrieve_memories(state: AgentState, *, store: BaseStore) -> dict:
        if not state["messages"]:
            return {"memories": []}
        query = state["messages"][-1].content
        if _is_catchall(query):
            results = await store.asearch(ns, limit=20)
        else:
            results = await store.asearch(ns, query=query, limit=3)
        return {"memories": [r.value["text"] for r in results if r.value.get("text")]}

    async def chat(state: AgentState) -> dict:
        user_msg = state["messages"][-1].content

        hits = cache.check(prompt=user_msg)
        if hits:
            return {"messages": [AIMessage(content=hits[0]["response"])]}

        system_parts = ["You are a helpful assistant."]
        if state.get("memories"):
            system_parts.append(
                "\nRelevant context from past sessions:\n"
                + "\n".join(f"- {m}" for m in state["memories"])
            )
        response = await llm.ainvoke(
            [SystemMessage(content="\n".join(system_parts))] + list(state["messages"])
        )
        cache.store(prompt=user_msg, response=response.content)
        return {"messages": [AIMessage(content=response.content)]}

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
) -> None:
    if len(messages) < 2:
        return
    conv = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages[-10:]
    )
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(content=conv),
        ])
        data = json.loads(resp.content)
    except Exception:
        return

    ns = _user_ns(user_id)
    ts = str(int(time.time()))
    entries: list[tuple[str, str]] = (
        [(f"fact_{ts}_{i}", t) for i, t in enumerate(data.get("facts", []))]
        + [(f"pref_{ts}_{i}", t) for i, t in enumerate(data.get("preferences", []))]
        + ([(f"summary_{ts}", data["summary"])] if data.get("summary") else [])
    )
    for key, text in entries:
        await store.aput(
            ns,
            f"{session_id}_{key}",
            {"text": text, "session_id": session_id, "ts": ts},
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
        overwrite=False,
    )
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key="not-needed",
        temperature=0.25,
    )

    ttl_cfg = {"default_ttl": args.ttl, "refresh_on_read": True}
    index_cfg = {"embed": HFVectorizerEmbeddings(vectorizer), "dims": EMBEDDING_DIMS, "fields": ["text"]}

    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl_cfg) as checkpointer:
        await checkpointer.asetup()
        async with DragonflyRedisStore.from_conn_string(redis_url, index=index_cfg) as store:
            store.setup()

            compiled = build_graph(llm, cache, user_id).compile(checkpointer=checkpointer, store=store)

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

                    result = await compiled.ainvoke(
                        {"messages": [HumanMessage(content=user_input)], "memories": []},
                        config=config,
                    )
                    reply = result["messages"][-1].content
                    messages += [HumanMessage(content=user_input), AIMessage(content=reply)]
                    turn_count += 1

                    print(f"\nAssistant: {reply}\n")

                    if not args.no_background and turn_count % EXTRACTION_EVERY_N == 0:
                        pending.append(asyncio.create_task(
                            extract_and_store(list(messages), store, llm, session_id, user_id)
                        ))
                        print("[Background memory extraction scheduled]")

            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            print("Extracting session memories...")
            await extract_and_store(messages, store, llm, session_id, user_id)
            print("Memories saved.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangGraph agent with two-tier memory on DragonflyDB")
    p.add_argument("-H", "--host", default="localhost")
    p.add_argument("-p", "--port", type=int, default=7900)
    p.add_argument("-s", "--password", default=None)
    p.add_argument("-u", "--username", default=None)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--session", default=None, help="Resume a prior session by ID")
    p.add_argument("--ttl", type=int, default=86400, help="Session memory TTL in seconds (default 24h)")
    p.add_argument("--user", default="default", help="Stable user ID for memory namespace isolation")
    p.add_argument("--no-background", action="store_true", dest="no_background")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
