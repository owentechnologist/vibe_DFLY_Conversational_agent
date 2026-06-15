"""
Semantic LLM cache — LangChain + redisvl backed by DragonflyDB

Usage:
    python semanticcache.py [-H host] [-p port] [-s password] [-u username] [--threshold float]

LLM: locally-hosted localAI at http://localhost:6060/v1
"""

import argparse
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from langchain_openai import ChatOpenAI
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import FilterExpression, Tag
from redisvl.utils.vectorize import HFTextVectorizer

from settings import (
    CACHE_DISTANCE_THRESHOLD,
    CACHE_INDEX_NAME,
    CATCHALL_CACHE_ALL_TTL,
    CATCHALL_CACHE_SESSION_TTL,
    DRAGONFLY_HOST,
    DRAGONFLY_PASSWORD,
    DRAGONFLY_PORT,
    DRAGONFLY_USERNAME,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE_CHAT,
    RESPONSE_CHUNK_SIZE,
    build_redis_url,
)

CACHE_VECTOR_FIELD_NAME = "prompt_vector"


def _schema_fields_equivalent(a: Dict, b: Dict) -> bool:
    """Order-insensitive comparison of two redisvl schema dicts."""
    if a.get("index") != b.get("index"):
        return False
    a_fields = {f["name"]: f for f in a.get("fields", [])}
    b_fields = {f["name"]: f for f in b.get("fields", [])}
    return a_fields == b_fields


class DragonflySemanticCache(SemanticCache):
    """SemanticCache that uses KNN queries instead of VECTOR_RANGE.

    Dragonfly doesn't populate the $YIELD_DISTANCE_AS attribute used by
    VectorRangeQuery, so vector_distance is absent from results. VectorQuery
    (KNN) uses an explicit AS alias that Dragonfly does honour.

    Also works around Dragonfly's FT.INFO returning fields in a different
    order than redisvl defined them, which causes a false schema-mismatch
    error on startup.
    """

    def __init__(self, name: str = "llmcache", overwrite: bool = False, **kwargs):
        # redisvl does a strict (order-sensitive) list comparison of schema dicts.
        # Dragonfly's FT.INFO returns fields in a different order, triggering a
        # false mismatch.  We peek at the existing index ourselves using an
        # order-insensitive comparison; if schemas are functionally equivalent we
        # pass overwrite=True so redisvl skips its check and simply re-issues
        # FT.CREATE (safe: drop=False keeps existing data).
        resolved_overwrite = overwrite
        if not overwrite:
            try:
                redis_url = kwargs.get("redis_url", "redis://localhost:6379")
                existing = SearchIndex.from_existing(name, redis_url=redis_url)
                # Build the target schema the same way SemanticCache.__init__ would,
                # including any filterable_fields (e.g. user_id).
                from redisvl.extensions.cache.llm.schema import SemanticCacheIndexSchema
                vectorizer = kwargs.get("vectorizer")
                if vectorizer is None:
                    vectorizer = HFTextVectorizer()
                target_schema = SemanticCacheIndexSchema.from_params(
                    name, name, vectorizer.dims, vectorizer.dtype
                )
                for ff in kwargs.get("filterable_fields") or []:
                    target_schema.add_field(ff)
                if _schema_fields_equivalent(
                    existing.schema.to_dict(), target_schema.to_dict()
                ):
                    resolved_overwrite = True
            except Exception:
                pass  # index doesn't exist yet — let super handle it normally
        super().__init__(name=name, overwrite=resolved_overwrite, **kwargs)

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        filters: Optional[Dict] = None,
        ttl: Optional[int] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        if not user_id:
            raise ValueError("user_id is required — cache entries are per-user private")
        filters = {**(filters or {}), "user_id": user_id}
        if session_id is not None:
            filters["session_id"] = session_id

        store_response = response
        if len(response) > RESPONSE_CHUNK_SIZE:
            response_id = uuid.uuid4().hex
            chunks = [response[i:i + RESPONSE_CHUNK_SIZE] for i in range(0, len(response), RESPONSE_CHUNK_SIZE)]
            pipe = self._index.client.pipeline(transaction=False)
            for n, chunk in enumerate(chunks):
                chunk_key = f"{self.name}:resp:{response_id}:{n}"
                payload = json.dumps({"i": n, "t": len(chunks), "d": chunk})
                if ttl:
                    pipe.setex(chunk_key, ttl, payload)
                else:
                    pipe.set(chunk_key, payload)
            pipe.execute()
            store_response = f"CHUNKED:{response_id}:{len(chunks)}"

        return super().store(
            prompt=prompt,
            response=store_response,
            vector=vector,
            metadata=metadata,
            filters=filters,
            ttl=ttl,
        )

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not user_id:
            raise ValueError("user_id is required — cache lookups must be per-user isolated")
        user_filter = Tag("user_id") == user_id
        filter_expression = (
            user_filter & filter_expression
            if filter_expression is not None
            else user_filter
        )
        if session_id is not None:
            sess_filter = Tag("session_id") == session_id
            filter_expression = sess_filter & filter_expression

        if not any([prompt, vector]):
            raise ValueError("Either prompt or vector must be specified.")
        if return_fields and not isinstance(return_fields, list):
            raise TypeError("Return fields must be a list of values.")

        distance_threshold = distance_threshold or self._distance_threshold

        if vector is None and prompt is not None:
            vector = self._vectorize_prompt(prompt)

        if vector is not None:
            self._check_vector_dims(vector)
        else:
            raise ValueError("Failed to generate a valid vector for the query.")

        query = VectorQuery(
            vector=vector,
            vector_field_name=CACHE_VECTOR_FIELD_NAME,
            return_fields=self.return_fields,
            num_results=num_results,
            return_score=True,
            filter_expression=filter_expression,
            dtype=self._vectorizer.dtype,
        )

        cache_search_results = self._index.query(query)

        # Post-filter by distance threshold — VectorQuery is KNN (top-K),
        # not range-bounded, so we drop results that exceed the threshold.
        cache_search_results = [
            r for r in cache_search_results
            if float(r.get("vector_distance", 1.0)) <= distance_threshold
        ]

        # Drop incomplete entries missing fields required by CacheHit.
        # Dragonfly may return partial hashes for stale or malformed entries.
        _required = {"response", "entry_id", "inserted_at", "updated_at"}
        cache_search_results = [r for r in cache_search_results if _required.issubset(r)]

        redis_keys, cache_hits = self._process_cache_results(
            cache_search_results,
            return_fields,  # type: ignore
        )

        for key in redis_keys:
            self.expire(key)

        # Reassemble any chunked responses
        for hit in cache_hits:
            resp = hit.get("response", "")
            if isinstance(resp, str) and resp.startswith("CHUNKED:"):
                try:
                    _, resp_id, total_str = resp.split(":", 2)
                    total = int(total_str)
                    pipe = self._index.client.pipeline(transaction=False)
                    for n in range(total):
                        pipe.get(f"{self.name}:resp:{resp_id}:{n}")
                    parts = pipe.execute()
                    hit["response"] = "".join(json.loads(p)["d"] for p in parts if p)
                except Exception:
                    pass  # leave CHUNKED: marker in place if reassembly fails

        return cache_hits

    def refresh_catchall_cache(
        self,
        session_id: str,
        user_id: str,
        new_response: str,
        patterns: tuple,
        ttl: Optional[int] = CATCHALL_CACHE_SESSION_TTL,
    ) -> None:
        """Delete stale session catchall entries and store a fresh synthesized response.

        Stores the summary under each pattern's explicit pre-computed vector, and also
        stores one additional entry using the normalized centroid of all pattern vectors.
        The centroid entry sits between all patterns in embedding space, so semantically
        similar queries that don't fall within the threshold of any single pattern still
        find the updated summary.

        Deletion uses the centroid vector rather than per-pattern checks, so any variant
        of a catchall query ("what do you know about me?", with punctuation, etc.) is
        evicted in one pass based on vector proximity to the cluster centre.
        """
        pattern_list = list(patterns)
        vecs = self._vectorizer.embed_many(pattern_list)

        mat = np.array(vecs, dtype=np.float32)
        centroid = mat.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid_vec = (centroid / norm).tolist() if norm > 0 else centroid.tolist()

        seen: set = set()
        hits = self.check(
            vector=centroid_vec,
            user_id=user_id,
            session_id=session_id,
            num_results=20,
            distance_threshold=0.5,
        )
        for hit in hits:
            eid = hit.get("entry_id")
            if eid and eid not in seen:
                seen.add(eid)
                self._index.client.delete(f"{self.name}:{eid}")

        for pattern, vec in zip(pattern_list, vecs):
            self.store(
                prompt=pattern,
                response=new_response,
                vector=vec.tolist() if hasattr(vec, "tolist") else list(vec),
                user_id=user_id,
                session_id=session_id,
                ttl=ttl,
            )

        # Centroid entry for broad semantic coverage (already computed above)
        self.store(
            prompt=pattern_list[0],
            response=new_response,
            vector=centroid_vec,
            user_id=user_id,
            session_id=session_id,
            ttl=ttl,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Semantic LLM cache with DragonflyDB")
    parser.add_argument("-H", "--host",     default=DRAGONFLY_HOST,     help="Dragonfly host")
    parser.add_argument("-p", "--port",     type=int, default=DRAGONFLY_PORT, help="Dragonfly port")
    parser.add_argument("-s", "--password", default=DRAGONFLY_PASSWORD, help="Dragonfly password")
    parser.add_argument("-u", "--username", default=DRAGONFLY_USERNAME, help="Dragonfly username")
    parser.add_argument(
        "--threshold", type=float, default=CACHE_DISTANCE_THRESHOLD,
        help="Cosine distance threshold for cache hits (lower = stricter match)"
    )
    parser.add_argument("--user", default=None, help="User ID for cache isolation (required)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.user:
        print("Error: --user is required for cache isolation (e.g. --user alice)")
        return

    redis_url = build_redis_url(args.host, args.port, args.username, args.password)

    vectorizer = HFTextVectorizer(model=EMBEDDING_MODEL)
    cache = DragonflySemanticCache(
        name=CACHE_INDEX_NAME,
        vectorizer=vectorizer,
        redis_url=redis_url,
        distance_threshold=args.threshold,
        overwrite=True,
        filterable_fields=[{"name": "user_id", "type": "tag"}],
    )

    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key="not-needed",
        temperature=LLM_TEMPERATURE_CHAT,
    )

    print(f"Connected to Dragonfly at {args.host}:{args.port}")
    print(f"Cache index: {CACHE_INDEX_NAME}  |  Distance threshold: {args.threshold}  |  User: {args.user}")
    print("Type 'END' to quit.\n")

    while True:
        user_input = input("Prompt> ").strip()
        if not user_input or user_input.lower() == "end":
            break

        start = time.perf_counter()

        hits = cache.check(prompt=user_input, user_id=args.user)
        if hits:
            response = hits[0]["response"]
            source = "CACHE HIT"
        else:
            response = llm.invoke(user_input).content
            cache.store(prompt=user_input, response=response, user_id=args.user)
            source = "LLM"

        elapsed = time.perf_counter() - start
        print(f"\n[{source}] ({elapsed:.3f}s)\n{response}\n")


if __name__ == "__main__":
    main()
