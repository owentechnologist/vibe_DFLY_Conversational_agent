"""
Semantic LLM cache — LangChain + redisvl backed by DragonflyDB

Usage:
    python semanticcache.py [-H host] [-p port] [-s password] [-u username] [--threshold float]

LLM: locally-hosted localAI at http://localhost:6060/v1
"""

import argparse
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import FilterExpression
from redisvl.utils.vectorize import HFTextVectorizer

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
                # Build the target schema the same way SemanticCache.__init__ would.
                from redisvl.extensions.cache.llm.schema import SemanticCacheIndexSchema
                vectorizer = kwargs.get("vectorizer")
                if vectorizer is None:
                    vectorizer = HFTextVectorizer()
                target_schema = SemanticCacheIndexSchema.from_params(
                    name, name, vectorizer.dims, vectorizer.dtype
                )
                if _schema_fields_equivalent(
                    existing.schema.to_dict(), target_schema.to_dict()
                ):
                    resolved_overwrite = True
            except Exception:
                pass  # index doesn't exist yet — let super handle it normally
        super().__init__(name=name, overwrite=resolved_overwrite, **kwargs)

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
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

        redis_keys, cache_hits = self._process_cache_results(
            cache_search_results,
            return_fields,  # type: ignore
        )

        for key in redis_keys:
            self.expire(key)

        return cache_hits

LLM_BASE_URL = "http://localhost:6060/v1"
LLM_MODEL = "qwen3.5-9b-glm5.1-distill-v1"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CACHE_INDEX_NAME = "llm_semantic_cache"


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic LLM cache with DragonflyDB")
    parser.add_argument("-H", "--host", default="localhost", help="Dragonfly host")
    parser.add_argument("-p", "--port", type=int, default=6379, help="Dragonfly port")
    parser.add_argument("-s", "--password", default=None, help="Dragonfly password")
    parser.add_argument("-u", "--username", default=None, help="Dragonfly username")
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Cosine distance threshold for cache hits (lower = stricter match)"
    )
    return parser.parse_args()


def build_redis_url(host, port, username=None, password=None):
    if username and password:
        return f"redis://{username}:{password}@{host}:{port}"
    if password:
        return f"redis://:{password}@{host}:{port}"
    return f"redis://{host}:{port}"


def main():
    args = parse_args()
    redis_url = build_redis_url(args.host, args.port, args.username, args.password)

    vectorizer = HFTextVectorizer(model=EMBEDDING_MODEL)
    cache = DragonflySemanticCache(
        name=CACHE_INDEX_NAME,
        vectorizer=vectorizer,
        redis_url=redis_url,
        distance_threshold=args.threshold,
        overwrite=True,
    )

    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key="not-needed",
        temperature=0.25,
    )

    print(f"Connected to Dragonfly at {args.host}:{args.port}")
    print(f"Cache index: {CACHE_INDEX_NAME}  |  Distance threshold: {args.threshold}")
    print("Type 'END' to quit.\n")

    while True:
        user_input = input("Prompt> ").strip()
        if not user_input or user_input.lower() == "end":
            break

        start = time.perf_counter()

        hits = cache.check(prompt=user_input)
        if hits:
            response = hits[0]["response"]
            source = "CACHE HIT"
        else:
            response = llm.invoke(user_input).content
            cache.store(prompt=user_input, response=response)
            source = "LLM"

        elapsed = time.perf_counter() - start
        print(f"\n[{source}] ({elapsed:.3f}s)\n{response}\n")


if __name__ == "__main__":
    main()
