"""
Centralized configuration for the DragonflyDB agent project.

Every setting reads from an environment variable first, then falls back to the
hardcoded default below.  Override at runtime with environment variables or by
editing the defaults here — you should never need to touch app.py, agent.py, or
semanticcache.py for connection or tuning changes.

TLS quick-start:
    DRAGONFLY_TLS=true DRAGONFLY_SSL_CERT_REQS=none python app.py   # self-signed cert
    DRAGONFLY_TLS=true DRAGONFLY_SSL_CA_CERT=/path/to/ca.pem python app.py  # custom CA
"""

import os
import ssl

# ── Dragonfly connection ──────────────────────────────────────────────────────

DRAGONFLY_HOST     = os.getenv("DRAGONFLY_HOST",     "localhost")
DRAGONFLY_PORT     = int(os.getenv("DRAGONFLY_PORT", "7900"))
DRAGONFLY_USERNAME = os.getenv("DRAGONFLY_USERNAME") or None
DRAGONFLY_PASSWORD = os.getenv("DRAGONFLY_PASSWORD") or None

DRAGONFLY_USE_TLS       = os.getenv("DRAGONFLY_TLS", "false").lower() in ("true", "1", "yes")
DRAGONFLY_SSL_CA_CERT   = os.getenv("DRAGONFLY_SSL_CA_CERT")   or None  # None → skip CA verification
DRAGONFLY_SSL_CERT_REQS = os.getenv("DRAGONFLY_SSL_CERT_REQS", "none").lower()  # "none" | "required"

# ── Connection pool ───────────────────────────────────────────────────────────

REDIS_POOL_MAX_CONNECTIONS   = int(os.getenv("REDIS_POOL_MAX_CONNECTIONS",   "50"))
REDIS_SOCKET_TIMEOUT         = float(os.getenv("REDIS_SOCKET_TIMEOUT",       "5.0"))
REDIS_SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))

# ── LLM ──────────────────────────────────────────────────────────────────────

LLM_BASE_URL               = os.getenv("LLM_BASE_URL",  "http://localhost:6060/v1")
# the model is picked up from launch.sh if you set it there, this next line will not override it:
LLM_MODEL                  = os.getenv("LLM_MODEL",     "mlx-community/Qwen3-14B-4bit")
#This model only works with its trained data and rejects most session-based requests: LLM_MODEL                  = os.getenv("LLM_MODEL", "mlx-community/MiniCPM5-1B-OptiQ-4bit")
LLM_TEMPERATURE_CHAT       = float(os.getenv("LLM_TEMPERATURE_CHAT",       "0.25"))
LLM_TEMPERATURE_EXTRACTION = float(os.getenv("LLM_TEMPERATURE_EXTRACTION", "0.05"))
TOKEN_LIMIT                = int(os.getenv("TOKEN_LIMIT", "24576"))

# ── Embedding ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DIMS  = int(os.getenv("EMBEDDING_DIMS", "768"))

# ── Semantic cache ────────────────────────────────────────────────────────────

CACHE_INDEX_NAME           = os.getenv("CACHE_INDEX_NAME",          "llm_semantic_cache")
CACHE_DISTANCE_THRESHOLD   = float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.15"))
RESPONSE_CHUNK_SIZE        = int(os.getenv("RESPONSE_CHUNK_SIZE",   "3000"))
CATCHALL_CACHE_SESSION_TTL = int(os.getenv("CATCHALL_CACHE_SESSION_TTL", "600"))   # 10 min
CATCHALL_CACHE_ALL_TTL     = int(os.getenv("CATCHALL_CACHE_ALL_TTL",     "1800"))  # 30 min

# ── Session / web server ──────────────────────────────────────────────────────

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24 h
WEB_PORT            = int(os.getenv("WEB_PORT", "8080"))

# ── Memory extraction ─────────────────────────────────────────────────────────

EXTRACTION_EVERY_N      = int(os.getenv("EXTRACTION_EVERY_N",      "7"))
EXTRACTION_CHUNK_BUDGET = int(os.getenv("EXTRACTION_CHUNK_BUDGET", "7000"))
PROMPT_HUMAN_CHAR_LIMIT = int(os.getenv("PROMPT_HUMAN_CHAR_LIMIT", "2000"))
PROMPT_AI_TOKEN_LIMIT   = int(os.getenv("PROMPT_AI_TOKEN_LIMIT",   "1500"))
DEDUP_THRESHOLD         = float(os.getenv("DEDUP_THRESHOLD",       "0.85"))

# ── TopK token tracking ───────────────────────────────────────────────────────

TOPK_SESSION_KEY = "topk:tokens:session"
TOPK_USER_KEY    = "topk:tokens:user"
TOPK_K           = int(os.getenv("TOPK_K",     "100"))
TOPK_WIDTH       = int(os.getenv("TOPK_WIDTH", "200"))
TOPK_DEPTH       = int(os.getenv("TOPK_DEPTH", "7"))
TOPK_DECAY       = float(os.getenv("TOPK_DECAY", "0.9"))
TOPK_DAY_TTL     = int(os.getenv("TOPK_DAY_TTL",   str(86400 * 30)))
TOPK_MONTH_TTL   = int(os.getenv("TOPK_MONTH_TTL", str(86400 * 395)))
TOPK_YEAR_TTL    = int(os.getenv("TOPK_YEAR_TTL",  str(86400 * 1100)))


# ── Connection helpers ────────────────────────────────────────────────────────

def build_redis_url(
    host: str = DRAGONFLY_HOST,
    port: int = DRAGONFLY_PORT,
    username: str | None = None,
    password: str | None = None,
    use_tls: bool = DRAGONFLY_USE_TLS,
) -> str:
    """
    Builds a Redis connection URL adapting dynamically to credentials.
    Delivers a secure URL if credentials are provided, otherwise an unauthenticated one.
    """
    scheme = "rediss" if use_tls else "redis"
    
    # Scenario 1: Both username and password are provided
    if username and password:
        return f"{scheme}://{username}:{password}@{host}:{port}"
        
    # Scenario 2: Only a password is provided (Standard Redis default auth)
    if password:
        return f"{scheme}://:{password}@{host}:{port}"
        
    # Scenario 3: Unauthenticated / Only host and port are used
    return f"{scheme}://{host}:{port}"


'''def build_redis_url(
    host: str = DRAGONFLY_HOST,
    port: int = DRAGONFLY_PORT,
    username: str | None = DRAGONFLY_USERNAME,
    password: str | None = DRAGONFLY_PASSWORD,
    use_tls: bool = DRAGONFLY_USE_TLS,
) -> str:
    """Build a Redis connection URL; uses rediss:// scheme when use_tls=True."""
    scheme = "rediss" if use_tls else "redis"
    if username and password:
        return f"{scheme}://{username}:{password}@{host}:{port}"
    if password:
        return f"{scheme}://:{password}@{host}:{port}"
    return f"{scheme}://{host}:{port}"

'''

def build_ssl_context(
    ca_cert: str | None = DRAGONFLY_SSL_CA_CERT,
    cert_reqs: str = DRAGONFLY_SSL_CERT_REQS,
) -> ssl.SSLContext | None:
    """Return an ssl.SSLContext for TLS connections, or None if TLS is disabled.

    Pass the result as ssl_context= to aioredis or other Redis clients.
    cert_reqs='none'     → accept self-signed certs (no CA verification)
    cert_reqs='required' → verify against system CAs or the provided ca_cert file
    """
    if not DRAGONFLY_USE_TLS:
        return None
    ctx = ssl.create_default_context()
    if cert_reqs == "none":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    elif ca_cert:
        ctx.load_verify_locations(ca_cert)
    return ctx
