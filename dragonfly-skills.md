markdown
# DragonflyDB Development Skills

> Give this file to Claude (or any LLM) as context when building applications with DragonflyDB.
> It encodes architecture knowledge, hard-won operational lessons, and optimal patterns
> so the AI doesn't fall into common Redis-assumption traps.

## 1. What Is DragonflyDB

DragonflyDB is a modern, multi-threaded, in-memory data store that is API-compatible with Redis and Memcached. It is NOT Redis. Do not reason about Dragonfly using Redis internals — they share an API surface, not an implementation.

**Key architectural differences from Redis:**
- **Multi-threaded, shared-nothing architecture.** Each thread manages its own shard of the keyspace. There is NO single-threaded event loop. Never reference an "event loop" when discussing Dragonfly.
- **Dashtable** — a novel hash table design with ~30% better memory efficiency than Redis's dict.
- **Uses `io_uring`** for async I/O — iowait metrics are meaningless for Dragonfly workloads.
- **Fibers, not callbacks** — each thread uses fibers for non-blocking, async operations.
- **Built-in multi-key atomicity** without the performance penalty of Redis's single-threaded model.

## 2. Connection & Client Setup

DragonflyDB works with any standard Redis client library (redis-py, ioredis, Jedis, go-redis, etc.). No special client needed.
python
# Python example
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.set('key', 'value')
```

```typescript
// Node.js (ioredis)
import Redis from 'ioredis';
const redis = new Redis({ host: 'localhost', port: 6379 });
await redis.set('key', 'value');
### Connection Best Practices
- **Use connection pooling.** Dragonfly handles many concurrent connections efficiently, but pooling reduces connection overhead.
- **Prefer pipelining for bulk operations.** Dragonfly's multi-threaded engine processes pipelined commands in parallel across threads — you get much better throughput than sequential calls.
- **TLS is supported.** Use `--tls` flags or configure via `tls-cert-file` / `tls-key-file`.

## 3. Data Patterns & Best Practices

### Cache-Aside (Lazy Loading)
The most common pattern. Check cache first, fall back to DB on miss, populate cache on read.
python
def get_user(user_id: str) -> dict:
    cached = r.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    user = db.query_user(user_id)
    r.setex(f"user:{user_id}", 3600, json.dumps(user))  # 1h TTL
    return user
### Batch Operations — Use Pipelines
Dragonfly processes pipelined commands across multiple threads in parallel. This is where you get the biggest performance wins vs Redis.
python
pipe = r.pipeline()
for key in keys:
    pipe.get(key)
results = pipe.execute()
For writes:python
pipe = r.pipeline()
for item in items:
    pipe.setex(f"item:{item['id']}", 3600, json.dumps(item))
pipe.execute()
### MULTI/EXEC Transactions
Dragonfly supports `MULTI`/`EXEC` with configurable atomicity:
- `--multi_exec_mode=1` — global atomicity (strongest, most overhead)
- `--multi_exec_mode=2` — locking ahead (default, good balance)
- `--multi_exec_mode=3` — non-atomic (fastest, no guarantees)
python
pipe = r.pipeline(transaction=True)  # MULTI/EXEC
pipe.decrby("inventory:item1", 1)
pipe.rpush("orders:user1", order_json)
pipe.execute()  # atomic
**Tip:** Minimize commands inside transactions. Dragonfly's multi-threaded design means long transactions block the relevant shards.

### Lua Scripting
Dragonfly supports Lua scripting with one important difference:

**⚠️ Dragonfly enforces strict key declaration.** All keys accessed by a Lua script MUST be declared in the `KEYS` array. Scripts that access undeclared keys will fail on first invocation — but Dragonfly auto-mitigates by setting an `undeclared_keys` flag for that script SHA, so subsequent calls succeed.
python
# Correct — all keys declared
script = """
local current = redis.call('GET', KEYS[1])
if current then
    redis.call('SET', KEYS[2], current)
end
return current
"""
r.eval(script, 2, 'source_key', 'dest_key')
```

```python
# Wrong — accessing a key not in KEYS array
# This will fail on first call (but auto-heal on retry)
script = """
local key = 'dynamic:' .. ARGV[1]
return redis.call('GET', key)  -- undeclared key!
"""
**If using BullMQ, Sidekiq, or similar queue frameworks:** Their Lua scripts may access dynamic keys. Dragonfly handles this gracefully after the first failure per script SHA — but be aware of initial error bursts on first deployment.

### Pub/Sub
Works identically to Redis. Dragonfly's multi-threaded design handles high fan-out more efficiently.
python
# Publisher
r.publish('events', json.dumps({'type': 'order', 'id': '123'}))

# Subscriber
pubsub = r.pubsub()
pubsub.subscribe('events')
for message in pubsub.listen():
    if message['type'] == 'message':
        handle_event(json.loads(message['data']))
### Streams
Redis Streams API is fully supported. Use for event sourcing, task queues, and log aggregation.
python
# Producer
r.xadd('mystream', {'sensor': 'temp', 'value': '22.5'})

# Consumer group
r.xgroup_create('mystream', 'mygroup', id='0', mkstream=True)
messages = r.xreadgroup('mygroup', 'consumer1', {'mystream': '>'}, count=10)
## 4. Memory Management

### Key Concepts
- `maxmemory` — the memory limit. When reached, Dragonfly applies the eviction policy.
- **RSS vs used_memory:** Dragonfly's actual RSS can be significantly higher than `used_memory` due to allocator overhead, replication buffers, and snapshot serialization. Size your nodes for RSS, not used_memory.
- **Eviction policies:** Same as Redis (`allkeys-lru`, `volatile-lru`, `allkeys-random`, `noeviction`, etc.)

### TTL Best Practices
- **Always set TTLs on cache data.** Use `SETEX`/`SET EX` or `EXPIRE`.
- **Understand the TTL equilibrium:** With high-churn workloads (lots of writes + TTLs), steady-state key count is a balance between incoming writes and TTL expiry. Operational events (replication, failover) can temporarily shift this equilibrium.
- **Use `PTTL`/`TTL` to inspect expiry** — useful for debugging unexpected evictions.
python
# Set with TTL
r.setex('session:abc', 1800, session_data)  # 30 min

# Set TTL on existing key
r.expire('session:abc', 1800)
### Avoid Memory Pitfalls
- **Don't store huge values (>1MB) without tiered storage.** Large values block the thread during serialization.
- **Use hash encoding for objects** instead of serialized JSON strings when you need partial reads:
 python
  # Better — allows partial read/write
  r.hset('user:123', mapping={'name': 'Owen', 'role': 'eng', 'score': '42'})
  name = r.hget('user:123', 'name')

  # Worse for partial access — must deserialize entire blob
  r.set('user:123', json.dumps({'name': 'Owen', 'role': 'eng', 'score': 42}))
 
## 5. Tiered Storage (SSD Data Tiering)

Dragonfly can offload values to NVMe/SSD, reducing RAM usage by 2-5x while maintaining sub-millisecond latency.

**How it works:**
- String values > 64 bytes are candidates for offloading to SSD
- Keys, metadata, and small values stay in RAM
- Reads transparently fetch from SSD when needed
- Writes, deletes, and expires are managed in-memory

**Enable with:**bash
dragonfly --tiered_prefix /mnt/nvme/dragonfly --tiered_max_file_size 100G
**Requirements:** Linux kernel ≥ 5.19, `io_uring` enabled, fast SSD/NVMe.

**When to use:** Large datasets that exceed available RAM but have a working set that fits in memory. Ideal for caching layers with long-tail access patterns.

## 6. Cluster Mode

### Emulated Cluster Mode (Single Node)
A single Dragonfly instance emulates a Redis Cluster — useful for migrating apps that use Redis Cluster clients without code changes.
bash
dragonfly --cluster_mode=emulated
- Supports `CLUSTER SLOTS`, `CLUSTER NODES`, etc.
- No actual horizontal scaling — one node handles everything
- A single Dragonfly node can often replace a multi-node Redis Cluster

### Multi-Node Cluster Mode
For datasets that truly need horizontal scaling beyond one machine.

- 16,384 slots, hash-tag compatible
- Nodes don't self-manage — configuration is external (DragonflyDB Cloud handles this automatically)
- Use hash tags `{tag}` to co-locate related keys on the same shard:
 python
  r.set('{user:123}:profile', profile_data)
  r.set('{user:123}:sessions', session_data)
  # Both land on the same shard — safe for MULTI/EXEC
 
## 7. Replication & High Availability

Dragonfly supports master-replica replication compatible with Redis replication protocol.

**Key differences from Redis replication:**
- Full sync is multi-threaded and faster
- Dragonfly does NOT replicate ACLs — configure ACLs on each node independently
- During full sync, the master's write throughput may temporarily drop (serialization overhead)

**HA pattern:**
- 1 master + 1 replica per shard
- Automated failover via control plane (DragonflyDB Cloud) or manual `REPLICAOF NO ONE`
- Clients should use Sentinel-aware or Cluster-aware clients for automatic failover handling

## 8. Observability & Monitoring

### Prometheus Metrics
Dragonfly exposes metrics at `/metrics` with the `dragonfly_` prefix.

**Key metrics to monitor:**

| Metric | What it tells you |
|---|---|
| `dragonfly_memory_used_bytes` | Current memory usage |
| `dragonfly_memory_max_bytes` | Maxmemory limit |
| `dragonfly_used_memory_rss_bytes` | Actual RSS (what the OS sees) |
| `dragonfly_connected_clients` | Client connections |
| `dragonfly_commands_total` | Command throughput (by `cmd` label) |
| `dragonfly_commands_duration_seconds` | Command latency (by `cmd` label) |
| `dragonfly_db_keys` | Key count per DB |
| `dragonfly_expired_keys_total` | TTL expiry rate |
| `dragonfly_uptime_in_seconds` | Process uptime (drops = restarts) |

**⚠️ Metric naming gotcha:** It's `dragonfly_memory_used_bytes`, NOT `dragonfly_used_memory_bytes`. The naming is not always intuitive — check the `/metrics` endpoint directly when in doubt.

### INFO Command
Standard Redis `INFO` command works. Useful for quick debugging:bash
redis-cli INFO memory
redis-cli INFO stats
redis-cli INFO replication
### OpenTelemetry
Dragonfly has built-in OpenTelemetry support for tracing.

## 9. Configuration Flags

Key flags to know when deploying:
bash
# Memory
--maxmemory=8G                    # Memory limit
--eviction_policy=allkeys-lru     # What to evict when full

# Performance
--proactor_threads=0              # 0 = auto-detect (num CPUs)
--pipeline_squash=10              # Squash pipeline commands (default)
--multi_exec_squash=true          # Optimize multi/exec (default)

# Persistence
--dbfilename=dump.rdb             # Snapshot filename
--save_schedule=":"             # Cron-style save schedule

# Cluster
--cluster_mode=emulated           # or 'yes' for real cluster

# Tiered storage
--tiered_prefix=/mnt/nvme/df
--tiered_max_file_size=100G

# Network
--bind=0.0.0.0
--port=6379
--requirepass=yourpassword
## 10. Migration from Redis

### What works out of the box:
- All standard Redis commands (strings, hashes, lists, sets, sorted sets, streams, pub/sub, Lua scripting)
- RDB file import — `dragonfly --dbfilename /path/to/dump.rdb`
- Redis Cluster clients (with `--cluster_mode=emulated`)
- Sentinel-compatible clients

### What to watch for:
- **Lua scripts accessing undeclared keys** — will fail once then auto-heal (see Section 3)
- **ACLs are NOT replicated** — configure on each node separately
- **Module commands** — Dragonfly has native implementations of some Redis modules (Search, JSON, Bloom) but not all. Check compatibility.
- **`io_uring` dependency** — Dragonfly requires modern Linux. Not available on macOS for production (use Docker).

### Migration approaches:
1. **RDB import** — stop Redis, copy `dump.rdb`, start Dragonfly with it
2. **Live replication** — point Dragonfly as a replica of Redis (`REPLICAOF redis-host 6379`), let it sync, then cut over
3. **Dual-write** — write to both during transition (application-level)

## 11. Common Anti-Patterns to Avoid

❌ **Don't use `KEYS *` in production.** Use `SCAN` instead — it's non-blocking and paginated.

❌ **Don't assume single-threaded behavior.** Dragonfly processes commands concurrently across threads. Race conditions that are impossible in Redis (due to its event loop) may need explicit handling with `WATCH`/`MULTI`/`EXEC` or Lua scripts in Dragonfly.

❌ **Don't ignore RSS vs used_memory.** Size your infrastructure for RSS, which can be 1.5-3x `used_memory` depending on workload and replication state.

❌ **Don't run backups during full sync.** If a node is replicating (initial full sync), triggering a backup snapshot simultaneously can cause OOM — both operations consume significant memory.

❌ **Don't store everything as serialized JSON strings.** Use native Redis data structures (hashes, sorted sets, lists) — they're more memory-efficient and allow partial operations.

❌ **Don't set maxmemory equal to physical RAM.** Leave headroom for RSS overhead, OS buffers, replication, and snapshots. A good rule: `maxmemory` ≤ 70-80% of available RAM.

## 12. Dragonfly-Specific Features

### Native JSON Supportpython
from redis.commands.json.path import Path
r.json().set('doc:1', Path.root_path(), {'name': 'Owen', 'scores': [1, 2, 3]})
r.json().get('doc:1', Path('.name'))
### Native Search
Full-text search built in (no separate module needed):bash
FT.CREATE idx ON HASH PREFIX 1 doc: SCHEMA title TEXT SORTABLE body TEXT
FT.SEARCH idx "dragonfly performance"
### Bloom Filtersbash
BF.ADD myfilter "element1"
BF.EXISTS myfilter "element1"  # returns 1
BF.EXISTS myfilter "element2"  # returns 0 (probably)
## 13. Quick Reference Links

- **Docs:** https://www.dragonflydb.io/docs
- **GitHub:** https://github.com/dragonflydb/dragonfly
- **Command Reference:** https://www.dragonflydb.io/docs/command-reference
- **Configuration Flags:** https://www.dragonflydb.io/docs/managing-dragonfly/flags
- **Cluster Mode:** https://www.dragonflydb.io/docs/managing-dragonfly/cluster-mode
- **Tiered Storage:** https://www.dragonflydb.io/docs/managing-dragonfly/flags (look for `tiered_` flags)
- **Architecture Deep Dive:** https://github.com/dragonflydb/dragonfly/blob/main/docs/df-share-nothing.md
---

