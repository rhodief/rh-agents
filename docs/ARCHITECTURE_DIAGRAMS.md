# Cache System Architecture Diagrams

## Execution Flow with Caching

```
┌─────────────────────────────────────────────────────────────────┐
│                    ExecutionEvent.__call__()                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Push Context     │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Actor Cacheable? │
                    └──────────────────┘
                          │         │
                     NO ──┘         └── YES
                     │                  │
                     │                  ▼
                     │       ┌────────────────────┐
                     │       │ Compute Cache Key  │
                     │       │ (address + input + │
                     │       │  name + version)   │
                     │       └────────────────────┘
                     │                  │
                     │                  ▼
                     │       ┌────────────────────┐
                     │       │ Check Cache        │
                     │       └────────────────────┘
                     │              │        │
                     │         HIT ─┘        └─ MISS
                     │         │                 │
                     │         ▼                 │
                     │   ┌──────────────┐        │
                     │   │ Return       │        │
                     │   │ Cached       │        │
                     │   │ Result       │        │
                     │   │              │        │
                     │   │ Status:      │        │
                     │   │ RECOVERED    │        │
                     │   │              │        │
                     │   │ from_cache:  │        │
                     │   │ True         │        │
                     │   └──────────────┘        │
                     │         │                 │
                     │         └───────┐         │
                     │                 │         │
                     └─────────────────┤         │
                                       │         │
                                       │         ▼
                                       │   ┌──────────────────┐
                                       │   │ Run Pre-         │
                                       │   │ conditions       │
                                       │   └──────────────────┘
                                       │         │
                                       │         ▼
                                       │   ┌──────────────────┐
                                       │   │ Add Event        │
                                       │   │ (STARTED)        │
                                       │   └──────────────────┘
                                       │         │
                                       │         ▼
                                       │   ┌──────────────────┐
                                       │   │ Execute Handler  │
                                       │   └──────────────────┘
                                       │         │
                                       │         ▼
                                       │   ┌──────────────────┐
                                       │   │ Run Post-        │
                                       │   │ conditions       │
                                       │   └──────────────────┘
                                       │         │
                                       │         ▼
                                       │   ┌──────────────────┐
                                       │   │ Add Event        │
                                       │   │ (COMPLETED)      │
                                       │   └──────────────────┘
                                       │         │
                                       │         ▼
                                       │   ┌──────────────────┐
                                       │   │ Store in Cache   │
                                       │   │ (if cacheable)   │
                                       │   └──────────────────┘
                                       │         │
                                       └─────────┤
                                                 │
                                                 ▼
                                       ┌──────────────────┐
                                       │ Pop Context      │
                                       └──────────────────┘
                                                 │
                                                 ▼
                                       ┌──────────────────┐
                                       │ Return Result    │
                                       └──────────────────┘
```

## Cache Key Composition

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cache Key Generation                       │
└─────────────────────────────────────────────────────────────────┘

Input Components:
┌──────────────────────┐
│ Address              │  "OmniAgent::DoctrineReceiver::OpenAI-LLM::llm_call"
└──────────────────────┘
          │
          ├─────────────┐
          │             │
┌──────────────────────┐│
│ Actor Name           ││  "OpenAI-LLM"
└──────────────────────┘│
          │             │
          ├─────────────┤
          │             │
┌──────────────────────┐│
│ Actor Version        ││  "1.0.0"
└──────────────────────┘│
          │             │
          ├─────────────┤
          │             │
┌──────────────────────┐│
│ Input Data           ││  {"prompt": "Analyze document", "model": "gpt-4"}
│                      ││           │
│  1. Serialize to JSON││           ▼
│  2. Sort keys        ││  '{"model": "gpt-4", "prompt": "Analyze document"}'
│  3. SHA256 hash      ││           │
│                      ││           ▼
└──────────────────────┘│  "a1b2c3d4e5f6g7h8" (first 16 chars)
          │             │
          └─────────────┘
                │
                ▼
    ┌──────────────────────┐
    │ Concatenate          │
    │                      │
    │ "OmniAgent::Doc...   │
    │  ::OpenAI-LLM::      │
    │  1.0.0::             │
    │  a1b2c3d4e5f6g7h8"   │
    └──────────────────────┘
                │
                ▼
    ┌──────────────────────┐
    │ SHA256 Hash          │
    └──────────────────────┘
                │
                ▼
    ┌──────────────────────┐
    │ Cache Key            │
    │                      │
    │ "f9e8d7c6b5a4..."    │
    └──────────────────────┘
```

## Cache Backend Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       CacheBackend (Abstract)                    │
├──────────────────────────────────────────────────────────────────┤
│  + get(key) → CachedResult | None                               │
│  + set(key, result, ttl)                                         │
│  + invalidate(key) → bool                                        │
│  + invalidate_pattern(pattern) → int                             │
│  + clear()                                                       │
│  + get_stats() → dict                                            │
└──────────────────────────────────────────────────────────────────┘
                          ▲              ▲
                          │              │
           ┌──────────────┘              └──────────────┐
           │                                            │
┌──────────────────────────┐              ┌─────────────────────────┐
│ InMemoryCacheBackend     │              │ FileCacheBackend        │
├──────────────────────────┤              ├─────────────────────────┤
│  Storage: dict           │              │  Storage: filesystem    │
│  Fast: < 1ms             │              │  Persistent: Yes        │
│  Persistent: No          │              │  Speed: ~5-10ms         │
│  Use: Development        │              │  Use: Production        │
└──────────────────────────┘              └─────────────────────────┘

                    Future Extensions:
                    
┌──────────────────────────┐              ┌─────────────────────────┐
│ RedisCacheBackend        │              │ S3CacheBackend          │
├──────────────────────────┤              ├─────────────────────────┤
│  Storage: Redis          │              │  Storage: AWS S3        │
│  Distributed: Yes        │              │  Scalable: Very         │
│  Fast: ~2-5ms            │              │  Speed: ~50-100ms       │
│  Use: Multi-process      │              │  Use: Large datasets    │
└──────────────────────────┘              └─────────────────────────┘
```

## Actor Caching Configuration

```
┌──────────────────────────────────────────────────────────────────┐
│                            BaseActor                             │
├──────────────────────────────────────────────────────────────────┤
│  + name: str                                                     │
│  + description: str                                              │
│  + cacheable: bool = False        ← Default: No caching         │
│  + version: str = "1.0.0"         ← For invalidation            │
│  + cache_ttl: int | None = None   ← TTL in seconds              │
└──────────────────────────────────────────────────────────────────┘
                          ▲
                          │
         ┌────────────────┼────────────────┐
         │                │                │
┌─────────────────┐ ┌─────────────┐ ┌────────────────┐
│ LLM             │ │ Tool        │ │ Agent          │
├─────────────────┤ ├─────────────┤ ├────────────────┤
│ cacheable: TRUE │ │ cacheable:  │ │ cacheable:     │
│ cache_ttl: 3600 │ │ FALSE       │ │ FALSE          │
│                 │ │             │ │                │
│ ✅ Cache by     │ │ ❌ Don't    │ │ ❌ Don't       │
│    default      │ │    cache    │ │    cache       │
│                 │ │    (side    │ │    (orchestr-  │
│ Why: Expensive  │ │    effects) │ │    ation)      │
│      API calls  │ │             │ │                │
└─────────────────┘ └─────────────┘ └────────────────┘
```

## Event Flow with Cache Visualization

```
Time: ─────────────────────────────────────────────────────────────>

Execution 1 (Cache Miss):
┌────┐    ┌────┐    ┌─────┐    ┌─────┐    ┌─────┐
│User│───▶│Evt │───▶│Cache│───▶│LLM  │───▶│Cache│
└────┘    │    │    │ X   │    │ API │    │✓Str │
          │    │    │Miss │    │Call │    │     │
          └────┘    └─────┘    └─────┘    └─────┘
          STARTED               2500ms     COMPLETED
          
          Status: STARTED → COMPLETED
          from_cache: False
          Time: 2500ms


Execution 2 (Cache Hit - Same Input):
┌────┐    ┌────┐    ┌─────┐    
│User│───▶│Evt │───▶│Cache│───▶ DONE!
└────┘    │    │    │ ✓   │    
          │    │    │ Hit │    
          └────┘    └─────┘    
          RECOVERED             < 1ms
          
          Status: RECOVERED (not STARTED/COMPLETED)
          from_cache: True
          Time: 0ms
          
          
Execution 3 (Cache Hit - Same Input Again):
┌────┐    ┌────┐    ┌─────┐    
│User│───▶│Evt │───▶│Cache│───▶ DONE!
└────┘    │    │    │ ✓   │    
          │    │    │ Hit │    
          └────┘    └─────┘    
          RECOVERED             < 1ms


Total time saved: 2500ms + 2500ms = 5000ms
Cache hit rate: 66.7% (2/3)
```

## Statistics Tracking Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         EventPrinter                            │
│                                                                 │
│  Tracks:                                                        │
│    • total_events                                               │
│    • started_events                                             │
│    • completed_events                                           │
│    • recovered_events        ← New!                             │
│    • failed_events                                              │
│    • cache_hits              ← New!                             │
│    • cache_misses            ← New!                             │
│    • total_execution_time                                       │
│    • events_by_type                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    On Event Received:
                              │
                ┌─────────────┴─────────────┐
                │                           │
         ExecutionStatus?                   │
                │                           │
    ┌───────────┼───────────┐               │
    │           │           │               │
STARTED      COMPLETED   RECOVERED          │
    │           │           │               │
    ▼           ▼           ▼               ▼
cache_miss++  Track Time  cache_hit++   Update Stats
started++     completed++ recovered++
                              │
                              ▼
                    ┌──────────────────┐
                    │ print_summary()  │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────────────────┐
                    │ Display:                     │
                    │  • Total events              │
                    │  • Success rate              │
                    │  • Cache hit rate      ← New!│
                    │  • Total execution time      │
                    │  • Events by type            │
                    └──────────────────────────────┘
```
