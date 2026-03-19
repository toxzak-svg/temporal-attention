# TemporalAttention Store

A memory system combining **time**, **events**, and **focus** for smarter information retrieval in AI.

## Three Implementations

### 1. EventBased (Best for conversation)

```python
from event_store import EventBasedStore

store = EventBasedStore(
    message_half_life=50,       # Messages until 0.5 decay
    focus_decay_factor=0.5,     # Multiplier when topic shifts
    temporal_weight=0.9,
    attention_weight=0.1,
)

# Track facts under topics
store.put("user_pref", "dark_mode", focus="settings")
store.advance(focus="settings")

# Topic shifts!
store.advance(focus="weather")
store.put("weather", "sunny", focus="weather")

result = store.get("weather")  # Returns "sunny"
```

### 2. TimeBased (Best for databases)

```python
from store import TemporalAttentionStore
from datetime import datetime

store = TemporalAttentionStore(temporal_weight=0.95, attention_weight=0.05)
store.put("ceo", "Alice", valid_from=datetime(2020, 1, 1), valid_to=datetime(2023, 6, 1))
store.put("ceo", "Bob", valid_from=datetime(2023, 6, 1))
result = store.get("ceo", at=datetime(2024, 1, 1))  # Returns "Bob"
```

### 3. Hybrid (Best of both worlds)

```python
from hybrid_store import HybridStore
from datetime import datetime

store = HybridStore(
    time_half_life_hours=24,
    message_half_life=50,
    time_weight=0.33,
    message_weight=0.33,
    focus_weight=0.34,
)
```

## How It Works

| Decay Type | Trigger | Best For |
|------------|---------|----------|
| Time | Wall clock | Stock prices, CEO changes |
| Message | # of messages | Conversation context |
| Focus | Topic shift | Natural context switching |

## Benchmark Results

| System | Score |
|--------|-------|
| EventBased | 100% |
| Hybrid | 100% |
| TimeBased | 100% |

All tests pass on: time facts, message decay, focus shift, attention tiebreaking.

## Key Features

- **Validity filtering**: Only returns facts valid at query time
- **Focus decay**: Topic shifts naturally decay old context
- **Attention tiebreak**: Small boost for frequently accessed facts
- **Configurable weights**: Tune for your use case

## Files

```
temporal-attention/
├── event_store.py       # Event-based (conversation)
├── store.py             # Time-based (databases)
├── hybrid_store.py      # Hybrid (best of both)
├── benchmark_simple.py  # Core tests
└── README.md
```

## When to Use What

| Use Case | Recommended |
|----------|-------------|
| Chatbot memory | EventBased |
| Fact database | TimeBased |
| AI assistant | Hybrid |
| Agent context | EventBased |

## vs SOTA

| System | Approach | Weakness |
|--------|----------|-----------|
| Mem0.ai | Recency + importance | No time validity windows |
| LangChain | Buffer window | No decay, just cuts off |
| RAG | Vector similarity | Ignores time/freshness |
| **Our System** | Time + Message + Focus | Unique focus decay |

### Why Focus Decay Wins

No other system models **topic/context switching**:

```python
# Our system naturally handles this:
store.put("topic", "ai_research", focus="ai")
store.advance(focus="ai")
store.advance(focus="weather")  # Topic shift!
store.put("topic", "sunny", focus="weather")

result = store.get("topic")  
# Returns: "sunny" — AI research decayed due to focus shift
```

This models how humans naturally "forget" old context when topic changes.
