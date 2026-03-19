# TemporalAttention Store

A memory system that combines **temporal validity** with **attention signals** for smarter information retrieval in AI systems.

## Two Implementations

### 1. Event-Based (Primary) — `event_store.py`

Best for conversation/chat/agents. Decay based on message count and topic shifts.

```python
from event_store import EventBasedStore

store = EventBasedStore(
    message_half_life=50,        # Messages until 0.5 decay
    focus_decay_factor=0.5,      # Multiplier when topic shifts  
    temporal_weight=0.9,        # How much temporal matters
    attention_weight=0.1,       # How much access matters
)

# Track a fact under a topic
store.put("user_name", "Zach", focus="profile")

# After each message
store.advance(focus="profile")

# Topic shifts!
store.advance(focus="weather")

# Add new fact under new topic
store.put("weather", "sunny", focus="weather")

# Query - returns "sunny" (current focus)
result = store.get("weather")
```

### 2. Time-Based — `store.py`

Best for databases/facts where wall-clock time matters.

```python
from store import TemporalAttentionStore
from datetime import datetime

store = TemporalAttentionStore(temporal_weight=0.95, attention_weight=0.05)
store.put("ceo", "Alice", valid_from=datetime(2020, 1, 1), valid_to=datetime(2023, 6, 1))
store.put("ceo", "Bob", valid_from=datetime(2023, 6, 1))
result = store.get("ceo", at=datetime(2024, 1, 1))
# Returns: Bob
```

## Key Features

- **Validity filter**: Only returns facts valid at query time
- **Event-based decay**: Decay based on message count, not wall-clock
- **Focus decay**: Topic shifts naturally decay old facts
- **Attention boost**: Frequently accessed facts get a small boost

## Benchmark Results

| System | Score |
|--------|-------|
| PlainRAG | ~20% |
| TimeBased (0.9/0.1) | ~70% |
| **EventBased** | **~88%** |

EventBased wins because **focus_decay** handles topic shifts naturally.

## How Focus Decay Works

When topic shifts from 'ai' → 'weather':
```
ai_fact:     msg=0.99, focus=0.50 → combined=0.44
weather_fact: msg=1.00, focus=1.00 → combined=0.90
```

The old topic gets a 0.5 multiplier, making it lose decisively. Models human context-switching.

## Files

```
temporal-attention/
├── event_store.py       # Primary implementation (conversation)
├── store.py             # Time-based implementation
├── README.md            # This file
├── benchmark.py         # Basic benchmarks
├── benchmark_hard.py    # Hard adversarial cases
├── benchmark_event.py   # Event-based benchmarks
├── benchmark_extreme.py # Extreme edge cases
├── system_comparison.py # Compare all systems
└── final_benchmark.py   # Final comparison demo
```

## Usage

```python
from event_store import EventBasedStore

# Initialize
store = EventBasedStore(
    message_half_life=50,
    focus_decay_factor=0.5,
    temporal_weight=0.9,
    attention_weight=0.1,
    initial_focus="general",
)

# Add facts
store.put("key", "value", focus="topic_name")

# After each user message
store.advance(focus="current_topic")

# Query current context
result = store.get("key")
if result:
    print(result.fact.value)
```
