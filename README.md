# TemporalAttention Store

A memory system for AI agents that combines **time**, **events**, and **focus** for smarter information retrieval.

**License**: MIT

## The Problem

Current memory systems are broken:

| System | What's Wrong |
|--------|-------------|
| Simple RAG | Returns latest fact, doesn't understand time |
| Mem0 | Can't answer "who was CEO in 2023?" |
| LangChain | Binary cutoff, no gradual decay |
| Vector DB | Ignores recency |

## The Solution

Three decay types working together:

```
Combined Score = Time Decay × Message Decay × Focus Decay × (1 + 0.001 × Attention)
```

1. **Time Decay**: Facts become stale as wall-clock time passes
2. **Message Decay**: Facts decay as conversation progresses  
3. **Focus Decay**: Facts decay when topic shifts
4. **Attention**: Recently accessed facts get a tiny boost

**Focus decay is unique** - no other system has it.

## What The Benchmarks Test

We tested on **10 hard scenarios**:

1. **Historical Query**: "Who was CEO in 2023?" - requires time validity windows
2. **Context Shift**: "We discussed AI, now weather, what's the current topic?"
3. **Attention Tiebreaker**: Old fact accessed 100x vs new fact - which wins?
4. **Focus Decay**: Facts from old topic should decay when topic changes
5. **Time Decay**: Facts from 5 hours ago vs now - with 1-hour half-life
6. **Validity Boundaries**: Fact expires at exact query time
7. **Message Decay**: 10 messages ago vs now - with 5-message half-life
8. **Multiple Focus**: 10 rapid topic switches - which fact wins?
9. **Weight Sensitivity**: Different time/message weights - results consistent?
10. **Empty Store**: Query non-existent key - returns None gracefully

## Results

| System | Tests Passed |
|--------|-------------|
| **TemporalAttention** | **10/10** |
| SimpleRAG | 6/10 |

SimpleRAG fails on: historical queries, focus decay, time validity windows.

## Quick Start

```python
from event_store import EventBasedStore

# For conversation/chat
store = EventBasedStore(
    message_half_life=50,    # Decay after 50 messages
    focus_decay_factor=0.5,  # 50% penalty when topic shifts
)

# Track facts
store.put("user_name", "Zach", focus="profile")
store.advance(focus="profile")

# Topic shifts!
store.advance(focus="weather")
store.put("weather", "sunny", focus="weather")

# Query
result = store.get("weather")
# Returns: "sunny" - old facts decayed due to focus shift
```

```python
from store import TemporalAttentionStore

# For facts with time windows
store = TemporalAttentionStore()

store.put("ceo", "Alice", valid_from=datetime(2020,1,1), valid_to=datetime(2023,6,1))
store.put("ceo", "Bob", valid_from=datetime(2023,6,1))

# Ask about the past
result = store.get("ceo", datetime(2022,6,1))
# Returns: "Alice"
```

## Why Focus Decay Matters

```
Before: focus="ai", topic="transformers"
After:  focus="weather", topic="sunny"

transformers: focus_decay=0.5 → loses decisively
sunny:        focus_decay=1.0 → wins
```

This models how humans naturally "switch context" when topics change.

## Installation

```bash
git clone https://github.com/toxzak-svg/temporal-attention
cd temporal-attention
pip install -e .
```

Or just copy the `.py` files you need.

## Files

- `event_store.py` - Event-based (conversation)
- `store.py` - Time-based (facts)  
- `hybrid_store.py` - Both combined
- `demo.py` - Interactive demo
- `benchmark_*.py` - Test suites

## License

MIT License - use it however you want.
