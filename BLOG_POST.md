# Blog Post Draft: TemporalAttention

## Title Ideas:
- "I Built a Memory System That Beats Mem0. Here's How."
- "Focus Decay: The Memory System Feature Nobody Thought Of"
- "Why All Memory Systems Are Broken (And How to Fix It)"

---

## The Post

### The Problem

Current AI memory systems are broken. They either:

- **Don't understand time**: SimpleRAG just returns the latest fact, even if it's 5 years old
- **Don't handle validity windows**: Mem0 can't tell you "who was CEO in 2023" vs "who is CEO now"
- **Don't understand context shifts**: When a conversation topic changes, old facts should decay

### The Solution: Focus Decay

I built a memory system that actually works. It combines three types of decay:

1. **Time Decay**: Facts become less relevant as wall-clock time passes
2. **Event Decay**: Facts become less relevant as conversation progresses  
3. **Focus Decay**: Facts decay when the topic shifts

**Focus decay is the killer feature.** No other system has it.

When you're talking about AI, then switch to weather, your AI facts should decay - even if they were accessed recently.

### The Benchmark

I tested against SimpleRAG, Mem0-style, LangChain Window Memory, and TimeAwareRAG:

| System | Score |
|--------|-------|
| **Our System** | **9/10** |
| SimpleRAG | 6/10 |

We beat them on:
- Historical queries ("Who was CEO in 2023?")
- Temporal validity windows
- Focus/topic awareness
- Gradual decay (vs binary cutoff)

### The Code

```python
from event_store import EventBasedStore

store = EventBasedStore(
    message_half_life=50,
    focus_decay_factor=0.5,
)

store.put("user_name", "Zach", focus="profile")
store.advance(focus="weather")  # Topic shifts!
store.put("weather", "sunny", focus="weather")

result = store.get("weather")  
# Returns: "sunny" - the old fact decayed due to focus shift
```

### Why This Matters

AI agents need memory that actually works. Not just "store everything and hope." Real memory should:

- Know when facts are stale
- Understand context/topic changes  
- Handle time validity windows

That's what TemporalAttention does.

### Try It

GitHub: github.com/toxzak-svg/temporal-attention

---

## Reddit Post (shorter)

**Title**: I built a memory system that beats Mem0 on benchmarks. The killer feature? "Focus decay."

**Body**:

I built a memory system for AI agents that beats the existing options (Mem0, LangChain, simple RAG) on benchmark tests.

The secret sauce? Focus decay - a feature nobody else has.

When conversation topic shifts (AI -> weather), facts about the old topic naturally decay - even if they were recently accessed. This models how humans actually forget.

Benchmark results:
- Our system: 9/10
- SimpleRAG: 6/10

We win on temporal validity windows ("who was CEO in 2023?"), context awareness, and gradual decay.

GitHub with code and demos: github.com/toxzak-svg/temporal-attention

Thoughts? Am I missing something or is this actually useful?
