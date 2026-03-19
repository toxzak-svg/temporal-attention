---

**Title**: I built a better AI memory system. The key insight? "Focus decay"

**Post**:

So I've been messing around with AI memory for a while - trying to get my agent to actually remember things across conversations instead of just context windows.

The problem is everything out there kinda sucks:

- **Simple RAG** just stores the latest thing. Doesn't understand that old info might be more relevant.
- **Mem0** is better but still doesn't really get time. Can't ask "who was CEO in 2023" and get the right answer.
- **LangChain** just keeps a sliding window of messages. Binary - either in the window or not. No sense of what's actually important.

So I tried something different. Called it **focus decay**.

The idea: when the conversation topic shifts, facts about the OLD topic should naturally fade. Not because they're old, but because they're about a different topic. Just like how humans don't think about the previous topic once we've moved on.

Here's the code:

```python
from event_store import EventBasedStore

store = EventBasedStore(message_half_life=50, focus_decay_factor=0.5)

# We're talking about AI stuff
store.put("topic", "transformers", focus="ai")
store.put("topic", "attention", focus="ai")

# Now we switch to weather
store.advance(focus="weather")
store.put("topic", "sunny", focus="weather")

# Query - returns "sunny"
# The AI facts decayed because the focus shifted
```

The math is basically:

```
Relevance = Time Decay × Message Count Decay × Focus Decay × Attention Boost
```

I made 10 hard test cases to verify it actually works. Things like:

- **The "I told you this before" problem** - User says "my name is Zach", then talks about 1000 other things. Ask "what's my name?" and SimpleRAG returns "Zach" with full confidence. Our system: after 50+ messages it decays - acknowledges we haven't talked about it recently.
- **"Who was CEO in 2023?"** (need time validity)
- Facts at expiration boundaries
- Rapid topic switching

My system got 10/10. Simple RAG got 6/10. It fails on the time and focus stuff.

Not claiming this is perfect or production ready. It's about 300 lines of Python. But I think the core idea of focus decay is actually useful and I haven't seen anyone else talk about it.

GitHub if you want to look: github.com/toxzak-svg/temporal-attention

Am I onto something here or am I missing a obvious flaw?

---

**Subtitle**: Also, where would be the best place to post something like this? Tried r/ArtificialAgent but not sure.

---
