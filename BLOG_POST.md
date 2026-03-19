# Float Permanence: How I Built Memory That Actually Remembers Names

I've been working on AI memory systems for a while and honestly... they're all kind of broken.

## The Problem

Current systems like Mem0, LangChain, and simple RAG treat all facts equally. They store the latest value and call it a day.

**The Overwrite Attack:**
User says their name once. Then 1000 other facts get stored. What happens to the name? Gone.

## The Solution: Float Permanence

I use a single number (0-1) for permanence.

```python
store.auto_put("user_name", "Zach")   # permanence=1.0
store.auto_put("project", "Alpha")      # permanence=0.8
store.auto_put("weather", "sunny")       # permanence=0.1
```

After 1000 messages:
- Zach: stays forever
- Alpha: still there (slow decay)
- sunny: decayed to 0.00004 (gone)

## The Hard Tests

**Test 1: Overwrite Attack**
- FloatMemory: remembers Zach after 1000 facts ✅
- SimpleRAG: overwritten ❌
- WindowMemory: lost after 10 ❌

**Test 2: Project History**
- FloatMemory: stores all versions, can answer "what was it before?" ✅
- SimpleRAG: can't answer ❌

That's what human memory does. We don't forget names. We forget yesterday's weather.

## Code

https://github.com/toxzak-svg/temporal-attention
