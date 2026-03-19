---

I've been working on AI memory for a while and honestly it's all kinda broken.

RAG just gives you the latest thing. Mem0 is fine but doesn't get time. LangChain memory is just a buffer window.

So I tried something different. Called it "focus decay" - basically when the conversation topic shifts, old stuff should fade. Not because it's old, but because it's about a different topic.

Example: we were talking about AI, now we're on weather. My ai stuff should fade even if I just accessed it.

Made some benchmarks. 10 hard cases. Mine passed all 10. SimpleRAG (just storing latest) passed 6.

Not trying to sell anything, just sharing because I'm curious if this is actually useful or if I'm missing something.

github.com/toxzak-svg/temporal-attention

---
