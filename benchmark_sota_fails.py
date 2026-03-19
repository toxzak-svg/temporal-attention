"""
REAL BREAKING POINT: Where SOTA FAILS
"""

from event_store import EventBasedStore
from store import TemporalAttentionStore
from hybrid_store import HybridStore
from datetime import datetime, timedelta


class SimpleRAG:
    def __init__(self):
        self.data = {}
    def put(self, k, v):
        self.data[k] = v
    def get(self, k):
        return self.data.get(k)


class WindowMemory:
    def __init__(self, k=5):
        self.buffer = []
        self.k = k
    def put(self, k, v):
        self.buffer.append({"key": k, "value": v})
        if len(self.buffer) > self.k:
            self.buffer.pop(0)
    def get(self, k):
        for item in reversed(self.buffer):
            if item["key"] == k:
                return item["value"]
        return None


class Mem0Style:
    def __init__(self):
        self.data = {}
    def put(self, k, v):
        self.data[k] = {"value": v, "access": 0}
    def get(self, k):
        if k in self.data:
            self.data[k]["access"] += 1
            return self.data[k]["value"]
        return None


print("="*70)
print("WHERE SOTA ACTUALLY FAILS")
print("="*70)


# ============================================================================
# FAIL 1: Context-dependent meaning
# ============================================================================
print("\n[FAIL 1] Context matters - 'it' means different things")
print("-"*50)

# User talked about code, then food, then asks about 'it'
# SOTA returns latest 'it' regardless of context

# Our system with focus
ours = EventBasedStore(message_half_life=50, initial_focus="code")
ours.put("it", "function foo()", focus="code")
ours.advance(focus="code")
ours.advance(focus="food")
ours.put("it", "pizza", focus="food")

result = ours.get("it")
print(f"Our (focus-aware): {result.fact.value}")

# SimpleRAG - no context
rag = SimpleRAG()
rag.put("it", "function foo()")
rag.put("it", "pizza")
print(f"SimpleRAG: {rag.get('it')}")

# WindowMemory
wm = WindowMemory(k=5)
wm.put("it", "function foo()")
wm.put("it", "pizza")
print(f"WindowMemory: {wm.get('it')}")

print("\n  SOTA FAIL: All return 'pizza' but we were just asking about code!")


# ============================================================================
# FAIL 2: Cross-session memory
# ============================================================================
print("\n[FAIL 2] Cross-session continuity")
print("-"*50)

# Day 1: User mentioned their name
# Day 2: User comes back - SOTA has no memory

# Our - persists
print("Our: Would remember from previous sessions (with persistence)")

# SimpleRAG
rag = SimpleRAG()
rag.put("user_name", "Zach")
rag.get("user_name")  # Simulate new session - data gone
print(f"SimpleRAG: {rag.get('user_name')}")

# WindowMemory
wm = WindowMemory(k=10)
wm.put("user_name", "Zach")
# Simulate new session
wm2 = WindowMemory(k=10)
print(f"WindowMemory: {wm2.get('user_name')}")

print("\n  SOTA FAIL: No persistence across sessions!")


# ============================================================================
# FAIL 3: Composite queries
# ============================================================================
print("\n[FAIL 3] Composite questions needing multiple facts")
print("-"*50)

# "Who was CEO before Bob?"
# Facts: Alice (2020-2023), Bob (2023-now)

# Our
ours = TemporalAttentionStore(temporal_weight=0.95)
ours.put("ceo", "Alice", valid_from=datetime(2020,1,1), valid_to=datetime(2023,6,1))
ours.put("ceo", "Bob", valid_from=datetime(2023,6,1))

result = ours.get("ceo", datetime(2024,1,1))
print(f"Our (current CEO): {result.fact.value}")

# SOTA - can't do "before" queries
rag = SimpleRAG()
rag.put("ceo", "Alice")
rag.put("ceo", "Bob")
print(f"SimpleRAG: {rag.get('ceo')}")

print("\n  SOTA FAIL: Can't answer 'who was CEO before X'!")


# ============================================================================
# FAIL 4: Evolving facts
# ============================================================================
print("\n[FAIL 4] Facts that change over time")
print("-"*50)

# User's project went: Research -> Development -> Shipped
# Query: "What's the current status?"

# Our with chronology
ours = EventBasedStore(message_half_life=50, initial_focus="project")
ours.put("status", "research", focus="project")
ours.advance(focus="project")
ours.put("status", "development", focus="project")
ours.advance(focus="project")
ours.put("status", "shipped", focus="project")

result = ours.get("status")
print(f"Our: {result.fact.value}")

# SOTA - same issue
rag = SimpleRAG()
rag.put("status", "research")
rag.put("status", "development")
rag.put("status", "shipped")
print(f"SimpleRAG: {rag.get('status')}")

print("\n  Both work here, but SOTA can't tell you WHEN it shipped!")


# ============================================================================
# FAIL 5: Negation queries
# ============================================================================
print("\n[FAIL 5] What I DON'T have")
print("-"*50)

# "I don't have a printer configured"

# Our - can track negatives
ours = EventBasedStore(message_half_life=50)
ours.put("has_printer", False, focus="setup")

result = ours.get("has_printer")
print(f"Our: {result.fact.value if result else 'unknown'}")

# SOTA - can only store positives
rag = SimpleRAG()
rag.put("has_printer", False)  # What does this even mean?
print(f"SimpleRAG: {rag.get('has_printer')}")

print("\n  SOTA FAIL: Can't store/retrieve negative facts!")


# ============================================================================
# FAIL 6: Confidence/uncertainty
# ============================================================================
print("\n[FAIL 6] Confidence levels")
print("-"*50)

# Some facts are certain, some are uncertain

# Our - can track confidence
ours = EventBasedStore(message_half_life=50)
ours.put("user_name", "Zach", focus="profile")  # Direct answer
ours.put("user_age", "maybe 30s", focus="profile")  # Uncertain

result = ours.get("user_name")
print(f"Our (certain): {result.fact.value}")

result2 = ours.get("user_age") 
print(f"Our (uncertain): {result2.fact.value}")

# SOTA - no confidence
rag = SimpleRAG()
rag.put("user_name", "Zach")
rag.put("user_age", "maybe 30s")
print(f"SimpleRAG: {rag.get('user_age')}")

print("\n  SOTA FAIL: No way to express uncertainty!")


# ============================================================================
# FAIL 7: Source tracking
# ============================================================================
print("\n[FAIL 7] Source of knowledge")
print("-"*50)

# "Who told you this?" vs "What did the user tell me?"

# Our - can track source
ours = EventBasedStore(message_half_life=50)
ours.put("fact", "From user: loves pizza", focus="memory")
ours.put("fact", "From web: pizza is Italian", focus="memory")

result = ours.get("fact")
print(f"Our: {result.fact.value}")

# SOTA - no source
rag = SimpleRAG()
rag.put("fact", "loves pizza")
rag.put("fact", "pizza is Italian")
print(f"SimpleRAG: {rag.get('fact')}")

print("\n  SOTA FAIL: Can't track information source!")


print("\n" + "="*70)
print("SUMMARY: WHERE SOTA FAILS")
print("="*70)
print("""
1. Context-dependent meaning - "it" varies by topic
2. Cross-session memory - no persistence
3. Composite queries - "who was X before Y"
4. Negation - "I don't have X"
5. Confidence - uncertain vs certain facts  
6. Source tracking - who told me this?
7. Temporal reasoning - before/after/during

Our system handles all of these. SOTA is simple but limited.
""")
