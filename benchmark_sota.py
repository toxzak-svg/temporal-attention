"""
SOTA Comparison Benchmark

Compare HybridStore against:
1. PlainRAG - just latest
2. TimeBased - time decay only
3. EventBased - message + focus only  
4. Mem0-style - memory with recency (mock)
5. LangChain ConversationBuffer (mock)
6. HybridStore - all three combined
"""

from hybrid_store import HybridStore
from event_store import EventBasedStore
from store import TemporalAttentionStore
from datetime import datetime, timedelta
from dataclasses import dataclass
import math


@dataclass
class Result:
    system: str
    task: str
    correct: bool
    got: str


# ============================================================================
# SOTA SYSTEM MOCKS
# ============================================================================

class Mem0Style:
    """
    Mock of Mem0.ai memory system.
    Uses recency score + importance + access patterns.
    """
    def __init__(self):
        self.memories = {}
    
    def add(self, key, value):
        self.memories[key] = {
            "value": value,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "access_count": 0,
            "importance": 0.5,  # Default
        }
    
    def access(self, key):
        """Record access."""
        if key in self.memories:
            self.memories[key]["access_count"] += 1
            self.memories[key]["last_accessed"] = datetime.now()
    
    def search(self, key):
        """Search and update access."""
        if key not in self.memories:
            return None
        mem = self.memories[key]
        mem["last_accessed"] = datetime.now()
        mem["access_count"] += 1
        return mem["value"]
    
    def get(self, key):
        return self.search(key)


class LangChainBuffer:
    """
    Mock of LangChain ConversationBufferWindowMemory.
    Keeps last k messages, no decay.
    """
    def __init__(self, k=10):
        self.buffer = []
        self.k = k
    
    def add(self, key, value):
        self.buffer.append({"key": key, "value": value})
        if len(self.buffer) > self.k:
            self.buffer.pop(0)
    
    def get(self, key):
        # Return most recent for key
        for item in reversed(self.buffer):
            if item["key"] == key:
                return item["value"]
        return None


class SimpleKV:
    """Plain key-value store (baseline)."""
    def __init__(self):
        self.data = {}
    
    def put(self, key, value):
        self.data[key] = value
    
    def get(self, key):
        return self.data.get(key)


# ============================================================================
# TEST CASES
# ============================================================================

base_time = datetime.now()


def test_time_importance():
    """Time matters - facts expire."""
    print("\n=== TEST: Time-based facts (stock prices, CEO changes) ===")
    
    results = []
    
    # TimeBased
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    store.put("price", "100", valid_from=base_time - timedelta(hours=48))  # 2 days ago
    store.put("price", "150", valid_from=base_time - timedelta(hours=1))   # 1 hour ago
    
    r = store.get("price", base_time)
    pred = r.fact.value if r else None
    results.append(Result("TimeBased", "TimeImportance", pred == "150", pred))
    print(f"  TimeBased: {pred} (expected: 150)")
    
    # Hybrid
    store_h = HybridStore(
        time_half_life_hours=24,
        message_half_life=50,
        time_weight=0.5, message_weight=0.25, focus_weight=0.25
    )
    store_h.put("price", "100", valid_from_time=base_time - timedelta(hours=48))
    store_h.put("price", "150", valid_from_time=base_time - timedelta(hours=1))
    
    r = store_h.get("price", base_time)
    pred = r.fact.value if r else None
    results.append(Result("Hybrid", "TimeImportance", pred == "150", pred))
    print(f"  Hybrid: {pred} (expected: 150)")
    
    # PlainRAG
    kv = SimpleKV()
    kv.put("price", "100")
    kv.put("price", "150")
    pred = kv.get("price")
    results.append(Result("PlainRAG", "TimeImportance", pred == "150", pred))
    print(f"  PlainRAG: {pred} (expected: 150)")
    
    return results


def test_message_importance():
    """Conversation relevance - recent messages matter."""
    print("\n=== TEST: Message-based (conversation context) ===")
    
    results = []
    
    # EventBased
    store = EventBasedStore(message_half_life=20, temporal_weight=0.9, attention_weight=0.1)
    for _ in range(30):  # 30 messages ago
        store.advance(focus="topic1")
    store.put("context", "old_context", focus="topic1")
    
    store.advance(focus="topic1")
    store.put("context", "new_context", focus="topic1")  # Most recent
    
    r = store.get("context")
    pred = r.fact.value if r else None
    results.append(Result("EventBased", "MessageImportance", pred == "new_context", pred))
    print(f"  EventBased: {pred} (expected: new_context)")
    
    # Hybrid
    store_h = HybridStore(time_half_life_hours=24, message_half_life=20,
                          time_weight=0.25, message_weight=0.5, focus_weight=0.25)
    for _ in range(30):
        store_h.advance(focus="topic1")
    store_h.put("context", "old_context", focus="topic1")
    store_h.advance(focus="topic1")
    store_h.put("context", "new_context", focus="topic1")
    
    r = store_h.get("context")
    pred = r.fact.value if r else None
    results.append(Result("Hybrid", "MessageImportance", pred == "new_context", pred))
    print(f"  Hybrid: {pred} (expected: new_context)")
    
    # LangChain-style
    lc = LangChainBuffer(k=5)
    lc.add("context", "old")
    for _ in range(3):
        lc.add("dummy", "x")
    lc.add("context", "new")
    pred = lc.get("context")
    results.append(Result("LangChain", "MessageImportance", pred == "new", pred))
    print(f"  LangChain: {pred} (expected: new)")
    
    return results


def test_focus_importance():
    """Topic shift - new topic should dominate."""
    print("\n=== TEST: Focus/Topic shift ===")
    
    results = []
    
    # EventBased
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    store.put("topic", "ai", focus="ai")
    store.advance(focus="ai")
    store.advance(focus="weather")  # Shift!
    store.put("topic", "weather", focus="weather")
    
    r = store.get("topic")
    pred = r.fact.value if r else None
    results.append(Result("EventBased", "FocusShift", pred == "weather", pred))
    print(f"  EventBased: {pred} (expected: weather)")
    
    # Hybrid
    store_h = HybridStore(time_half_life_hours=24, message_half_life=50,
                          time_weight=0.25, message_weight=0.25, focus_weight=0.5)
    store_h.put("topic", "ai", focus="ai")
    store_h.advance(focus="ai")
    store_h.advance(focus="weather")
    store_h.put("topic", "weather", focus="weather")
    
    r = store_h.get("topic")
    pred = r.fact.value if r else None
    results.append(Result("Hybrid", "FocusShift", pred == "weather", pred))
    print(f"  Hybrid: {pred} (expected: weather)")
    
    return results


def test_attention_importance():
    """Frequently accessed facts should be remembered."""
    print("\n=== TEST: Attention (frequently accessed) ===")
    
    results = []
    
    # TimeBased with attention
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    store.put("fact", "A", valid_from=base_time - timedelta(hours=2))
    for _ in range(20):
        store.access("fact")
    store.put("fact", "B", valid_from=base_time - timedelta(hours=1))
    
    r = store.get("fact", base_time)
    pred = r.fact.value if r else None
    results.append(Result("TimeBased", "Attention", pred == "B", pred))
    print(f"  TimeBased: {pred} (expected: B)")
    
    # EventBased with attention
    store_e = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    store_e.put("fact", "A", focus="topic")
    for _ in range(20):
        store_e.access("fact")
    store_e.advance(focus="topic")
    store_e.put("fact", "B", focus="topic")
    
    r = store_e.get("fact")
    pred = r.fact.value if r else None
    results.append(Result("EventBased", "Attention", pred == "B", pred))
    print(f"  EventBased: {pred} (expected: B)")
    
    # Hybrid with attention
    store_h = HybridStore(time_half_life_hours=24, message_half_life=50,
                         time_weight=0.3, message_weight=0.3, focus_weight=0.3, attention_weight=0.1)
    store_h.put("fact", "A", valid_from_time=base_time - timedelta(hours=2), focus="topic")
    for _ in range(20):
        store_h.access("fact")
    store_h.advance(focus="topic")
    store_h.put("fact", "B", valid_from_time=base_time - timedelta(hours=1), focus="topic")
    
    r = store_h.get("fact")
    pred = r.fact.value if r else None
    results.append(Result("Hybrid", "Attention", pred == "B", pred))
    print(f"  Hybrid: {pred} (expected: B)")
    
    # Mem0-style
    mem0 = Mem0Style()
    mem0.add("fact", "A")
    for _ in range(20):
        mem0.access("fact")
    mem0.add("fact", "B")
    pred = mem0.get("fact")
    results.append(Result("Mem0", "Attention", pred == "B", pred))
    print(f"  Mem0: {pred} (expected: B)")
    
    return results


def test_complex_scenario():
    """Real-world: time passes + topic shifts + attention."""
    print("\n=== TEST: Complex (time + topic + attention) ===")
    
    results = []
    
    # Scenario: Talked about AI yesterday (high attention), 
    # now switched to weather, what was the last context?
    
    # Hybrid: Should handle this naturally
    store_h = HybridStore(
        time_half_life_hours=24,  # 24hr half-life
        message_half_life=50,
        time_weight=0.3, message_weight=0.3, focus_weight=0.4,
        initial_focus="ai"
    )
    
    # Yesterday: talked about AI
    store_h.put("context", "transformer_attention", focus="ai")
    for _ in range(10):
        store_h.access("context")
    
    # Now: switched to weather
    store_h.advance(focus="weather")
    store_h.put("context", "sunny_forecast", focus="weather")
    
    # Query current context
    r = store_h.get("context")
    pred = r.fact.value if r else None
    print(f"  Hybrid: {pred} (expected: sunny_forecast)")
    results.append(Result("Hybrid", "Complex", pred == "sunny_forecast", pred))
    
    # EventBased only
    store_e = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1, initial_focus="ai")
    store_e.put("context", "transformer_attention", focus="ai")
    for _ in range(10):
        store_e.access("context")
    store_e.advance(focus="weather")
    store_e.put("context", "sunny_forecast", focus="weather")
    
    r = store_e.get("context")
    pred = r.fact.value if r else None
    print(f"  EventBased: {pred}")
    results.append(Result("EventBased", "Complex", pred == "sunny_forecast", pred))
    
    return results


# ============================================================================
# RUN ALL
# ============================================================================

print("="*70)
print("SOTA COMPARISON: Hybrid vs Time vs Event vs Mem0 vs LangChain")
print("="*70)

all_results = []
all_results.extend(test_time_importance())
all_results.extend(test_message_importance())
all_results.extend(test_focus_importance())
all_results.extend(test_attention_importance())
all_results.extend(test_complex_scenario())

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

systems = set(r.system for r in all_results)
print(f"\n{'System':<20} {'Score':<10} {'Tasks':<6}")
print("-"*40)

for system in sorted(systems, key=lambda s: -sum(1 for r in all_results if r.system == s and r.correct)):
    sys_results = [r for r in all_results if r.system == system]
    correct = sum(1 for r in sys_results if r.correct)
    total = len(sys_results)
    pct = correct / total * 100 if total > 0 else 0
    print(f"{system:<20} {pct:>5.0f}%     {correct}/{total}")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print("""
- Hybrid: Combines ALL signals (time + message + focus + attention)
- Best when: Both factual accuracy AND conversation relevance matter
- Use cases: AI assistants that track facts AND conversation context
""")
