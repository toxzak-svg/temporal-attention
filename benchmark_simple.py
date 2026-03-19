"""
Simplified SOTA Comparison
"""

from hybrid_store import HybridStore
from event_store import EventBasedStore
from store import TemporalAttentionStore
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class Result:
    system: str
    task: str
    correct: bool
    got: str


base_time = datetime.now()


def run_tests():
    results = []
    
    # =========================================================================
    # TEST 1: Time matters (stock prices)
    # =========================================================================
    print("\n[TEST 1] Time-based facts (48hr vs 1hr)")
    
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    store.put("price", "100", valid_from=base_time - timedelta(hours=48))
    store.put("price", "150", valid_from=base_time - timedelta(hours=1))
    r = store.get("price", base_time)
    pred = r.fact.value if r else None
    results.append(Result("TimeBased", "TimeFacts", pred == "150", pred))
    print(f"  TimeBased: {pred} (expected: 150)")
    
    # =========================================================================
    # TEST 2: Message matters (conversation context)
    # =========================================================================
    print("\n[TEST 2] Message-based (30 messages ago vs now)")
    
    store = EventBasedStore(message_half_life=20, temporal_weight=0.9, attention_weight=0.1)
    store.put("context", "old", focus="topic")  # Add first (30 messages ago)
    for _ in range(30):  # Advance 30 times
        store.advance(focus="topic")
    store.put("context", "new", focus="topic")  # Add now
    r = store.get("context")
    pred = r.fact.value if r else None
    results.append(Result("EventBased", "MsgFacts", pred == "new", pred))
    print(f"  EventBased: {pred} (expected: new)")
    
    # =========================================================================
    # TEST 3: Focus shift
    # =========================================================================
    print("\n[TEST 3] Focus/topic shift")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1, initial_focus="ai")
    store.put("topic", "ai_stuff", focus="ai")
    store.advance(focus="ai")
    store.advance(focus="weather")
    store.put("topic", "weather", focus="weather")
    r = store.get("topic")
    pred = r.fact.value if r else None
    results.append(Result("EventBased", "FocusShift", pred == "weather", pred))
    print(f"  EventBased: {pred} (expected: weather)")
    
    # =========================================================================
    # TEST 4: Hybrid handles both
    # =========================================================================
    print("\n[TEST 4] Hybrid (time + message + focus)")
    
    store = HybridStore(
        time_half_life_hours=24,
        message_half_life=50,
        time_weight=0.33, message_weight=0.33, focus_weight=0.34,
        initial_focus="ai"
    )
    store.put("data", "old_data", valid_from_time=base_time - timedelta(hours=48), focus="ai")
    store.advance(focus="ai")
    store.advance(focus="weather")
    store.put("data", "new_data", focus="weather")
    r = store.get("data", base_time)
    pred = r.fact.value if r else None
    results.append(Result("Hybrid", "Combined", pred == "new_data", pred))
    print(f"  Hybrid: {pred} (expected: new_data)")
    
    # =========================================================================
    # TEST 5: Attention (recent should win over old+accessed)
    # =========================================================================
    print("\n[TEST 5] Attention (new beats old+100 access)")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1, initial_focus="topic")
    store.put("fact", "old_hot", focus="topic")
    for _ in range(100):
        store.access("fact")
    store.advance(focus="topic")
    store.put("fact", "new_cold", focus="topic")
    r = store.get("fact")
    pred = r.fact.value if r else None
    results.append(Result("EventBased", "Attention", pred == "new_cold", pred))
    print(f"  EventBased: {pred} (expected: new_cold)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    systems = set(r.system for r in results)
    for sys in sorted(systems, key=lambda s: -sum(1 for r in results if r.system == s and r.correct)):
        sys_r = [r for r in results if r.system == sys]
        correct = sum(1 for r in sys_r if r.correct)
        total = len(sys_r)
        pct = correct / total * 100
        print(f"{sys}: {correct}/{total} ({pct:.0f}%)")


if __name__ == "__main__":
    run_tests()
