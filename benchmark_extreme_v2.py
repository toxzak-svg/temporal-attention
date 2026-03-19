"""
EXTREME ADVERSARIAL BENCHMARK - Break the system

Designed to find cracks in temporal+attention+focus.
"""

from event_store import EventBasedStore
from store import TemporalAttentionStore
from hybrid_store import HybridStore
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class Result:
    system: str
    task: str
    correct: bool
    got: str
    expected: str


base_time = datetime.now()


def test_extreme_1():
    """Micro-temporal-difference: 1 message apart vs 100 accesses"""
    print("\n[EXTREME 1] 1 message apart vs 100 accesses")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    store.put("key", "A", focus="topic")
    for _ in range(100):
        store.access("key")
    store.advance(focus="topic")
    store.put("key", "B", focus="topic")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    # B is 1 message newer. With 0.9 temporal, should B win?
    # A: msg=0.99, attn=0.15 → comb = 0.99 * (1 + 0.001*0.15) = 1.001
    # B: msg=1.00, attn=0 → comb = 1.00 * (1 + 0) = 1.00
    # A wins! Bug!
    
    print(f"  Result: {pred}")
    return Result("EventBased", "MicroTemporal", pred == "B", pred, "B")


def test_extreme_2():
    """Focus decay should NOT apply to current focus"""
    print("\n[EXTREME 2] Focus decay bug - current focus")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1, initial_focus="ai")
    store.put("key", "old", focus="ai")
    store.advance(focus="ai")
    store.advance(focus="ai") 
    store.advance(focus="ai")
    store.put("key", "new", focus="ai")  # Same focus, different time
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  Result: {pred}")
    return Result("EventBased", "SameFocus", pred == "new", pred, "new")


def test_extreme_3():
    """Multiple focus shifts"""
    print("\n[EXTREME 3] Multiple focus shifts")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1, initial_focus="a")
    store.put("key", "val_a", focus="a")
    store.advance(focus="a")
    store.advance(focus="b")
    store.put("key", "val_b", focus="b")
    store.advance(focus="b")
    store.advance(focus="c")
    store.put("key", "val_c", focus="c")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  Result: {pred}")
    return Result("EventBased", "MultiFocus", pred == "val_c", pred, "val_c")


def test_extreme_4():
    """Attention decay over time"""
    print("\n[EXTREME 4] Old access should decay")
    
    store = EventBasedStore(message_half_life=10, temporal_weight=0.9, attention_weight=0.1)
    store.put("key", "old_access", focus="topic")
    for _ in range(50):
        store.access("key")
    
    # Advance 20 messages - access should decay
    for _ in range(20):
        store.advance(focus="topic")
    
    store.put("key", "new_fact", focus="topic")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  Result: {pred}")
    return Result("EventBased", "AttnDecay", pred == "new_fact", pred, "new_fact")


def test_extreme_5():
    """Time + Event conflict"""
    print("\n[EXTREME 5] Time vs Event conflict")
    
    # Hybrid: time says old, event says new
    store = HybridStore(
        time_half_life_hours=1,  # 1 hour half-life
        message_half_life=100,   # 100 message half-life
        time_weight=0.5, message_weight=0.5,
        initial_focus="topic"
    )
    
    # Add fact 2 hours ago (time decay ~0.25), 5 messages ago (msg decay ~0.97)
    store.put("key", "old_by_time", valid_from_time=base_time - timedelta(hours=2), focus="topic")
    for _ in range(5):
        store.advance(focus="topic")
    
    # Add fact now (time decay 1.0, msg decay 1.0)
    store.put("key", "new", focus="topic")
    
    r = store.get("key", base_time)
    pred = r.fact.value if r else None
    
    print(f"  old: time={r.time_decay:.3f}, msg={r.message_decay:.3f}")
    print(f"  Result: {pred}")
    return Result("Hybrid", "TimeEventConflict", pred == "new", pred, "new")


def test_extreme_6():
    """Validity window with attention"""
    print("\n[EXTREME 6] Validity window + attention")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    
    # Fact valid for first 10 messages only
    store.put("key", "temp_fact", focus="topic", valid_to_message=10)
    
    for _ in range(5):
        store.advance(focus="topic")
    store.put("key", "perm_fact", focus="topic")
    
    for _ in range(10):
        store.advance(focus="topic")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  Result: {pred}")
    return Result("EventBased", "ValidityAttn", pred == "perm_fact", pred, "perm_fact")


def test_extreme_7():
    """Equal temporal, focus difference"""
    print("\n[EXTREME 7] Equal temporal, focus difference")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1, initial_focus="ai")
    store.put("key", "ai_val", focus="ai")
    store.put("key", "weather_val", focus="weather")  # Same message count, different focus
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    # Current focus is "ai", so ai_val should win
    print(f"  Current focus: {store.current_focus}")
    print(f"  Result: {pred}")
    return Result("EventBased", "EqualTempFocus", pred == "ai_val", pred, "ai_val")


def test_extreme_8():
    """Many facts, pick highest score"""
    print("\n[EXTREME 8] 10 facts, various scores")
    
    store = EventBasedStore(message_half_life=30, temporal_weight=0.9, attention_weight=0.1, initial_focus="topic")
    
    for i in range(10):
        store.put("key", f"fact_{i}", focus="topic")
        if i == 5:
            for _ in range(20):
                store.access("key")
        store.advance(focus="topic")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  Result: {pred}")
    # fact_9 is newest, should win (despite fact_5 having attention)
    return Result("EventBased", "ManyFacts", pred == "fact_9", pred, "fact_9")


def test_extreme_9():
    """Negative time (future facts)"""
    print("\n[EXTREME 9] Future facts should be ignored")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    store.put("key", "future", focus="topic", valid_from_message=1000)
    store.put("key", "present", focus="topic")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  Result: {pred}")
    return Result("EventBased", "FutureFacts", pred == "present", pred, "present")


def test_extreme_10():
    """Time decay sensitivity"""
    print("\n[EXTREME 10] Time decay at boundary")
    
    store = HybridStore(time_half_life_hours=24, message_half_life=50,
                       time_weight=0.5, message_weight=0.5)
    
    # Exactly at half-life
    store.put("key", "old", valid_from_time=base_time - timedelta(hours=24))
    store.put("key", "new", valid_from_time=base_time)
    
    r = store.get("key", base_time)
    pred = r.fact.value if r else None
    
    print(f"  old time_decay: {r.time_decay:.3f}")
    print(f"  Result: {pred}")
    # Should pick new (higher time_decay)
    return Result("Hybrid", "TimeBoundary", pred == "new", pred, "new")


# Run all
print("="*60)
print("EXTREME ADVERSARIAL BENCHMARK")
print("="*60)

results = []
tests = [
    test_extreme_1,
    test_extreme_2,
    test_extreme_3,
    test_extreme_4,
    test_extreme_5,
    test_extreme_6,
    test_extreme_7,
    test_extreme_8,
    test_extreme_9,
    test_extreme_10,
]

for test in tests:
    try:
        r = test()
        results.append(r)
    except Exception as e:
        print(f"  ERROR: {e}")

# Summary
print("\n" + "="*60)
print("RESULTS")
print("="*60)

passed = sum(1 for r in results if r.correct)
total = len(results)
print(f"\nScore: {passed}/{total} ({passed/total*100:.0f}%)")

for r in results:
    status = "OK" if r.correct else "FAIL"
    print(f"  {status} {r.task}: got={r.got}, expected={r.expected}")

if passed < total:
    print(f"\n[!] FAILED TASKS:")
    for r in results:
        if not r.correct:
            print(f"    - {r.task}")
