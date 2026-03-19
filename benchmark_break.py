"""
BREAK THE SYSTEM - Ultra-hard edge cases
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


def test_break_1():
    """0 message difference - pure attention tiebreaker"""
    print("\n[BREAK 1] 0 messages apart, attention only")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    store.put("key", "has_access", focus="topic")
    store.access("key")
    store.access("key")
    store.access("key")
    store.put("key", "no_access", focus="topic")  # Same message, added AFTER
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  has_access was accessed 3x before no_access was added")
    print(f"  Result: {pred}")
    # has_access should win - it was accessed first (max by valid_from_message)
    return Result("EventBased", "AttnTiebreaker", pred == "has_access", pred, "has_access")


def test_break_2():
    """Focus shift during access"""
    print("\n[BREAK 2] Access before focus shift")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1, initial_focus="ai")
    store.put("key", "ai_data", focus="ai")
    for _ in range(10):
        store.access("key")  # Access while in ai focus
    
    store.advance(focus="ai")
    store.advance(focus="weather")  # Shift focus
    
    store.put("key", "weather_data", focus="weather")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  ai_data accessed 10x before focus shift")
    print(f"  Result: {pred}")
    return Result("EventBased", "AccessFocusShift", pred == "weather_data", pred, "weather_data")


def test_break_3():
    """Time decay near zero"""
    print("\n[BREAK 3] Facts at expiration boundary")
    
    store = HybridStore(time_half_life_hours=1, message_half_life=50,
                       time_weight=0.5, message_weight=0.5)
    
    # Add fact 10 hours ago (way past 1hr half-life)
    store.put("key", "ancient", valid_from_time=base_time - timedelta(hours=10))
    store.put("key", "recent", valid_from_time=base_time - timedelta(minutes=30))
    
    r = store.get("key", base_time)
    pred = r.fact.value if r else None
    
    print(f"  ancient time_decay: {r.time_decay:.6f}")
    print(f"  recent time_decay: {r.time_decay:.6f}")
    print(f"  Result: {pred}")
    return Result("Hybrid", "NearExpiration", pred == "recent", pred, "recent")


def test_break_4():
    """Multiple keys, cross-contamination"""
    print("\n[BREAK 4] Multiple keys isolation")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    
    store.put("key_a", "val_a_old", focus="topic")
    for _ in range(10):
        store.access("key_a")
    
    store.advance(focus="topic")
    store.put("key_a", "val_a_new", focus="topic")
    store.put("key_b", "val_b", focus="topic")
    
    r_a = store.get("key_a")
    r_b = store.get("key_b")
    
    print(f"  key_a: {r_a.fact.value}")
    print(f"  key_b: {r_b.fact.value}")
    
    return Result("EventBased", "MultiKey", 
                  r_a.fact.value == "val_a_new" and r_b.fact.value == "val_b",
                  f"{r_a.fact.value}, {r_b.fact.value}", "val_a_new, val_b")


def test_break_5():
    """Attention overflow"""
    print("\n[BREAK 5] Massive attention (1000 accesses)")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    store.put("key", "old_hot", focus="topic")
    for _ in range(1000):
        store.access("key")
    
    store.advance(focus="topic")
    store.put("key", "new_cold", focus="topic")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  old: msg={r.message_decay:.3f}, attn={r.attention_score:.3f}")
    print(f"  Result: {pred}")
    return Result("EventBased", "MassiveAttn", pred == "new_cold", pred, "new_cold")


def test_break_6():
    """Rapid focus cycling"""
    print("\n[BREAK 6] Rapid focus changes")
    
    store = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1, initial_focus="a")
    
    for i in range(20):
        next_focus = chr(ord('a') + (i % 3))
        store.advance(focus=next_focus)
        store.put("key", f"val_{i}", focus=next_focus)
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  Current focus: {store.current_focus}")
    print(f"  Result: {pred}")
    # Should be val_19 (most recent)
    return Result("EventBased", "RapidFocus", pred == "val_19", pred, "val_19")


def test_break_7():
    """Hybrid weight sensitivity"""
    print("\n[BREAK 7] Weight sensitivity")
    
    results = []
    
    for tw in [0.3, 0.5, 0.7, 0.9]:
        store = HybridStore(time_half_life_hours=24, message_half_life=50,
                           time_weight=tw, message_weight=1-tw)
        
        store.put("key", "old", valid_from_time=base_time - timedelta(hours=12))
        store.put("key", "new", valid_from_time=base_time)
        
        r = store.get("key", base_time)
        pred = r.fact.value if r else None
        results.append((tw, pred))
        print(f"  tw={tw}: {pred}")
    
    # All should return "new" regardless of weights
    all_new = all(p == "new" for _, p in results)
    return Result("Hybrid", "WeightSensitivity", all_new, str(results), "all new")


def test_break_8():
    """Empty store"""
    print("\n[BREAK 8] Query empty store")
    
    store = EventBasedStore(message_half_life=50)
    r = store.get("nonexistent")
    pred = r.fact.value if r else None
    
    print(f"  Result: {pred}")
    return Result("EventBased", "EmptyStore", pred is None, str(pred), "None")


def test_break_9():
    """All facts expired"""
    print("\n[BREAK 9] All facts expired")
    
    store = EventBasedStore(message_half_life=10)
    store.put("key", "fact1", valid_to_message=5)
    store.put("key", "fact2", valid_to_message=10)
    
    for _ in range(20):
        store.advance(focus="topic")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  Result: {pred}")
    return Result("EventBased", "AllExpired", pred is None, str(pred), "None")


def test_break_10():
    """Boundary: valid_to exactly at query"""
    print("\n[BREAK 10] valid_to at exact query time")
    
    store = EventBasedStore(message_half_life=50)
    store.put("key", "expiring", valid_to_message=10)
    store.put("key", "next", valid_from_message=10)
    
    for _ in range(10):
        store.advance(focus="topic")
    
    r = store.get("key")
    pred = r.fact.value if r else None
    
    print(f"  Result: {pred}")
    # valid_from <= at < valid_to, so both could be valid but next is newer
    return Result("EventBased", "BoundaryValidTo", pred == "next", pred, "next")


# Run
print("="*60)
print("BREAK THE SYSTEM - Ultra-hard")
print("="*60)

results = []
tests = [test_break_1, test_break_2, test_break_3, test_break_4, test_break_5,
         test_break_6, test_break_7, test_break_8, test_break_9, test_break_10]

for t in tests:
    try:
        r = t()
        results.append(r)
    except Exception as e:
        print(f"  ERROR in {t.__name__}: {e}")

passed = sum(1 for r in results if r.correct)
print(f"\nScore: {passed}/{len(results)} ({passed/len(results)*100:.0f}%)")

for r in results:
    status = "OK" if r.correct else "FAIL"
    print(f"  {status} {r.task}")

if passed < len(results):
    print(f"\n[!] FAILED:")
    for r in results:
        if not r.correct:
            print(f"    {r.task}: got {r.got}, expected {r.expected}")
