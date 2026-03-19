"""
SOTA HARD COMPARISON - Each system pushed to FAIL

Compare systems on scenarios designed to expose weaknesses.
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
    expected: str


base_time = datetime.now()


# ============================================================================
# ADVERSARIAL TESTS
# ============================================================================

def test_time_vs_message_conflict():
    """
    SCENARIO: Time says old fact is more recent, but message count says new.
    This can happen if user was idle for a day but had 0 messages.
    
    Time-based: returns OLD (correct for facts)
    Message-based: returns NEW (correct for conversation)
    Hybrid: balances both
    """
    print("\n=== TIME vs MESSAGE CONFLICT ===")
    print("Scenario: User idle 24hrs (0 messages), but 48hrs passed in time")
    
    results = []
    
    # TimeBased: time says fact2 is newer (1hr vs 48hr)
    store_t = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    store_t.put("fact", "old_by_time", valid_from=base_time - timedelta(hours=48))
    store_t.put("fact", "new_by_time", valid_from=base_time - timedelta(hours=1))
    
    r = store_t.get("fact", base_time)
    pred = r.fact.value if r else None
    results.append(Result("TimeBased", "TimeVsMsg", pred == "new_by_time", pred, "new_by_time"))
    print(f"  TimeBased: {pred}")
    
    # EventBased: message count same, so returns latest added
    store_e = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    store_e.put("fact", "msg1", focus="topic")
    store_e.put("fact", "msg2", focus="topic")  # Latest
    
    r = store_e.get("fact")
    pred = r.fact.value if r else None
    results.append(Result("EventBased", "TimeVsMsg", pred == "msg2", pred, "msg2"))
    print(f"  EventBased: {pred}")
    
    # Hybrid: should balance both
    store_h = HybridStore(time_half_life_hours=24, message_half_life=50,
                          time_weight=0.5, message_weight=0.5)
    store_h.put("fact", "both_old", valid_from_time=base_time - timedelta(hours=48))
    store_h.put("fact", "both_new", valid_from_time=base_time - timedelta(hours=1))
    
    r = store_h.get("fact", base_time)
    pred = r.fact.value if r else None
    results.append(Result("Hybrid", "TimeVsMsg", pred == "both_new", pred, "both_new"))
    print(f"  Hybrid: {pred}")
    
    return results


def test_focus_with_time():
    """
    SCENARIO: Old focus but accessed recently vs new focus not accessed.
    Which wins?
    """
    print("\n=== FOCUS + ATTENTION TRADE ===")
    print("Scenario: Old topic (accessed 10x) vs new topic (accessed 0x)")
    
    results = []
    
    # EventBased: attention might win
    store_e = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    store_e.put("topic", "old_focus", focus="ai")
    for _ in range(10):
        store_e.access("topic")
    store_e.advance(focus="ai")
    store_e.advance(focus="weather")
    store_e.put("topic", "new_focus", focus="weather")
    
    r = store_e.get("topic")
    pred = r.fact.value if r else None
    results.append(Result("EventBased", "FocusAttention", pred == "new_focus", pred, "new_focus"))
    print(f"  EventBased: {pred}")
    
    # Hybrid with focus_weight=0.5
    store_h = HybridStore(time_half_life_hours=24, message_half_life=50,
                          time_weight=0.25, message_weight=0.25, focus_weight=0.5)
    store_h.put("topic", "old_focus", focus="ai")
    for _ in range(10):
        store_h.access("topic")
    store_h.advance(focus="ai")
    store_h.advance(focus="weather")
    store_h.put("topic", "new_focus", focus="weather")
    
    r = store_h.get("topic")
    pred = r.fact.value if r else None
    results.append(Result("Hybrid_50fw", "FocusAttention", pred == "new_focus", pred, "new_focus"))
    print(f"  Hybrid (focus=0.5): {pred}")
    
    # Hybrid with focus_weight=0.8 - focus dominates
    store_h2 = HybridStore(time_half_life_hours=24, message_half_life=50,
                           time_weight=0.1, message_weight=0.1, focus_weight=0.8)
    store_h2.put("topic", "old_focus", focus="ai")
    for _ in range(10):
        store_h2.access("topic")
    store_h2.advance(focus="ai")
    store_h2.advance(focus="weather")
    store_h2.put("topic", "new_focus", focus="weather")
    
    r = store_h2.get("topic")
    pred = r.fact.value if r else None
    results.append(Result("Hybrid_80fw", "FocusAttention", pred == "new_focus", pred, "new_focus"))
    print(f"  Hybrid (focus=0.8): {pred}")
    
    return results


def test_mixed_validity():
    """
    SCENARIO: Some facts have time validity, some have message validity.
    """
    print("\n=== MIXED VALIDITY ===")
    print("Scenario: One fact expires by time, other by message count")
    
    results = []
    
    # Hybrid handles both
    store_h = HybridStore(time_half_life_hours=24, message_half_life=10,
                          time_weight=0.5, message_weight=0.5)
    
    # Expired by time (48hr ago)
    store_h.put("fact", "time_expired", valid_from_time=base_time - timedelta(hours=48))
    
    # Valid by time, expired by message (20 messages ago, half_life=10 = 0.25 decay)
    # But wait - we need to add it BEFORE advancing
    store_h.put("fact", "msg_expired", focus="topic")
    for _ in range(20):
        store_h.advance(focus="topic")
    
    # Valid fact
    store_h.put("fact", "valid", focus="topic")
    
    # Check what's valid
    print("  All facts:")
    for r in store_h.get_all("fact", base_time):
        print(f"    {r.fact.value}: time_decay={r.time_decay:.2f}, msg_decay={r.message_decay:.2f}")
    
    r = store_h.get("fact", base_time)
    pred = r.fact.value if r else None
    # Should return 'valid' as it's both time-valid and message-valid
    results.append(Result("Hybrid", "MixedValidity", pred == "valid", pred, "valid"))
    print(f"  Hybrid: {pred}")
    
    return results


def test_equal_signals():
    """
    SCENARIO: All signals equal - tiebreaker should be recency (last added).
    """
    print("\n=== EQUAL SIGNALS (TIEBREAKER) ===")
    
    results = []
    
    # Hybrid with all facts having same characteristics
    store_h = HybridStore(time_half_life_hours=24, message_half_life=50,
                          time_weight=0.33, message_weight=0.33, focus_weight=0.34,
                          initial_focus="topic")
    
    store_h.put("fact", "first", focus="topic")
    store_h.advance(focus="topic")  # Add temporal difference
    store_h.put("fact", "second", focus="topic")
    
    r = store_h.get("fact")
    pred = r.fact.value if r else None
    # Now second has lower message_decay, should win
    results.append(Result("Hybrid", "EqualSignals", pred == "second", pred, "second"))
    print(f"  Hybrid: {pred}")
    
    return results


def test_attention_vs_temporal():
    """
    SCENARIO: Old fact with HUGE attention vs new fact with 0 attention.
    When should attention win?
    """
    print("\n=== ATTENTION vs TEMPORAL ===")
    print("Scenario: 100 accesses on old vs 0 on new")
    
    results = []
    
    # TimeBased
    store_t = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    store_t.put("fact", "old_hot", valid_from=base_time - timedelta(hours=1))
    for _ in range(100):
        store_t.access("fact")
    store_t.put("fact", "new_cold", valid_from=base_time)
    
    r = store_t.get("fact", base_time)
    pred = r.fact.value if r else None
    print(f"  TimeBased (0.9/0.1): {pred}")
    results.append(Result("TimeBased", "AttnVsTemp", pred == "new_cold", pred, "new_cold"))
    
    # EventBased - FIX: set initial focus
    store_e = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1, initial_focus="topic")
    store_e.put("fact", "old_hot", focus="topic")
    for _ in range(100):
        store_e.access("fact")
    store_e.put("fact", "new_cold", focus="topic")
    
    r = store_e.get("fact")
    pred = r.fact.value if r else None
    print(f"  EventBased (0.9/0.1): {pred}")
    results.append(Result("EventBased", "AttnVsTemp", pred == "new_cold", pred, "new_cold"))
    
    # Hybrid
    store_h = HybridStore(time_half_life_hours=24, message_half_life=50,
                          time_weight=0.4, message_weight=0.4, focus_weight=0.1, attention_weight=0.1,
                          initial_focus="topic")
    store_h.put("fact", "old_hot", valid_from_time=base_time - timedelta(hours=1), focus="topic")
    for _ in range(100):
        store_h.access("fact")
    store_h.put("fact", "new_cold", valid_from_time=base_time, focus="topic")
    
    r = store_h.get("fact", base_time)
    pred = r.fact.value if r else None
    print(f"  Hybrid: {pred}")
    results.append(Result("Hybrid", "AttnVsTemp", pred == "new_cold", pred, "new_cold"))
    
    return results


# ============================================================================
# RUN
# ============================================================================

print("="*70)
print("SOTA HARD COMPARISON - ADVERSARIAL TESTS")
print("="*70)

all_results = []
all_results.extend(test_time_vs_message_conflict())
all_results.extend(test_focus_with_time())
all_results.extend(test_mixed_validity())
all_results.extend(test_equal_signals())
all_results.extend(test_attention_vs_temporal())

# Summary
print("\n" + "="*70)
print("RESULTS")
print("="*70)

systems = sorted(set(r.system for r in all_results), 
                key=lambda s: -sum(1 for r in all_results if r.system == s and r.correct))

print(f"\n{'System':<20} {'Score':<10} {'Failed On':<30}")
print("-"*60)

for system in systems:
    sys_results = [r for r in all_results if r.system == system]
    correct = sum(1 for r in sys_results if r.correct)
    total = len(sys_results)
    pct = correct / total * 100 if total > 0 else 0
    
    failed = [r.task for r in sys_results if not r.correct]
    failed_str = ", ".join(failed) if failed else "-"
    
    print(f"{system:<20} {pct:>5.0f}%     {failed_str}")

print("\n" + "="*70)
