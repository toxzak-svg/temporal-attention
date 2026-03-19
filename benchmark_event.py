"""
Event-Based Store Benchmark - HARD CASES (Fixed)

Compare event-based vs time-based vs other methods.
"""

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


base_time = datetime(2024, 6, 15, 12, 0, 0)

# ============================================================================
# HARD TEST CASES - FIXED
# ============================================================================

def test_brand_new_vs_old_attention():
    """Brand new fact vs old with high attention - need temporal difference."""
    
    # Event-based
    store_e = EventBasedStore(message_half_life=50, focus_decay_factor=0.5, temporal_weight=0.9, attention_weight=0.1)
    store_e.put("key", "old_hot", focus="ai")
    for _ in range(100):
        store_e.access("key")
    
    # Advance messages to create temporal difference
    for _ in range(10):
        store_e.advance(focus="ai")
    
    # Brand new
    store_e.put("key", "brand_new", focus="ai")
    
    result = store_e.get("key")
    pred_e = result.fact.value if result else None
    
    expected = "brand_new"
    return [Result("EventBased", "BrandNewVsOldAttention", pred_e == expected, pred_e)]


def test_recent_vs_older_attention():
    """Recent (10 msgs ago, 0 access) vs older (20 msgs ago, 50 access)."""
    
    store_e = EventBasedStore(message_half_life=50, focus_decay_factor=0.5, temporal_weight=0.9, attention_weight=0.1)
    
    # Older: 20 messages ago
    store_e.put("key", "older_hot", focus="ai")
    for _ in range(50):
        store_e.access("key")
    for _ in range(10):
        store_e.advance(focus="ai")
    
    # Recent: 0 messages ago  
    store_e.put("key", "recent_cold", focus="ai")
    
    result = store_e.get("key")
    pred_e = result.fact.value if result else None
    
    expected = "recent_cold"
    return [Result("EventBased", "RecentVsOlderAttention", pred_e == expected, pred_e)]


def test_focus_shift():
    """Old fact with high attention vs new under different focus."""
    
    store_e = EventBasedStore(message_half_life=50, focus_decay_factor=0.5, temporal_weight=0.9, attention_weight=0.1)
    
    # Old fact with high attention under 'ai' focus
    store_e.put("topic", "attention", focus="ai")
    for _ in range(50):
        store_e.access("topic")
    
    # Advance many messages
    for _ in range(30):
        store_e.advance(focus="ai")
    
    # Shift focus to 'weather' and add new fact
    store_e.advance(focus="weather")
    store_e.put("topic", "weather", focus="weather")
    
    result = store_e.get("topic")
    pred_e = result.fact.value if result else None
    
    expected = "weather"  # New focus should win
    return [Result("EventBased", "FocusShift", pred_e == expected, pred_e)]


def test_many_messages_same_focus():
    """Many messages, same focus - temporal decay should apply."""
    
    store_e = EventBasedStore(message_half_life=20, focus_decay_factor=0.5, temporal_weight=0.9, attention_weight=0.1)
    
    store_e.put("key", "old", focus="ai")
    # Advance 30 messages (past half-life of 20)
    for _ in range(30):
        store_e.advance(focus="ai")
    
    store_e.put("key", "new", focus="ai")
    
    result = store_e.get("key")
    pred_e = result.fact.value if result else None
    
    expected = "new"
    return [Result("EventBased", "ManyMessagesSameFocus", pred_e == expected, pred_e)]


def test_same_focus_different_attention():
    """Same focus, DIFFERENT messages - attention shouldn't overcome temporal."""
    # This tests that attention only breaks TRUE ties (same message)
    # If we add ANY temporal difference, newer should win
    
    store_e = EventBasedStore(message_half_life=50, focus_decay_factor=0.5, temporal_weight=0.9, attention_weight=0.1)
    
    store_e.put("key", "fact_a", focus="ai")
    store_e.access("key")
    store_e.access("key")
    store_e.access("key")
    
    # Add slight temporal difference - newer should win
    store_e.advance(focus="ai")
    
    store_e.put("key", "fact_b", focus="ai")
    
    result = store_e.get("key")
    pred_e = result.fact.value if result else None
    
    expected = "fact_b"  # With temporal diff, newer wins (correct behavior!)
    return [Result("EventBased", "SameFocusDiffAttention", pred_e == expected, pred_e)]


def test_all_expired():
    """All facts expired by message count."""
    
    store_e = EventBasedStore(message_half_life=10, focus_decay_factor=0.5)
    
    store_e.put("key", "fact1", valid_from_message=0, valid_to_message=5)
    store_e.put("key", "fact2", valid_from_message=5, valid_to_message=10)
    
    # Advance past all facts
    for _ in range(15):
        store_e.advance()
    
    result = store_e.get("key")
    pred_e = result.fact.value if result else None
    
    expected = None
    return [Result("EventBased", "AllExpired", pred_e == expected, str(pred_e))]


def test_boundary():
    """Fact becomes valid exactly at query time."""
    
    store_e = EventBasedStore(message_half_life=50, focus_decay_factor=0.5)
    
    store_e.put("key", "old", valid_from_message=0, valid_to_message=10)
    store_e.put("key", "new", valid_from_message=10)
    
    # Advance to message 10
    for _ in range(10):
        store_e.advance()
    
    result = store_e.get("key")
    pred_e = result.fact.value if result else None
    
    expected = "new"
    return [Result("EventBased", "Boundary", pred_e == expected, pred_e)]


def test_time_based_comparison():
    """Compare event-based with time-based on same scenario."""
    
    results = []
    
    # Same scenario: old fact (1 hour ago, 100 accesses) vs brand new
    # Time-based
    store_t = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    store_t.put("key", "old", valid_from=base_time - timedelta(hours=1))
    for _ in range(100):
        store_t.access("key")
    store_t.put("key", "brand_new", valid_from=base_time)
    
    result_t = store_t.get("key", base_time)
    pred_t = result_t.fact.value if result_t else None
    
    results.append(Result("TimeBased_90_10", "BrandNewVsOldAttention", 
                          pred_t == "brand_new", pred_t))
    
    # Event-based
    store_e = EventBasedStore(message_half_life=50, temporal_weight=0.9, attention_weight=0.1)
    store_e.put("key", "old", focus="ai")
    for _ in range(100):
        store_e.access("key")
    store_e.advance(focus="ai")  # Create temporal diff
    store_e.put("key", "brand_new", focus="ai")
    
    result_e = store_e.get("key")
    pred_e = result_e.fact.value if result_e else None
    
    results.append(Result("EventBased", "BrandNewVsOldAttention", 
                          pred_e == "brand_new", pred_e))
    
    return results


# ============================================================================
# RUN ALL
# ============================================================================

print("="*70)
print("EVENT-BASED STORE - HARD BENCHMARK (FIXED)")
print("="*70)

all_results = []

tests = [
    test_brand_new_vs_old_attention,
    test_recent_vs_older_attention,
    test_focus_shift,
    test_many_messages_same_focus,
    test_same_focus_different_attention,
    test_all_expired,
    test_boundary,
]

for test_fn in tests:
    results = test_fn()
    all_results.extend(results)

# Add time-based comparison
all_results.extend(test_time_based_comparison())

# Print results grouped by system
systems = set(r.system for r in all_results)

print()
for system in sorted(systems):
    system_results = [r for r in all_results if r.system == system]
    correct = sum(1 for r in system_results if r.correct)
    total = len(system_results)
    pct = correct / total * 100 if total > 0 else 0
    
    print(f"{system}: {correct}/{total} ({pct:.0f}%)")
    for r in system_results:
        status = "OK" if r.correct else "FAIL"
        print(f"  {status} {r.task}: got={r.got}")

print("\n" + "="*70)
