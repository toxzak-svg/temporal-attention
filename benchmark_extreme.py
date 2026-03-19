"""
EXTREME ADVERSARIAL BENCHMARK - Break TemporalAttention

Push until TemporalAttention starts failing.
"""

from store import TemporalAttentionStore
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    setup: callable
    query_time: datetime
    expected_key: str
    expected: str


base = datetime(2024, 6, 15, 12, 0, 0)

# ============================================================================
# EXTREME TASKS - Designed to break TemporalAttention
# ============================================================================

def test_equal_temporal_diff_attention():
    """Both facts have SAME temporal score but DIFFERENT attention.
    Attention should be the tiebreaker.
    But with 0.9/0.1 weights, temporal dominates so much attention can't overcome.
    """
    # Two facts with nearly identical temporal age
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    
    # Fact A: 10 hours old, 100 accesses
    store.put("key", "A", valid_from=base - timedelta(hours=10))
    for _ in range(100):
        store.access("key")
    
    # Fact B: 11 hours old (slightly older), 0 accesses
    store.put("key", "B", valid_from=base - timedelta(hours=11))
    # No access
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "B"


def test_temporal_attention_tradeoff():
    """Fact A: 1 hour old, 0 accesses
    Fact B: 5 hours old, 50 accesses
    Temporal diff: 5 hours
    Attention diff: 50 accesses
    
    With 0.9/0.1, temporal wins. But maybe attention should win?
    """
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    
    store.put("key", "recent_cold", valid_from=base - timedelta(hours=1))
    store.put("key", "older_hot", valid_from=base - timedelta(hours=5))
    for _ in range(50):
        store.access("key")
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "recent_cold"


def test_micro_temporal_diff():
    """Both valid but only 1 minute apart in valid_from.
    Massive attention difference should win but temporal might dominate.
    """
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    
    store.put("key", "older", valid_from=base - timedelta(minutes=60))
    for _ in range(100):
        store.access("key")
    
    store.put("key", "newer", valid_from=base - timedelta(minutes=59))  # Only 1 min newer
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "newer"


def test_attention_boost_insufficient():
    """Even with 100 accesses, attention (0.1) can't overcome temporal difference."""
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    
    # Fact A: 24 hours old (half-life = 24), temporal score ~0.5
    store.put("key", "old_100access", valid_from=base - timedelta(hours=24))
    for _ in range(100):
        store.access("key")
    
    # Fact B: 1 hour old, temporal score ~0.97
    store.put("key", "recent_0access", valid_from=base - timedelta(hours=1))
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "recent_0access"


def test_validity_edge_case():
    """Fact valid at exact query time boundary."""
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    
    store.put("key", "expiring", valid_from=base - timedelta(hours=1), valid_to=base)
    store.put("key", "new", valid_from=base)  # Exactly now
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "new"


def test_many_valid_facts():
    """10 facts all valid. With attention, which wins?"""
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    
    for i in range(10):
        store.put("key", f"fact_{i}", valid_from=base - timedelta(days=i*10))
        if i == 5:  # Middle one gets attention
            for _ in range(20):
                store.access("key")
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "fact_0"  # Most recent


def test_zero_temporal_score():
    """Fact just became valid (temporal score near 1) vs old with attention."""
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    
    # Just became valid (temporal score = 1.0)
    store.put("key", "brand_new", valid_from=base)
    
    # Old with high attention
    store.put("key", "old_hot", valid_from=base - timedelta(hours=1))
    for _ in range(100):
        store.access("key")
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "brand_new"


def test_decay_sensitivity():
    """Test different half-life settings."""
    # With longer half-life, temporal decays slower
    store = TemporalAttentionStore(
        temporal_weight=0.9, attention_weight=0.1,
        decay_half_life_hours=168  # 1 week
    )
    
    store.put("key", "old", valid_from=base - timedelta(days=30))
    for _ in range(50):
        store.access("key")
    
    store.put("key", "new", valid_from=base - timedelta(days=1))
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "new"


def test_attention_decay_fast():
    """Attention decays in 6 hours. What if access was 1 hour ago?"""
    store = TemporalAttentionStore(
        temporal_weight=0.9, attention_weight=0.1,
        attention_decay_hours=6
    )
    
    # Fact 1: 1 hour old, 10 accesses (1 hour ago)
    store.put("key", "recent_access", valid_from=base - timedelta(hours=2))
    for _ in range(10):
        store.access("key")
    
    # Fact 2: 30 min old, 0 accesses
    store.put("key", "new_cold", valid_from=base - timedelta(minutes=30))
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "new_cold"


def test_equal_everything():
    """Both facts have identical temporal AND attention.
    Should return most recently added."""
    store = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    
    store.put("key", "first", valid_from=base - timedelta(hours=5))
    store.put("key", "second", valid_from=base - timedelta(hours=5))  # Same time
    
    result = store.get("key", base)
    return store, result.fact.value if result else None, "second"  # Last added


# Run all extreme tests
print("="*70)
print("EXTREME ADVERSARIAL - BREAKING TEMPORAL+ATTENTION")
print("="*70)

tests = [
    ("Equal Temporal, Diff Attention", test_equal_temporal_diff_attention),
    ("Temporal-Attention Tradeoff", test_temporal_attention_tradeoff),
    ("Micro Temporal Diff", test_micro_temporal_diff),
    ("Attention Boost Insufficient", test_attention_boost_insufficient),
    ("Validity Edge Case", test_validity_edge_case),
    ("Many Valid Facts", test_many_valid_facts),
    ("Zero Temporal Score", test_zero_temporal_score),
    ("Decay Sensitivity", test_decay_sensitivity),
    ("Attention Decay Fast", test_attention_decay_fast),
    ("Equal Everything", test_equal_everything),
]

failures = 0
for name, test_fn in tests:
    store, predicted, expected = test_fn()
    status = "OK" if predicted == expected else "FAIL"
    if status == "FAIL":
        failures += 1
    print(f"\n[{status}] {name}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {predicted}")
    
    # Show scores for failed tests
    if status == "FAIL":
        key = list(store.facts.keys())[0]
        all_results = store.get_all(key, base)
        print(f"  All valid facts:")
        for r in all_results[:5]:
            print(f"    {r.fact.value}: temporal={r.temporal_score:.3f}, attention={r.attention_score:.3f}, combined={r.combined_score:.3f}")

print("\n" + "="*70)
print(f"RESULT: {len(tests) - failures}/{len(tests)} passed")
print("="*70)

# Now test each system on the hardest tasks
print("\n\n" + "="*70)
print("SYSTEM COMPARISON ON HARDEST TASKS")
print("="*70)

# Test: Temporal-Attention Tradeoff is the hardest
print("\n--- Test: Temporal-Attention Tradeoff ---")
print("Scenario: Fact A (1hr old, 0 accesses) vs Fact B (5hrs old, 50 accesses)")
print("Expected: recent_cold (more recent)")

for name, tw, aw in [("TemporalOnly", 1.0, 0.0), ("AttentionOnly", 0.0, 1.0), ("TemporalAttention", 0.9, 0.1)]:
    store = TemporalAttentionStore(temporal_weight=tw, attention_weight=aw)
    store.put("key", "recent_cold", valid_from=base - timedelta(hours=1))
    store.put("key", "older_hot", valid_from=base - timedelta(hours=5))
    for _ in range(50):
        store.access("key")
    result = store.get("key", base)
    pred = result.fact.value if result else None
    print(f"  {name}: {pred}")

print("\n--- Test: Equal Temporal, Diff Attention ---")
print("Scenario: Fact A (10hr old, 100 accesses) vs Fact B (11hr old, 0 accesses)")
print("Expected: B (slightly more recent)")

for name, tw, aw in [("TemporalOnly", 1.0, 0.0), ("AttentionOnly", 0.0, 1.0), ("TemporalAttention", 0.9, 0.1)]:
    store = TemporalAttentionStore(temporal_weight=tw, attention_weight=aw)
    store.put("key", "A", valid_from=base - timedelta(hours=10))
    for _ in range(100):
        store.access("key")
    store.put("key", "B", valid_from=base - timedelta(hours=11))
    result = store.get("key", base)
    pred = result.fact.value if result else None
    print(f"  {name}: {pred}")
