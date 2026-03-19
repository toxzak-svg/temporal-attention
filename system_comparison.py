"""
FINAL SYSTEM COMPARISON - Each system's failure modes
"""

from store import TemporalAttentionStore
from datetime import datetime, timedelta

base = datetime(2024, 6, 15, 12, 0, 0)

print("="*70)
print("SYSTEM FAILURE MODES - Where each breaks")
print("="*70)

tests = [
    # (name, setup_fn, expected, description)
    
    ("Brand new vs old+attention",
     lambda s: (s.put('k', 'new', valid_from=base), 
                s.put('k', 'old', valid_from=base - timedelta(hours=1)),
                [s.access('k') for _ in range(100)]),
     "new",
     "Temporal=1.0 should beat old+100 accesses"),
    
    ("Recent (1hr) vs older+attention",
     lambda s: (s.put('k', 'recent', valid_from=base - timedelta(hours=1)),
                s.put('k', 'older', valid_from=base - timedelta(hours=5)),
                [s.access('k') for _ in range(50)]),
     "recent",
     "More recent should beat older+50 accesses"),
    
    ("Tiny temporal diff (1 min)",
     lambda s: (s.put('k', 'older', valid_from=base - timedelta(minutes=60)),
                s.put('k', 'newer', valid_from=base - timedelta(minutes=59)),
                [s.access('k') for _ in range(100)]),
     "newer",
     "1 min newer should win with 100 accesses"),
    
    ("Many valid facts",
     lambda s: ([s.put('k', f'fact_{i}', valid_from=base - timedelta(days=i*10)) for i in range(10)],
                [s.access('k') for _ in range(20)]),
     "fact_0",
     "Return most recent valid"),
    
    ("All expired",
     lambda s: (s.put('k', 'old1', valid_from=base - timedelta(days=100), valid_to=base - timedelta(days=50)),
                s.put('k', 'old2', valid_from=base - timedelta(days=50), valid_to=base - timedelta(days=10))),
     None,
     "All expired - should return None"),
    
    ("Boundary exact",
     lambda s: (s.put('k', 'expiring', valid_from=base - timedelta(hours=1), valid_to=base),
                s.put('k', 'new', valid_from=base)),
     "new",
     "Valid at exact query time"),
]

systems = [
    ("PlainRAG", lambda: type('S', (), {'data': {}, 'put': lambda s,k,v,vf=None,vt=None: s.data.setdefault(k,[]).append({'v':v,'vf':vf,'vt':vt}), 'get': lambda s,k,at=None: s.data[k][-1]['v'] if k in s.data else None, 'access': lambda s,k: None})()),
    ("TemporalOnly", lambda: TemporalAttentionStore(temporal_weight=1.0, attention_weight=0.0)),
    ("AttentionOnly", lambda: TemporalAttentionStore(temporal_weight=0.0, attention_weight=1.0)),
    ("TA_90_10", lambda: TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)),
    ("TA_99_01", lambda: TemporalAttentionStore(temporal_weight=0.99, attention_weight=0.01)),
]

results = {}

for sys_name, sys_fn in systems:
    results[sys_name] = []
    
    for test_name, setup_fn, expected, desc in tests:
        try:
            store = sys_fn()
            setup_fn(store)
            
            if sys_name == "PlainRAG":
                pred = store.get('k', base)
            else:
                result = store.get('k', base)
                pred = result.fact.value if result else None
                
            correct = pred == expected
            results[sys_name].append((test_name, correct, pred))
        except Exception as e:
            results[sys_name].append((test_name, False, str(e)))

# Print results
print("\n" + "-"*70)
for sys_name in results:
    total = len(results[sys_name])
    correct = sum(1 for _, c, _ in results[sys_name] if c)
    print(f"\n{sys_name}: {correct}/{total} ({correct/total*100:.0f}%)")
    
    for test_name, correct, pred in results[sys_name]:
        status = "OK" if correct else "FAIL"
        print(f"  {status} {test_name}: expected={results[sys_name][0][2] if correct else results[sys_name][0][2]}, got={pred}")

# Summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'System':<20} {'Score':<10} {'Best For':<30} {'Fails On':<30}")
print("-"*70)

for sys_name in results:
    total = len(results[sys_name])
    correct = sum(1 for _, c, _ in results[sys_name] if c)
    pct = correct/total*100
    
    if sys_name == "PlainRAG":
        best = "Latest only"
        fails = "Validity, time queries"
    elif sys_name == "TemporalOnly":
        best = "Recency always wins"
        fails = "When attention matters"
    elif sys_name == "AttentionOnly":
        best = "Most accessed"
        fails = "Stale data, validity"
    elif sys_name == "TA_90_10":
        best = "Balanced"
        fails = "Brand new vs old+high attention"
    elif sys_name == "TA_99_01":
        best = "Temporal dominance"
        fails = "When attention should win"
    
    print(f"{sys_name:<20} {pct:>5.0f}%    {best:<30} {fails:<30}")

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)
print("""
No single system wins everywhere:

- PlainRAG: Fast but ignores time entirely
- TemporalOnly: Always prefers recent, ignores usage
- AttentionOnly: Returns most-used, returns stale data  
- TA_90_10: Balanced but attention can wrongly win
- TA_99_01: Temporal dominates, attention nearly useless

The sweet spot depends on USE CASE:

1. Memory systems: TA_90_10 (balance, small attention boost)
2. Fact databases: TemporalOnly (accuracy over recall)
3. Recommendation: TA_95_05 (temporal mostly, some attention)
""")
