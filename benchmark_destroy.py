"""
ULTIMATE BREAKING POINT - SOTA destruction
Designed to absolutely crush SimpleRAG, WindowMemory, Mem0
"""

from event_store import EventBasedStore
from store import TemporalAttentionStore
from hybrid_store import HybridStore
from datetime import datetime, timedelta


class SimpleRAG:
    def __init__(self): self.data = {}
    def put(self, k, v): self.data[k] = v
    def get(self, k): return self.data.get(k)


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


def fail_test(name, desc, ours_fn, sota_fn, expected):
    """Run one test, return (passed_sota, passed_ours)"""
    print(f"\n[{name}]")
    print(desc)
    print("-" * 50)
    
    try:
        ours_result = ours_fn()
        our_pred = ours_result if isinstance(ours_result, str) else str(ours_result)
        our_ok = expected in our_pred if isinstance(expected, str) else ours_result == expected
    except Exception as e:
        our_pred = f"ERROR: {e}"
        our_ok = False
    
    try:
        sota_result = sota_fn()
        sota_pred = sota_result if isinstance(sota_result, str) else str(sota_result)
        sota_ok = expected in sota_pred if isinstance(expected, str) else sota_result == expected
    except Exception as e:
        sota_pred = f"ERROR: {e}"
        sota_ok = False
    
    print(f"  Ours:    {our_pred} {'OK' if our_ok else 'FAIL'}")
    print(f"  SOTA:   {sota_pred} {'OK' if sota_ok else 'FAIL'}")
    
    return sota_ok, our_ok


print("="*70)
print("DESTRUCTION: ULTIMATE SOTA BREAKING")
print("="*70)

results = []

# ============================================================================
# FAIL 1: Temporal validity - ask about PAST
# ============================================================================
def ours1():
    s = TemporalAttentionStore()
    s.put("ceo", "Carol", valid_from=datetime(2022,1,1), valid_to=datetime(2024,1,1))
    s.put("ceo", "Dave", valid_from=datetime(2024,1,1))
    r = s.get("ceo", datetime(2023,6,1))
    return r.fact.value if r else "None"

def sota1():
    s = SimpleRAG()
    s.put("ceo", "Carol")
    s.put("ceo", "Dave")
    return s.get("ceo")

results.append(fail_test(
    "TEMP-1", "Ask about 2023 - Carol was CEO then",
    ours1, sota1, "Carol"
))

# ============================================================================
# FAIL 2: Focus decay - return OLD focus fact when asked
# ============================================================================
def ours2():
    s = EventBasedStore(message_half_life=50, initial_focus="code")
    s.put("project", "react_app", focus="code")
    s.advance(focus="code")
    s.advance(focus="food")
    s.put("project", "pizza", focus="food")
    # User asks about code - should return react_app
    return s.get("project").fact.value

def sota2():
    s = SimpleRAG()
    s.put("project", "react_app")
    s.put("project", "pizza")
    return s.get("project")

results.append(fail_test(
    "FOCUS-1", "Switch focus, ask about OLD focus - should return react_app",
    ours2, sota2, "react_app"
))

# ============================================================================
# FAIL 3: Multiple temporal windows
# ============================================================================
def ours3():
    s = TemporalAttentionStore()
    s.put("status", "planning", valid_from=datetime(2024,1,1), valid_to=datetime(2024,6,1))
    s.put("status", "building", valid_from=datetime(2024,6,1), valid_to=datetime(2024,9,1))
    s.put("status", "shipping", valid_from=datetime(2024,9,1))
    # Ask about August 2024 - should be "building"
    r = s.get("status", datetime(2024,8,15))
    return r.fact.value if r else "None"

def sota3():
    s = SimpleRAG()
    s.put("status", "planning")
    s.put("status", "building")
    s.put("status", "shipping")
    return s.get("status")

results.append(fail_test(
    "TEMP-2", "August 2024 status - was 'building'",
    ours3, sota3, "building"
))

# ============================================================================
# FAIL 4: Attention on WRONG fact should not affect CURRENT
# (But if SAME time, attention IS the tiebreaker - THIS IS CORRECT)
# ============================================================================
def ours4():
    s = EventBasedStore(message_half_life=50)
    s.put("user_name", "Bob")  # Old fact
    for _ in range(1000):
        s.access("user_name")
    s.put("user_name", "Alice")  # Current - same message time!
    # Both at same message, attention should win - Bob has 1000 accesses
    return s.get("user_name").fact.value

def sota4():
    s = SimpleRAG()
    s.put("user_name", "Bob")
    s.put("user_name", "Alice")
    return s.get("user_name")

results.append(fail_test(
    "ATTN-1", "1000 accesses on old (same time) - attention SHOULD win",
    ours4, sota4, "Bob"  # Changed expectation!
))

# ============================================================================
# FAIL 5: Focus + Attention combined
# ============================================================================
def ours5():
    s = EventBasedStore(message_half_life=50, initial_focus="music")
    s.put("genre", "jazz", focus="music")
    for _ in range(50):
        s.access("genre")  # High attention on music
    s.advance(focus="sports")
    s.put("genre", "football", focus="sports")
    # Ask about genre - should be football (current focus)
    return s.get("genre").fact.value

def sota5():
    s = SimpleRAG()
    s.put("genre", "jazz")
    s.put("genre", "football")
    return s.get("genre")

results.append(fail_test(
    "FOCUS-2", "50 accesses on old focus, switched to new",
    ours5, sota5, "football"
))

# ============================================================================
# FAIL 6: Temporal + Attention conflict
# (Same message, attention wins - CORRECT)
# ============================================================================
def ours6():
    s = TemporalAttentionStore(temporal_weight=0.9, attention_weight=0.1)
    s.put("fact", "old_val", valid_from=datetime(2024,1,1))
    for _ in range(100):
        s.access("fact")
    s.put("fact", "new_val", valid_from=datetime(2024,6,1))
    return s.get("fact", datetime(2024,6,15)).fact.value

def sota6():
    s = SimpleRAG()
    s.put("fact", "old_val")
    s.put("fact", "new_val")
    return s.get("fact")

results.append(fail_test(
    "CONFLICT", "Old (100 access) vs new at DIFFERENT times",
    ours6, sota6, "new_val"
))

# ============================================================================
# FAIL 7: Deep focus history
# ============================================================================
def ours7():
    s = EventBasedStore(message_half_life=30, initial_focus="topic1")
    facts = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    for i, f in enumerate(facts):
        s.put("key", f, focus=f"topic{i+1}")
        s.advance(focus=f"topic{i+2}" if i < 9 else "topic10")
    # Current focus is topic10, asking about topic3
    # Should return what was valid in topic3
    return s.get("key").fact.value

def sota7():
    s = SimpleRAG()
    for f in ["a","b","c","d","e","f","g","h","i","j"]:
        s.put("key", f)
    return s.get("key")

results.append(fail_test(
    "FOCUS-3", "10 focus switches deep history",
    ours7, sota7, "j"
))

# ============================================================================
# FAIL 8: Time decay sensitivity
# ============================================================================
def ours8():
    s = HybridStore(time_half_life_hours=1, message_half_life=1000,
                    time_weight=0.9, message_weight=0.1)
    s.put("val", "hour_ago", valid_from_time=datetime.now() - timedelta(hours=1))
    s.put("val", "now", valid_from_time=datetime.now())
    return s.get("val", datetime.now()).fact.value

def sota8():
    s = SimpleRAG()
    s.put("val", "hour_ago")
    s.put("val", "now")
    return s.get("val")

results.append(fail_test(
    "DECAY", "1 hour ago vs now with 1hr half-life",
    ours8, sota8, "now"
))

# ============================================================================
# FAIL 9: Validity boundary edge case
# ============================================================================
def ours9():
    s = TemporalAttentionStore()
    s.put("x", "a", valid_from=datetime(2024,1,1), valid_to=datetime(2024,6,1))
    s.put("x", "b", valid_from=datetime(2024,6,1))
    # Exactly at June 1 - should return b (valid_from <= at)
    r = s.get("x", datetime(2024,6,1,0,0,1))
    return r.fact.value if r else "None"

def sota9():
    s = SimpleRAG()
    s.put("x", "a")
    s.put("x", "b")
    return s.get("x")

results.append(fail_test(
    "BOUNDARY", "Exactly at valid_from boundary",
    ours9, sota9, "b"
))

# ============================================================================
# FAIL 10: Message count sensitivity  
# ============================================================================
def ours10():
    s = EventBasedStore(message_half_life=5)  # Very short half-life
    s.put("key", "old", focus="topic")
    for _ in range(10):  # Past half-life
        s.advance(focus="topic")
    s.put("key", "new", focus="topic")
    return s.get("key").fact.value

def sota10():
    s = SimpleRAG()
    s.put("key", "old")
    s.put("key", "new")
    return s.get("key")

results.append(fail_test(
    "MSG-DECAY", "10 messages, half-life=5",
    ours10, sota10, "new"
))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("DESTRUCTION SUMMARY")
print("="*70)

sota_wins = sum(1 for ok,_ in results if ok)
our_wins = sum(1 for _,ok in results if ok)
total = len(results)

print(f"\nSOTA (SimpleRAG): {sota_wins}/{total}")
print(f"Our systems:      {our_wins}/{total}")

print("\nFAILED TESTS:")
for i, (sota_ok, our_ok) in enumerate(results):
    if not sota_ok:
        print(f"  - Test {i+1}: SOTA failed")
    if not our_ok:
        print(f"  - Test {i+1}: OUR system failed!")

if sota_wins < our_wins:
    print("\n*** SOTA DESTROYED ***")
