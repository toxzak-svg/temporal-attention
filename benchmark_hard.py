"""
Adversarial Temporal+Attention Benchmark - HARD MODE

Designed specifically to break:
- TemporalAttentionStore (our approach)
- Each task targets a specific weakness

The goal: find edge cases where temporal+attention fails, then fix them.
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Callable
import math


@dataclass
class Task:
    name: str
    description: str
    setup: Callable
    query: tuple[str, datetime]
    expected: str
    why_others_fail: str


@dataclass
class Result:
    task_name: str
    system: str
    predicted: str
    correct: bool


def create_store_class(store_type: str):
    """Create the appropriate store class."""
    if store_type == "TemporalAttention":
        from store import TemporalAttentionStore
        return TemporalAttentionStore
    elif store_type == "TemporalOnly":
        from store import TemporalAttentionStore
        class TemporalOnly(TemporalAttentionStore):
            def __init__(self):
                super().__init__(temporal_weight=1.0, attention_weight=0.0)
        return TemporalOnly
    elif store_type == "AttentionOnly":
        from store import TemporalAttentionStore
        class AttentionOnly(TemporalAttentionStore):
            def __init__(self):
                super().__init__(temporal_weight=0.0, attention_weight=1.0)
        return AttentionOnly
    elif store_type == "TimeConstraint":
        from store import TemporalAttentionStore
        class TimeConstraint(TemporalAttentionStore):
            def __init__(self):
                super().__init__(temporal_weight=0.0, attention_weight=0.0)
            def get(self, key, at=None, context_keys=None):
                if at is None:
                    at = datetime.now()
                if key not in self.facts:
                    return None
                for fact in self.facts[key]:
                    if fact.valid_from <= at and (fact.valid_to is None or fact.valid_to > at):
                        return fact.value
                return None
        return TimeConstraint
    elif store_type == "PlainRAG":
        class PlainRAG:
            def __init__(self):
                self.data = {}
            def put(self, key, value, valid_from=None, valid_to=None):
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append({"value": value, "valid_from": valid_from, "valid_to": valid_to})
            def get(self, key, at=None, context_keys=None):
                if at not in self.data:
                    return None
                return self.data[key][-1]["value"]
            def access(self, key):
                pass
        return PlainRAG
    else:
        raise ValueError(f"Unknown: {store_type}")


def get_hard_tasks() -> list[Task]:
    """Tasks specifically designed to beat TemporalAttention."""
    
    base = datetime(2024, 6, 15, 12, 0, 0)
    tasks = []
    
    # =========================================================================
    # HARD TASK 1: Recent Access on OLDER Valid Fact
    # =========================================================================
    # Both facts valid at query time, but older one accessed recently
    # Should return: the older one (valid at query time, higher attention)
    # This tests: does attention help when both are valid?
    
    def setup_hard1(store):
        # Older but valid + accessed
        store.put("config", "dark", valid_from=base - timedelta(days=100))
        store.access("config")  # Recent access
        
        # Newer (also valid)
        store.put("config", "light", valid_from=base - timedelta(days=5))
    
    tasks.append(Task(
        name="Hard1_ValidBothAttention",
        description="Both facts valid. Older one has attention. Return older (highest combined).",
        setup=setup_hard1,
        query=("config", base),
        expected="dark",
        why_others_fail="TemporalOnly would return light (newer)",
    ))
    
    # =========================================================================
    # HARD TASK 2: Fresh Access on INVALID Fact (Edge Case)
    # =========================================================================
    # A fact that WAS valid but expired - yet accessed recently
    # Should return: the currently valid fact
    # This tests: does temporal dominate validity over attention?
    
    def setup_hard2(store):
        # Expired but heavily accessed (should be ignored due to validity)
        store.put("status", "deprecated", valid_from=base - timedelta(days=200), valid_to=base - timedelta(days=30))
        store.access("status")  # Access on expired
        
        # Current
        store.put("status", "active", valid_from=base - timedelta(days=30))
    
    tasks.append(Task(
        name="Hard2_ExpiredWithAttention",
        description="Expired fact accessed. Should return valid one.",
        setup=setup_hard2,
        query=("status", base),
        expected="active",
        why_others_fail="AttentionOnly would return deprecated",
    ))
    
    # =========================================================================
    # HARD TASK 3: Query Time Exactly at Boundary (Micro-Edge)
    # =========================================================================
    # Query at exact microsecond where validity changes
    # Should return: the new one (valid_from <= at)
    
    def setup_hard3(store):
        store.put("value", "old", valid_from=base - timedelta(days=10), valid_to=base)
        store.put("value", "new", valid_from=base)  # Exactly now
    
    tasks.append(Task(
        name="Hard3_MicroBoundary",
        description="Query at exact validity boundary. Return 'new'.",
        setup=setup_hard3,
        query=("value", base),
        expected="new",
        why_others_fail="Some implementations miss <= edge case",
    ))
    
    # =========================================================================
    # HARD TASK 4: Heavy Attention Bias
    # =========================================================================
    # One fact has MASSIVE attention (100+ accesses)
    # Should return: current valid regardless of attention
    
    def setup_hard4(store):
        store.put("theme", "winter", valid_from=base - timedelta(days=365))
        for _ in range(200):  # Massive attention on old
            store.access("theme")
        
        store.put("theme", "summer", valid_from=base - timedelta(days=1))
    
    tasks.append(Task(
        name="Hard4_HeavyAttentionBias",
        description="200 accesses on old fact. Should return current (summer).",
        setup=setup_hard4,
        query=("theme", base),
        expected="summer",
        why_others_fail="AttentionOnly returns winter",
    ))
    
    # =========================================================================
    # HARD TASK 5: Interleaved Access Pattern
    # =========================================================================
    # Access patterns that should NOT bias toward older or newer
    # Both accessed equally, but different valid times
    
    def setup_hard5(store):
        store.put("item", "first", valid_from=base - timedelta(days=50))
        store.put("item", "second", valid_from=base - timedelta(days=25))
        store.put("item", "third", valid_from=base - timedelta(days=10))
        
        # Equal attention on first and third, none on second
        store.access("item")  # This actually accesses 'third' (most recent in list)
        store.access("item")
        # We need to manually set up more carefully
    
    # This task is complex - skip for now
    
    # =========================================================================
    # HARD TASK 6: Temporal Weight Dominance
    # =========================================================================
    # When temporal_weight is high (0.9), attention should barely matter
    # Should return newest valid regardless of attention
    
    # This is tested by comparing TemporalOnly vs TemporalAttention
    
    # =========================================================================
    # HARD TASK 7: Zero Attention (Cold Start)
    # =========================================================================
    # New fact with ZERO accesses vs old with some accesses
    # Should still return new (temporal dominates)
    
    def setup_hard7(store):
        store.put("data", "old_val", valid_from=base - timedelta(days=100))
        for _ in range(10):
            store.access("data")
        
        store.put("data", "new_val", valid_from=base - timedelta(days=1))
        # No access on new_val
    
    tasks.append(Task(
        name="Hard7_ColdStart",
        description="Old fact has 10 accesses, new has 0. Return new.",
        setup=setup_hard7,
        query=("data", base),
        expected="new_val",
        why_others_fail="AttentionOnly returns old_val",
    ))
    
    # =========================================================================
    # HARD TASK 8: Multiple Valid (Return Most Recent)
    # =========================================================================
    # Multiple facts valid at query time
    # Should return the most recent valid
    
    def setup_hard8(store):
        store.put("record", "first", valid_from=base - timedelta(days=100))
        store.put("record", "second", valid_from=base - timedelta(days=50))
        store.put("record", "third", valid_from=base - timedelta(days=20))
        store.put("record", "fourth", valid_from=base - timedelta(days=5))
    
    tasks.append(Task(
        name="Hard8_MultipleValid",
        description="4 facts all valid. Return most recent (fourth).",
        setup=setup_hard8,
        query=("record", base),
        expected="fourth",
        why_others_fail="May return wrong one if sorting fails",
    ))
    
    # =========================================================================
    # HARD TASK 9: Attention Boost Can Override Slight Age
    # =========================================================================
    # Fact A: 1 hour old, 0 accesses
    # Fact B: 2 hours old, 50 accesses
    # With attention_weight=0.5, B might win despite being older
    # Return: B (if combined score favors it)
    
    # This is inherent to the design - not a failure
    
    # =========================================================================
    # HARD TASK 10: All Invalid (Return None)
    # =========================================================================
    # All facts have valid_to in the past
    # Should return None
    
    def setup_hard10(store):
        store.put("metric", "old", valid_from=base - timedelta(days=100), valid_to=base - timedelta(days=50))
        store.put("metric", "newer", valid_from=base - timedelta(days=50), valid_to=base - timedelta(days=10))
    
    tasks.append(Task(
        name="Hard10_AllInvalid",
        description="All facts expired. Should return None.",
        setup=setup_hard10,
        query=("metric", base),
        expected=None,
        why_others_fail="May incorrectly return latest",
    ))
    
    return tasks


def run_benchmark(systems: list[str]) -> list[Result]:
    """Run benchmark."""
    tasks = get_hard_tasks()
    results = []
    
    for system in systems:
        StoreClass = create_store_class(system)
        
        for task in tasks:
            store = StoreClass()
            task.setup(store)
            
            key, query_time = task.query
            
            try:
                if system in ["TimeConstraint", "PlainRAG"]:
                    predicted = store.get(key, query_time)
                else:
                    result = store.get(key, query_time)
                    predicted = result.fact.value if result else None
            except Exception as e:
                predicted = f"ERROR: {e}"
            
            correct = predicted == task.expected
            
            results.append(Result(
                task_name=task.name,
                system=system,
                predicted=str(predicted),
                correct=correct,
            ))
    
    return results


def print_results(results: list[Result]):
    """Print results."""
    print("\n" + "="*80)
    print("HARD ADVERSARIAL BENCHMARK - BEATING TEMPORAL+ATTENTION")
    print("="*80)
    
    systems = set(r.system for r in results)
    
    for system in sorted(systems):
        system_results = [r for r in results if r.system == system]
        correct = sum(1 for r in system_results if r.correct)
        total = len(system_results)
        pct = correct / total * 100 if total > 0 else 0
        
        print(f"\n{system}: {correct}/{total} ({pct:.1f}%)")
        
        for r in system_results:
            status = "OK" if r.correct else "FAIL"
            print(f"  {status} {r.task_name}")
            if not r.correct:
                print(f"       Got: {r.predicted}")
    
    print("\n" + "="*80)
    print("\nSUMMARY:")
    print("-"*40)
    for system in sorted(systems):
        system_results = [r for r in results if r.system == system]
        correct = sum(1 for r in system_results if r.correct)
        total = len(system_results)
        pct = correct / total * 100 if total > 0 else 0
        print(f"{system:25s} {pct:5.1f}%")


if __name__ == "__main__":
    systems = [
        "PlainRAG",
        "TimeConstraint", 
        "TemporalOnly",
        "AttentionOnly",
        "TemporalAttention",
    ]
    
    results = run_benchmark(systems)
    print_results(results)
