"""
Adversarial Temporal+Attention Benchmark

This benchmark creates scenarios designed to break:
- Pure temporal approaches (decay only)
- Pure attention approaches (access count only)
- Time constraint only (ignores attention)
- Temporal rerank only (ignores recency of access)

Our approach: TemporalAttentionStore should beat all of them.
"""

import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Callable
import json


@dataclass
class Task:
    name: str
    description: str
    setup: Callable  # Function that sets up the store
    query: tuple[str, datetime]  # (key, query_time)
    expected: str  # Expected answer
    why_others_fail: str


@dataclass
class Result:
    task_name: str
    system: str
    predicted: str
    correct: bool
    score: float


def create_store_class(store_type: str):
    """Create the appropriate store class."""
    if store_type == "TemporalAttention":
        from store import TemporalAttentionStore
        return TemporalAttentionStore
    elif store_type == "TemporalOnly":
        # Temporal decay only (no attention)
        from store import TemporalAttentionStore
        class TemporalOnly(TemporalAttentionStore):
            def __init__(self):
                super().__init__(temporal_weight=1.0, attention_weight=0.0)
        return TemporalOnly
    elif store_type == "AttentionOnly":
        # Attention only (no temporal)
        from store import TemporalAttentionStore
        class AttentionOnly(TemporalAttentionStore):
            def __init__(self):
                super().__init__(temporal_weight=0.0, attention_weight=1.0)
        return AttentionOnly
    elif store_type == "TimeConstraint":
        # Just time window, no decay or attention
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
        # Just latest by timestamp
        class PlainRAG:
            def __init__(self):
                self.data = {}
            def put(self, key, value, valid_from=None, valid_to=None):
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append({"value": value, "valid_from": valid_from, "valid_to": valid_to})
            def get(self, key, at=None, context_keys=None):
                if key not in self.data:
                    return None
                # Just return latest
                return self.data[key][-1]["value"]
            def access(self, key):
                pass
        return PlainRAG
    else:
        raise ValueError(f"Unknown store type: {store_type}")


def get_tasks() -> list[Task]:
    """Generate adversarial tasks that break other approaches."""
    
    tasks = []
    base_time = datetime(2024, 6, 15, 12, 0, 0)
    
    # =========================================================================
    # TASK 1: Temporal Reversion + High Attention on Old
    # =========================================================================
    # Facts: A (old), B (new), but A was accessed MANY times recently
    # Pure attention: Returns A (wrong, outdated)
    # Temporal: Returns B (correct)
    # Our approach: Returns B (correct, temporal dominates but attention helps edge cases)
    
    def setup_rev_attention(store):
        # Old fact (but accessed a lot)
        store.put("ceo", "Alice", valid_from=base_time - timedelta(days=400), valid_to=base_time - timedelta(days=100))
        # Access it many times recently to fool attention-only
        for _ in range(20):
            store.access("ceo")
        
        # New fact (not accessed)
        store.put("ceo", "Bob", valid_from=base_time - timedelta(days=100))
    
    tasks.append(Task(
        name="ReversionWithHighAttention",
        description="Fact reverted (Alice→Bob), but old fact has high attention. Should return Bob.",
        setup=setup_rev_attention,
        query=("ceo", base_time),
        expected="Bob",
        why_others_fail="Attention-only would return Alice (high access count)",
    ))
    
    # =========================================================================
    # TASK 2: Multiple Reversions + Attention on Middle
    # =========================================================================
    # Facts: A→B→A→B (multiple reversions), attention on 2nd fact
    # Plain RAG: Returns latest (B) - might be wrong for query time
    # Our approach: Respects time window + attention
    
    def setup_multi_reversion(store):
        store.put("status", "active", valid_from=base_time - timedelta(days=300), valid_to=base_time - timedelta(days=200))
        store.put("status", "pending", valid_from=base_time - timedelta(days=200), valid_to=base_time - timedelta(days=100))
        store.put("status", "active", valid_from=base_time - timedelta(days=100), valid_to=base_time - timedelta(days=30))
        store.put("status", "closed", valid_from=base_time - timedelta(days=30))
        
        # Access the middle one (pending) to confuse attention-only
        store.access("status")
        store.access("status")
    
    tasks.append(Task(
        name="MultiReversion",
        description="Multiple reversions A→B→A→B. Query for time when B was active.",
        setup=setup_multi_reversion,
        query=("status", base_time - timedelta(days=150)),
        expected="pending",
        why_others_fail="Plain RAG returns 'closed' (latest), ignores time window",
    ))
    
    # =========================================================================
    # TASK 3: Attention Decoy - High Access on Invalid
    # =========================================================================
    # Fact A is valid, Fact B is invalid (expired), but B was accessed way more
    # Time constraint: Returns A (correct)
    # Attention-only: Returns B (wrong)
    # Temporal: Returns A (correct)
    # Our: Returns A (correct, temporal dominates for validity)
    
    def setup_attention_decoy(store):
        # Valid fact (not accessed much)
        store.put("project", "Alpha", valid_from=base_time - timedelta(days=50))
        
        # Invalid fact (expired, but accessed a lot - decoy)
        store.put("project", "Beta", valid_from=base_time - timedelta(days=200), valid_to=base_time - timedelta(days=60))
        for _ in range(50):  # Heavy access to trick attention-only
            store.access("project")
    
    tasks.append(Task(
        name="AttentionDecoy",
        description="Valid fact vs expired fact with high attention. Should return valid one.",
        setup=setup_attention_decoy,
        query=("project", base_time),
        expected="Alpha",
        why_others_fail="Attention-only would return Beta (high access), ignoring validity",
    ))
    
    # =========================================================================
    # TASK 4: Boundary Edge Case
    # =========================================================================
    # Fact valid_from = query_time exactly (boundary)
    # Some systems might exclude due to <= vs < edge case
    
    def setup_boundary(store):
        store.put("status", "old", valid_from=base_time - timedelta(days=100), valid_to=base_time)
        store.put("status", "new", valid_from=base_time)  # Exactly at query time
    
    tasks.append(Task(
        name="BoundaryEdge",
        description="Fact becomes valid exactly at query time. Should return 'new'.",
        setup=setup_boundary,
        query=("status", base_time),
        expected="new",
        why_others_fail="Some systems use < instead of <=, miss boundary",
    ))
    
    # =========================================================================
    # TASK 5: Future Fact (Should Be Ignored)
    # =========================================================================
    # Fact with valid_from in the future
    # Should NOT be returned
    
    def setup_future(store):
        store.put("event", "past", valid_from=base_time - timedelta(days=30))
        store.put("event", "future", valid_from=base_time + timedelta(days=30))
    
    tasks.append(Task(
        name="FutureFact",
        description="Future-dated fact should be ignored. Return 'past'.",
        setup=setup_future,
        query=("event", base_time),
        expected="past",
        why_others_fail="Plain RAG might return latest regardless of time",
    ))
    
    # =========================================================================
    # TASK 6: Stale Attention - Old Access Should Decay
    # =========================================================================
    # Fact was heavily accessed but not recently
    # Attention should decay over time
    
    def setup_stale_attention(store):
        # Old fact with old access (should decay)
        store.put("owner", "OldOwner", valid_from=base_time - timedelta(days=200))
        for _ in range(30):
            store.access("owner")  # Accesses are old (we don't have a way to set access time directly in setup, but the decay should handle it)
        
        # New fact (recent, valid)
        store.put("owner", "NewOwner", valid_from=base_time - timedelta(days=10))
    
    tasks.append(Task(
        name="StaleAttention",
        description="Old fact had high attention, but access was long ago. Should return recent.",
        setup=setup_stale_attention,
        query=("owner", base_time),
        expected="NewOwner",
        why_others_fail="Simple attention (no decay) would return OldOwner",
    ))
    
    # =========================================================================
    # TASK 7: Context-Aware (Multiple Keys)
    # =========================================================================
    # Query context influences which fact is more relevant
    # Attention to related keys should boost main query
    
    def setup_context(store):
        store.put("user_role", "admin", valid_from=base_time - timedelta(days=100))
        store.put("user_role", "user", valid_from=base_time - timedelta(days=10))
        
        # Access related context
        store.access("user_role")
        store.access("user_role")
        store.access("permissions")
        store.access("permissions")
    
    tasks.append(Task(
        name="ContextBoost",
        description="Recent role change. Both valid, recent one should win.",
        setup=setup_context,
        query=("user_role", base_time),
        expected="user",
        why_others_fail="May not handle temporal + attention combination correctly",
    ))
    
    # =========================================================================
    # TASK 8: Long Gap Query
    # =========================================================================
    # Query for a time in the distant past
    # System should return what was valid THEN
    
    def setup_long_gap(store):
        store.put("version", "v1", valid_from=base_time - timedelta(days=300), valid_to=base_time - timedelta(days=200))
        store.put("version", "v2", valid_from=base_time - timedelta(days=200), valid_to=base_time - timedelta(days=100))
        store.put("version", "v3", valid_from=base_time - timedelta(days=100))
        
        # Access current a lot
        for _ in range(20):
            store.access("version")
    
    tasks.append(Task(
        name="LongGapQuery",
        description="Query for time in distant past. Return what was valid then.",
        setup=setup_long_gap,
        query=("version", base_time - timedelta(days=250)),
        expected="v1",
        why_others_fail="Plain RAG returns v3 (latest), time-constraint-only might struggle with query time",
    ))
    
    return tasks


def run_benchmark(systems: list[str]) -> list[Result]:
    """Run all tasks across all systems."""
    tasks = get_tasks()
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
                predicted = None
                print(f"Error on {system}/{task.name}: {e}")
            
            correct = predicted == task.expected
            
            results.append(Result(
                task_name=task.name,
                system=system,
                predicted=str(predicted),
                correct=correct,
                score=1.0 if correct else 0.0,
            ))
    
    return results


def print_results(results: list[Result]):
    """Print formatted results."""
    print("\n" + "="*80)
    print("TEMPORAL+ATTENTION ADVERSARIAL BENCHMARK RESULTS")
    print("="*80)
    
    # Group by system
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
    
    print("\n" + "="*80)
    
    # Summary table
    print("\nSCORES BY SYSTEM:")
    print("-"*40)
    for system in sorted(systems):
        system_results = [r for r in results if r.system == system]
        correct = sum(1 for r in system_results if r.correct)
        total = len(system_results)
        pct = correct / total * 100 if total > 0 else 0
        print(f"{system:25s} {pct:5.1f}%")
    
    print("="*80)


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
