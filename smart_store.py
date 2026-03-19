"""
EventBasedStore with IMPORTANCE and FACT TYPES

Key insight: Not all facts are equal.
- Names should never/rarely decay
- Preferences should decay slowly
- Context should decay normally
- Ephemeral facts should decay fast
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import math


# Fact types with different decay profiles
class FactType:
    NAME = "name"           # Never decays - user names are sacred
    PREFERENCE = "pref"     # Decays slowly - preferences matter
    CONTEXT = "context"    # Normal decay - general info
    EPHEMERAL = "ephemeral" # Fast decay - temporary stuff


@dataclass
class Fact:
    key: str
    value: Any
    valid_from_message: int = 0
    valid_to_message: Optional[int] = None
    focus_id: str = "default"
    fact_type: str = FactType.CONTEXT  # NEW: type affects decay
    importance: float = 0.5  # 0-1, higher = more important
    access_count: int = 0
    last_accessed_message: int = 0


@dataclass
class ScoredFact:
    fact: Fact
    message_decay: float
    focus_decay: float
    type_decay: float
    importance_boost: float
    attention_score: float
    combined_score: float


# How each fact type decays
FACT_TYPE_HALFLIFE = {
    FactType.NAME: float('inf'),      # Never decays
    FactType.PREFERENCE: 1000,         # Very slow
    FactType.CONTEXT: 50,             # Normal
    FactType.EPHEMERAL: 10,           # Fast
}


class SmartEventStore:
    """
    Event-based store with importance and fact types.
    
    Key insight: Names don't decay. Preferences decay slowly.
    Everything else decays normally.
    """
    
    def __init__(
        self,
        focus_decay_factor: float = 0.5,
        temporal_weight: float = 0.9,
        attention_weight: float = 0.1,
        initial_focus: str = "default",
    ):
        self.facts: dict[str, list[Fact]] = {}
        self.message_count = 0
        self.current_focus = initial_focus
        
        self.focus_decay_factor = focus_decay_factor
        self.temporal_weight = temporal_weight
        self.attention_weight = attention_weight
        
        # NEW: Track volatile facts - they auto-expire
        self.volatile_facts: dict[str, datetime] = {}  # key -> expires_at
    
    def advance(self, focus: Optional[str] = None):
        self.message_count += 1
        if focus and focus != self.current_focus:
            self._apply_focus_decay(focus)
            self.current_focus = focus
    
    def _apply_focus_decay(self, new_focus: str):
        old_focus = self.current_focus
        for facts in self.facts.values():
            for fact in facts:
                if fact.focus_id != new_focus and "_decayed" not in fact.focus_id:
                    fact.focus_id = f"{fact.focus_id}_was_{old_focus}"
    
    def put(
        self,
        key: str,
        value: Any,
        fact_type: str = FactType.CONTEXT,
        importance: float = 0.5,
        focus: Optional[str] = None,
    ):
        """Add a fact with type and importance."""
        
        # NEW: Check if this fact type auto-expires
        config = FACT_TYPE_CONFIG.get(fact_type, FACT_TYPE_CONFIG[FactType.CONTEXT])
        
        fact = Fact(
            key=key,
            value=value,
            valid_from_message=self.message_count,
            focus_id=focus or self.current_focus,
            fact_type=fact_type,
            importance=importance,
        )
        
        if key not in self.facts:
            self.facts[key] = []
        self.facts[key].append(fact)
        
        # NEW: Auto-expire volatile facts
        # If fact changed (different value), old one is invalid
        existing = [f for f in self.facts[key] if f.value != value]
        if existing and config["volatility"] > 0:
            # Old value is now invalid - mark it
            for f in existing:
                f.valid_to_message = self.message_count
    
    def get(self, key: str) -> Optional[ScoredFact]:
        if key not in self.facts:
            return None
        
        facts = self.facts[key]
        valid_facts = []
        
        for fact in facts:
            # Check validity
            if fact.valid_from_message > self.message_count:
                continue
            if fact.valid_to_message and fact.valid_to_message <= self.message_count:
                continue
            
            # Calculate scores
            msg_decay = self._message_decay(fact)
            focus_decay = self._focus_decay(fact)
            type_decay = self._type_decay(fact)
            importance = self._importance(fact)
            attention = self._attention(fact)
            
            # Combined: importance and type affect the base
            base = msg_decay * focus_decay * type_decay * importance
            
            combined = base * (1 + 0.001 * attention)
            
            valid_facts.append(ScoredFact(
                fact=fact,
                message_decay=msg_decay,
                focus_decay=focus_decay,
                type_decay=type_decay,
                importance_boost=importance,
                attention_score=attention,
                combined_score=combined,
            ))
        
        if not valid_facts:
            return None
        
        valid_facts.sort(key=lambda x: x.combined_score, reverse=True)
        return valid_facts[0]
    
    def _message_decay(self, fact: Fact) -> float:
        """Message count decay - affected by fact type."""
        half_life = FACT_TYPE_HALFLIFE.get(fact.fact_type, 50)
        
        if half_life == float('inf'):
            return 1.0  # Names never decay
        
        messages_since = self.message_count - fact.valid_from_message
        if messages_since < 0:
            messages_since = 0
        
        return math.exp(-0.693 * messages_since / half_life)
    
    def _type_decay(self, fact: Fact) -> float:
        """Type-based decay rate multiplier."""
        # Already handled in message_decay via halflife
        return 1.0
    
    def _importance(self, fact: Fact) -> float:
        """Importance multiplier."""
        # Maps 0-1 to 0.5-1.5
        return 0.5 + fact.importance
    
    def _focus_decay(self, fact: Fact) -> float:
        # Names are immune to focus decay - they're always relevant
        if fact.fact_type == FactType.NAME:
            return 1.0
        
        if fact.focus_id == self.current_focus:
            return 1.0
        elif "_was_" in fact.focus_id:
            return self.focus_decay_factor
        return self.focus_decay_factor
    
    def _attention(self, fact: Fact) -> float:
        if fact.access_count == 0:
            return 0.0
        count_score = math.log1p(fact.access_count) / 100
        return min(1.0, count_score)
    
    def access(self, key: str):
        if key in self.facts:
            latest = max(self.facts[key], key=lambda f: f.valid_from_message)
            latest.access_count += 1
            latest.last_accessed_message = self.message_count


# Demo
if __name__ == "__main__":
    store = SmartEventStore()
    
    print("="*60)
    print("SMART EVENT STORE - NAME PRESERVATION")
    print("="*60)
    
    # Test 1: Name never decays
    print("\n[Test 1] Name preservation")
    store.put("user_name", "Zach", fact_type=FactType.NAME, importance=1.0)
    
    # Advance 1000 messages
    for _ in range(1000):
        store.advance(focus="random")
    
    result = store.get("user_name")
    print(f"  After 1000 messages: {result.fact.value}")
    print(f"  Message decay: {result.message_decay:.4f}")
    print(f"  Importance: {result.importance_boost:.2f}")
    print(f"  Combined: {result.combined_score:.4f}")
    
    # Test 2: Preference decays slowly
    print("\n[Test 2] Preference decay")
    store.put("food", "pizza", fact_type=FactType.PREFERENCE, importance=0.8)
    
    for _ in range(100):
        store.advance(focus="other")
    
    result = store.get("food")
    print(f"  After 100 messages: {result.fact.value}")
    print(f"  Decay: {result.message_decay:.4f}")
    
    # Test 3: Normal fact decays
    print("\n[Test 3] Normal context decay")
    store.put("topic", "weather", fact_type=FactType.CONTEXT, importance=0.5)
    
    for _ in range(100):
        store.advance(focus="other")
    
    result = store.get("topic")
    print(f"  After 100 messages: {result.fact.value}")
    print(f"  Decay: {result.message_decay:.4f}")
    
    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("- Names: halflife=inf (never decay)")
    print("- Preferences: halflife=1000 (slow)")
    print("- Context: halflife=50 (normal)")
    print("- Ephemeral: halflife=10 (fast)")
    print("="*60)
