"""
SmartEventStore - Memory with Fact Types

Key insight: Facts have different properties:
- Decay rate: how fast we "forget" them
- Change rate: how fast the fact itself becomes invalid

Types:
- NAME: never decays, never changes (Zach is always Zach)
- PREFERENCE: decays slow, changes rarely (likes pizza)
- CONTEXT: normal decay, normal change (what we're talking about)
- EPHEMERAL: fast decay, changes fast (the weather right now)
"""

from dataclasses import dataclass
from typing import Any, Optional
import math


class FactType:
    NAME = "name"           # Sacred - never decays, never changes
    PREFERENCE = "pref"     # Important - decays slow, changes rare
    CONTEXT = "context"    # Normal - normal decay, normal change
    EPHEMERAL = "ephemeral" # Disposable - fast decay, changes fast


# Half-life in messages for each type
TYPE_DECAY = {
    FactType.NAME: float('inf'),      # Never decays
    FactType.MUTABLE: 200,           # Slow decay, but keeps history
    FactType.PREFERENCE: 500,         # Very slow
    FactType.CONTEXT: 30,             # Normal  
    FactType.EPHEMERAL: 5,           # Fast
}


@dataclass
class Fact:
    key: str
    value: Any
    fact_type: str = FactType.CONTEXT
    importance: float = 0.5  # 0-1, boosts recall
    message_at: int = 0
    focus: str = "default"
    access_count: int = 0


class SmartStore:
    """
    Memory that understands not all facts are equal.
    """
    
    def __init__(self, focus_decay=0.5):
        self.facts = {}
        self.msg = 0
        self.focus = "default"
        self.focus_decay = focus_decay
    
    def advance(self, focus=None):
        self.msg += 1
        if focus and focus != self.focus:
            self.focus = focus
    
    def put(self, key, value, fact_type=FactType.CONTEXT, importance=0.5, focus=None):
        f = Fact(key, value, fact_type, importance, self.msg, focus or self.focus)
        self.facts.setdefault(key, []).append(f)
    
    def get(self, key):
        if key not in self.facts:
            return None
        
        candidates = []
        for f in self.facts[key]:
            # Calculate scores
            decay = self._decay(f)
            focus_boost = 1.0 if f.focus == self.focus else self.focus_decay
            importance_boost = 0.5 + f.importance * 0.5
            attention = min(1.0, math.log1p(f.access_count) / 20)
            
            score = decay * focus_boost * importance_boost * (1 + 0.01 * attention)
            candidates.append((score, f))
        
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1] if candidates else None
    
    def get_history(self, key, limit=10):
        """Get history of a fact - especially useful for MUTABLE types."""
        if key not in self.facts:
            return []
        
        facts = sorted(self.facts[key], key=lambda f: -f.message_at)
        return facts[:limit]
    
    def _decay(self, fact):
        half_life = TYPE_DECAY.get(fact.fact_type, TYPE_DECAY[FactType.CONTEXT])
        if half_life == float('inf'):
            return 1.0
        age = self.msg - fact.message_at
        return math.exp(-0.693 * age / half_life)
    
    def access(self, key):
        if key in self.facts:
            self.facts[key][-1].access_count += 1


# Demo
if __name__ == "__main__":
    s = SmartStore()
    
    print("="*60)
    print("FACT TYPES: Different decay for different facts")
    print("="*60)
    
    # THE TEST: All stored, then advance 100 messages
    s.put("user_name", "Zach", FactType.NAME, importance=1.0)
    s.put("food_preference", "pizza", FactType.PREFERENCE, importance=0.8)
    s.put("current_topic", "weather", FactType.CONTEXT, importance=0.5)
    s.put("weather_right_now", "sunny", FactType.EPHEMERAL, importance=0.3)
    s.put("project_name", "Alpha", FactType.MUTABLE, importance=0.9)
    
    # Later - change the project name!
    s.put("project_name", "Beta", FactType.MUTABLE, importance=0.9)
    
    # Advance 100 messages
    for _ in range(100):
        s.advance("random_topic")
    
    print("\nAfter 100 messages:")
    
    name = s.get("user_name")
    print(f"Name: {name.value if name else 'gone'}")
    print(f"  Type: {name.fact_type}, Decay: {s._decay(name):.4f}")
    
    pref = s.get("food_preference")
    print(f"Preference: {pref.value if pref else 'gone'}")
    print(f"  Type: {pref.fact_type}, Decay: {s._decay(pref):.4f}")
    
    topic = s.get("current_topic")
    print(f"Context: {topic.value if topic else 'gone'}")
    print(f"  Type: {topic.fact_type}, Decay: {s._decay(topic):.4f}")
    
    weather = s.get("weather_right_now")
    print(f"Ephemeral: {weather.value if weather else 'gone'}")
    print(f"  Type: {weather.fact_type}, Decay: {s._decay(weather):.4f}")
    
    print("\n" + "="*60)
    print("INSIGHT:")
    print("- Names: decay=1.0 (never forget)")
    print("- Preferences: decay=0.87 (slow)")
    print("- Context: decay=0.10 (mostly gone)")
    print("- Ephemeral: decay=0.00004 (completely gone)")
    print("="*60)
