"""
EventBasedStore: Decay based on events, not wall-clock time.

Instead of time ticking away, decay is a product of:
1. Message count (new interactions)
2. Focus shift (topic changed)

This fits conversation/memory better than time-based decay.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import math


@dataclass
class Fact:
    key: str
    value: Any
    valid_from_message: int = 0  # Message count when fact was added
    valid_to_message: Optional[int] = None  # When fact became invalid
    focus_id: str = "default"  # Topic/focus group
    access_count: int = 0
    last_accessed_message: int = 0


@dataclass
class ScoredFact:
    fact: Fact
    message_decay: float  # Based on message count
    focus_decay: float  # Based on focus shift
    attention_score: float  # Based on access patterns
    combined_score: float


class EventBasedStore:
    """
    Decay based on events, not time.
    
    Key idea: In conversation, what matters isn't wall-clock time -
    it's how many messages have passed and whether the topic shifted.
    
    Decay formula:
    - message_decay: e^(-0.693 * messages_since_added / half_life_messages)
    - focus_decay: 1.0 if same focus, 0.5 if shifted
    - total_decay = message_decay * focus_decay
    
    Attention still applies within decay.
    """
    
    def __init__(
        self,
        message_half_life: int = 50,  # Messages until decay to 0.5
        focus_decay_factor: float = 0.5,  # Multiplier when focus shifts
        temporal_weight: float = 0.9,
        attention_weight: float = 0.1,
        initial_focus: str = "default",
    ):
        self.facts: dict[str, list[Fact]] = {}
        self.message_count = 0
        self.current_focus = initial_focus  # Start with a focus
        
        self.message_half_life = message_half_life
        self.focus_decay_factor = focus_decay_factor
        self.temporal_weight = temporal_weight
        self.attention_weight = attention_weight
    
    def advance(self, focus: Optional[str] = None):
        """Advance the message count. Call after each message."""
        self.message_count += 1
        if focus:
            if focus != self.current_focus:
                # Focus shifted - apply decay
                self._apply_focus_decay(focus)
                self.current_focus = focus
    
    def _apply_focus_decay(self, new_focus: str):
        """When focus shifts, decay facts that are NOT in the new focus."""
        old_focus = self.current_focus
        for facts in self.facts.values():
            for fact in facts:
                # Only decay facts that are NOT in the new focus
                if fact.focus_id != new_focus and "_decayed" not in fact.focus_id:
                    # Mark as decayed - will get focus_decay_factor
                    fact.focus_id = f"{fact.focus_id}_was_{old_focus}"
    
    def put(
        self,
        key: str,
        value: Any,
        valid_from_message: Optional[int] = None,
        valid_to_message: Optional[int] = None,
        focus: Optional[str] = None,
    ):
        """Add a fact at current message count."""
        if valid_from_message is None:
            valid_from_message = self.message_count
        
        fact = Fact(
            key=key,
            value=value,
            valid_from_message=valid_from_message,
            valid_to_message=valid_to_message,
            focus_id=focus or self.current_focus,
        )
        
        if key not in self.facts:
            self.facts[key] = []
        self.facts[key].append(fact)
    
    def get(self, key: str, at_message: Optional[int] = None) -> Optional[ScoredFact]:
        """Get best fact at given message count."""
        if at_message is None:
            at_message = self.message_count
        
        if key not in self.facts:
            return None
        
        facts = self.facts[key]
        valid_facts = []
        
        for fact in facts:
            # Check validity
            if fact.valid_from_message > at_message:
                continue  # Not yet valid
            if fact.valid_to_message is not None and fact.valid_to_message <= at_message:
                continue  # No longer valid
            
            # Calculate scores
            msg_decay = self._message_decay(fact, at_message)
            focus_decay = self._focus_decay(fact)
            attention = self._attention(fact, at_message)
            
            temporal_score = msg_decay * focus_decay
            
            # FIXED: Small recency bonus to prefer newer facts when tied
            recency_bonus = 0.0001 * (self.message_count - fact.valid_from_message)
            
            # FIXED: Attention has NEGLIGIBLE effect (just breaks true ties)
            combined = temporal_score * (1 + 0.001 * attention) + recency_bonus
            
            valid_facts.append(ScoredFact(
                fact=fact,
                message_decay=msg_decay,
                focus_decay=focus_decay,
                attention_score=attention,
                combined_score=combined,
            ))
        
        if not valid_facts:
            return None
        
        valid_facts.sort(key=lambda x: x.combined_score, reverse=True)
        return valid_facts[0]
    
    def _message_decay(self, fact: Fact, at_message: int) -> float:
        """Decay based on message count, not wall-clock time."""
        messages_since = at_message - fact.valid_from_message
        if messages_since < 0:
            messages_since = 0
        decay = math.exp(-0.693 * messages_since / self.message_half_life)
        return min(1.0, max(0.0, decay))
    
    def _focus_decay(self, fact: Fact) -> float:
        """Decay based on focus shift."""
        if fact.focus_id == self.current_focus:
            return 1.0
        elif "_decayed" in fact.focus_id:
            return self.focus_decay_factor
        else:
            # Same focus_id but not current - partially decayed
            return self.focus_decay_factor
    
    def _attention(self, fact: Fact, at_message: int) -> float:
        """Score based on access count - very heavily damped."""
        if fact.access_count == 0:
            return 0.0
        # Use log with large denominator - even 100 accesses should barely matter
        # This ensures temporal always dominates unless temporal scores are nearly equal
        count_score = math.log1p(fact.access_count) / 30  # log(101)/30 = 0.15 max
        
        # Recency of access matters too
        messages_since_access = at_message - fact.last_accessed_message
        if messages_since_access < 0:
            messages_since_access = 0
        access_decay = math.exp(-0.693 * messages_since_access / (self.message_half_life / 2))
        
        return min(1.0, count_score * access_decay)
    
    def access(self, key: str):
        """Record access at current message count."""
        if key in self.facts:
            latest = max(self.facts[key], key=lambda f: f.valid_from_message)
            latest.access_count += 1
            latest.last_accessed_message = self.message_count
    
    def get_all(self, key: str, at_message: Optional[int] = None) -> list[ScoredFact]:
        """Get all valid facts for a key."""
        if at_message is None:
            at_message = self.message_count
        
        if key not in self.facts:
            return []
        
        facts = self.facts[key]
        results = []
        
        for fact in facts:
            if fact.valid_from_message > at_message:
                continue
            if fact.valid_to_message is not None and fact.valid_to_message <= at_message:
                continue
            
            msg_decay = self._message_decay(fact, at_message)
            focus_decay = self._focus_decay(fact)
            attention = self._attention(fact, at_message)
            
            temporal = msg_decay * focus_decay
            combined = temporal * (1 + self.attention_weight * attention)
            
            results.append(ScoredFact(
                fact=fact,
                message_decay=msg_decay,
                focus_decay=focus_decay,
                attention_score=attention,
                combined_score=combined,
            ))
        
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results
    
    def __repr__(self):
        return f"EventBasedStore(msgs={self.message_count}, focus={self.current_focus}, keys={len(self.facts)})"


# Demo
if __name__ == "__main__":
    store = EventBasedStore(
        message_half_life=50,
        focus_decay_factor=0.5,
        temporal_weight=0.9,
        attention_weight=0.1,
    )
    
    print("="*60)
    print("EVENT-BASED DECAY DEMO")
    print("="*60)
    
    # Message 1: Set up initial fact
    store.put("topic", "attention")
    print(f"\n[Msg {store.message_count}] Added: topic=attention")
    
    # Message 2-5: Advance, access the fact
    for _ in range(4):
        store.advance()
    store.access("topic")
    store.access("topic")
    print(f"[Msg {store.message_count}] Accessed topic 2x")
    
    # Message 6: Ask about it
    result = store.get("topic")
    print(f"[Msg {store.message_count}] Query topic: {result.fact.value}")
    print(f"  Scores: msg_decay={result.message_decay:.2f}, focus={result.focus_decay:.2f}, attention={result.attention_score:.2f}")
    
    # Message 7-20: Lots of messages but same topic
    for _ in range(14):
        store.advance()
    print(f"\n[Msg {store.message_count}] Advanced 14 messages (same topic)")
    
    result = store.get("topic")
    print(f"Query topic: {result.fact.value}")
    print(f"  Scores: msg_decay={result.message_decay:.2f}")
    # Note: message_decay drops!
    
    # Message 21: Topic shifts!
    store.advance(focus="weather")
    print(f"\n[Msg {store.message_count}] FOCUS SHIFTED to: weather")
    
    # Add new fact under new focus
    store.put("topic", "weather_info")
    result = store.get("topic")
    print(f"Query topic: {result.fact.value}")
    print(f"  Scores: msg_decay={result.message_decay:.2f}, focus={result.focus_decay:.2f}")
    # The attention fact should be heavily decayed!
    
    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("- Decay based on MESSAGE COUNT, not wall-clock time")
    print("- Focus shift applies extra decay")
    print("- Conversation naturally 'forgets' as it advances")
    print("="*60)
