"""
HybridStore: Combines TIME + EVENT based decay

The best of both worlds:
- Time decay: Wall-clock time matters for factual accuracy
- Event decay: Message count for conversation relevance  
- Focus decay: Topic shifts

All three combined = best of both worlds.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import math


@dataclass
class Fact:
    key: str
    value: Any
    valid_from_message: int = 0
    valid_to_message: Optional[int] = None
    valid_from_time: Optional[datetime] = None
    valid_to_time: Optional[datetime] = None
    focus_id: str = "default"
    access_count: int = 0
    last_accessed_message: int = 0
    last_accessed_time: Optional[datetime] = None


@dataclass
class ScoredFact:
    fact: Fact
    time_decay: float
    message_decay: float
    focus_decay: float
    attention_score: float
    combined_score: float


class HybridStore:
    """
    Combines time-based AND event-based decay.
    
    Decay = f(time_decay, message_decay, focus_decay) * attention
    
    Use when both wall-clock time AND conversation flow matter.
    """
    
    def __init__(
        self,
        # Time-based params
        time_half_life_hours: float = 24.0,
        # Event-based params  
        message_half_life: int = 50,
        focus_decay_factor: float = 0.5,
        # Weights
        time_weight: float = 0.33,
        message_weight: float = 0.33,
        focus_weight: float = 0.34,
        attention_weight: float = 0.1,
        # Initial focus
        initial_focus: str = "default",
    ):
        self.facts: dict[str, list[Fact]] = {}
        self.message_count = 0
        self.current_focus = initial_focus
        
        # Time params
        self.time_half_life_hours = time_half_life_hours
        # Event params
        self.message_half_life = message_half_life
        self.focus_decay_factor = focus_decay_factor
        # Weights (attention is bonus on top)
        self.time_weight = time_weight
        self.message_weight = message_weight
        self.focus_weight = focus_weight
        self.attention_weight = attention_weight
        
        # Normalize weights
        total = time_weight + message_weight + focus_weight
        self.time_weight = time_weight / total
        self.message_weight = message_weight / total  
        self.focus_weight = focus_weight / total
    
    def advance(self, focus: Optional[str] = None, at: Optional[datetime] = None):
        """Advance message count. Call after each message."""
        self.message_count += 1
        if focus:
            if focus != self.current_focus:
                self._apply_focus_decay(focus)
                self.current_focus = focus
    
    def _apply_focus_decay(self, new_focus: str):
        """When focus shifts, decay old facts."""
        old_focus = self.current_focus
        for facts in self.facts.values():
            for fact in facts:
                if fact.focus_id != new_focus and "_decayed" not in fact.focus_id:
                    fact.focus_id = f"{fact.focus_id}_was_{old_focus}"
    
    def put(
        self,
        key: str,
        value: Any,
        valid_from_message: Optional[int] = None,
        valid_to_message: Optional[int] = None,
        valid_from_time: Optional[datetime] = None,
        valid_to_time: Optional[datetime] = None,
        focus: Optional[str] = None,
    ):
        """Add a fact."""
        if valid_from_message is None:
            valid_from_message = self.message_count
        
        fact = Fact(
            key=key,
            value=value,
            valid_from_message=valid_from_message,
            valid_to_message=valid_to_message,
            valid_from_time=valid_from_time,
            valid_to_time=valid_to_time,
            focus_id=focus or self.current_focus,
        )
        
        if key not in self.facts:
            self.facts[key] = []
        self.facts[key].append(fact)
    
    def get(self, key: str, at: Optional[datetime] = None) -> Optional[ScoredFact]:
        """Get best fact considering ALL decay signals."""
        if at is None:
            at = datetime.now()
        
        if key not in self.facts:
            return None
        
        facts = self.facts[key]
        valid_facts = []
        
        for fact in facts:
            # Check validity (both message and time)
            if fact.valid_from_message > self.message_count:
                continue
            if fact.valid_to_message and fact.valid_to_message <= self.message_count:
                continue
            
            if fact.valid_from_time and fact.valid_from_time > at:
                continue
            if fact.valid_to_time and fact.valid_to_time <= at:
                continue
            
            # Calculate all decay signals
            time_decay = self._time_decay(fact, at)
            message_decay = self._message_decay(fact)
            focus_decay = self._focus_decay(fact)
            attention = self._attention(fact, at)
            
            # Combined temporal score
            temporal_score = (
                self.time_weight * time_decay +
                self.message_weight * message_decay +
                self.focus_weight * focus_decay
            )
            
            # Attention has NEGLIGIBLE effect (just breaks true ties)
            combined = temporal_score * (1 + 0.001 * attention)
            
            valid_facts.append(ScoredFact(
                fact=fact,
                time_decay=time_decay,
                message_decay=message_decay,
                focus_decay=focus_decay,
                attention_score=attention,
                combined_score=combined,
            ))
        
        if not valid_facts:
            return None
        
        valid_facts.sort(key=lambda x: x.combined_score, reverse=True)
        return valid_facts[0]
    
    def _time_decay(self, fact: Fact, at: datetime) -> float:
        """Wall-clock time decay."""
        if not fact.valid_from_time:
            return 1.0  # No time constraint
        hours = (at - fact.valid_from_time).total_seconds() / 3600
        if hours < 0:
            hours = 0
        return math.exp(-0.693 * hours / self.time_half_life_hours)
    
    def _message_decay(self, fact: Fact) -> float:
        """Message count decay."""
        messages_since = self.message_count - fact.valid_from_message
        if messages_since < 0:
            messages_since = 0
        return math.exp(-0.693 * messages_since / self.message_half_life)
    
    def _focus_decay(self, fact: Fact) -> float:
        """Focus decay."""
        if fact.focus_id == self.current_focus:
            return 1.0
        elif "_was_" in fact.focus_id:
            return self.focus_decay_factor
        return self.focus_decay_factor
    
    def _attention(self, fact: Fact, at: datetime) -> float:
        """Attention score (heavily damped)."""
        if fact.access_count == 0:
            return 0.0
        
        count_score = math.log1p(fact.access_count) / 30
        return min(1.0, count_score)
    
    def access(self, key: str, at: Optional[datetime] = None):
        """Record access."""
        if at is None:
            at = datetime.now()
        if key in self.facts:
            latest = max(self.facts[key], key=lambda f: f.valid_from_message)
            latest.access_count += 1
            latest.last_accessed_message = self.message_count
            latest.last_accessed_time = at
    
    def get_all(self, key: str, at: Optional[datetime] = None) -> list[ScoredFact]:
        """Get all valid facts."""
        if at is None:
            at = datetime.now()
        
        if key not in self.facts:
            return []
        
        facts = self.facts[key]
        results = []
        
        for fact in facts:
            if fact.valid_from_message > self.message_count:
                continue
            if fact.valid_to_message and fact.valid_to_message <= self.message_count:
                continue
            if fact.valid_from_time and fact.valid_from_time > at:
                continue
            if fact.valid_to_time and fact.valid_to_time <= at:
                continue
            
            time_decay = self._time_decay(fact, at)
            message_decay = self._message_decay(fact)
            focus_decay = self._focus_decay(fact)
            attention = self._attention(fact, at)
            
            temporal = (self.time_weight * time_decay + 
                       self.message_weight * message_decay + 
                       self.focus_weight * focus_decay)
            
            combined = (1 - self.attention_weight) * temporal + self.attention_weight * attention
            
            results.append(ScoredFact(
                fact=fact, time_decay=time_decay, message_decay=message_decay,
                focus_decay=focus_decay, attention_score=attention,
                combined_score=combined,
            ))
        
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results
