"""
TemporalAttentionStore: Combines temporal validity + attention signals for retrieval.

The insight: Pure temporal decay treats all old facts equally.
            Pure attention ignores time (can be stale).
            Combined: both signals weighted for smarter routing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import math


@dataclass
class Fact:
    key: str
    value: Any
    valid_from: datetime
    valid_to: Optional[datetime] = None  # None = currently valid
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ScoredFact:
    fact: Fact
    temporal_score: float  # How recent (0-1)
    attention_score: float  # How much accessed recently (0-1)
    combined_score: float  # Weighted combination
    is_valid: bool  # Respects time window


class TemporalAttentionStore:
    """
    Combines temporal validity + attention signals for retrieval.
    
    Key features:
    - Time window enforcement (valid_from, valid_to) - HARD FILTER
    - Access tracking (attention signal)
    - Decay over time (temporal signal)
    - Combined scoring for ranking within valid facts
    
    Key insight: Validity is a HARD constraint. A fact that isn't valid
    should NEVER be returned, regardless of attention.
    """
    
    def __init__(
        self,
        decay_half_life_hours: float = 24.0,
        attention_decay_hours: float = 6.0,
        temporal_weight: float = 0.5,
        attention_weight: float = 0.5,
        validity_threshold: float = 0.0,  # Below this, fact is considered invalid
    ):
        self.facts: dict[str, list[Fact]] = {}
        self.decay_half_life_hours = decay_half_life_hours
        self.attention_decay_hours = attention_decay_hours
        self.temporal_weight = temporal_weight
        self.attention_weight = attention_weight
        self.validity_threshold = validity_threshold
    
    def put(
        self,
        key: str,
        value: Any,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
    ):
        """Add a fact with optional time window."""
        if valid_from is None:
            valid_from = datetime.now()
        
        if key not in self.facts:
            self.facts[key] = []
        
        fact = Fact(
            key=key,
            value=value,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        self.facts[key].append(fact)
    
    def get(
        self,
        key: str,
        at: Optional[datetime] = None,
        context_keys: Optional[list[str]] = None,
    ) -> Optional[ScoredFact]:
        """Get the best fact, considering both temporal and attention signals."""
        if at is None:
            at = datetime.now()
        
        if key not in self.facts:
            return None
        
        facts = self.facts[key]
        scored_facts = []
        
        for fact in facts:
            # Check temporal validity
            is_valid = (
                fact.valid_from <= at and
                (fact.valid_to is None or fact.valid_to > at)
            )
            
            if not is_valid:
                continue
            
            # Calculate temporal score (decay based on how old)
            temporal_score = self._temporal_decay(fact, at)
            
            # Calculate attention score (based on access patterns)
            attention_score = self._attention_decay(fact, at)
            
            # Combined score
            combined = (
                self.temporal_weight * temporal_score +
                self.attention_weight * attention_score
            )
            
            scored_facts.append(ScoredFact(
                fact=fact,
                temporal_score=temporal_score,
                attention_score=attention_score,
                combined_score=combined,
                is_valid=True,
            ))
        
        if not scored_facts:
            return None
        
        # Return highest combined score
        scored_facts.sort(key=lambda x: x.combined_score, reverse=True)
        return scored_facts[0]
    
    def access(self, key: str):
        """Record that a fact was accessed (boosts attention)."""
        if key in self.facts:
            # Access most recent fact for this key
            latest = max(self.facts[key], key=lambda f: f.created_at)
            latest.access_count += 1
            latest.last_accessed = datetime.now()
    
    def _temporal_decay(self, fact: Fact, at: datetime) -> float:
        """Calculate temporal score with exponential decay.
        
        Uses valid_from (when fact became true) not created_at.
        """
        # Use valid_from for temporal decay - that's when the fact became true
        effective_time = fact.valid_from if fact.valid_from else fact.created_at
        age_hours = (at - effective_time).total_seconds() / 3600
        if age_hours < 0:
            age_hours = 0  # Future facts get max decay
        decay = math.exp(-0.693 * age_hours / self.decay_half_life_hours)
        return min(1.0, max(0.0, decay))
    
    def _attention_decay(self, fact: Fact, at: datetime) -> float:
        """Calculate attention score based on recency of access."""
        if fact.last_accessed is None:
            return 0.0
        
        hours_since_access = (at - fact.last_accessed).total_seconds() / 3600
        if hours_since_access < 0:
            hours_since_access = 0
        decay = math.exp(-0.693 * hours_since_access / self.attention_decay_hours)
        
        # Very heavy dampening - even 100 accesses should barely matter
        # This ensures temporal always dominates unless temporal scores are nearly equal
        count_boost = math.log1p(fact.access_count) / 30  # max ~0.15 for 100 accesses
        
        return min(1.0, decay * count_boost)
    
    def get_all(self, key: str, at: Optional[datetime] = None) -> list[ScoredFact]:
        """Get all valid facts for a key, sorted by combined score."""
        if at is None:
            at = datetime.now()
        
        if key not in self.facts:
            return []
        
        results = []
        for fact in self.facts[key]:
            is_valid = (
                fact.valid_from <= at and
                (fact.valid_to is None or fact.valid_to > at)
            )
            
            if not is_valid:
                continue
            
            temporal = self._temporal_decay(fact, at)
            attention = self._attention_decay(fact, at)
            combined = self.temporal_weight * temporal + self.attention_weight * attention
            
            results.append(ScoredFact(
                fact=fact,
                temporal_score=temporal,
                attention_score=attention,
                combined_score=combined,
                is_valid=True,
            ))
        
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results
    
    def __repr__(self):
        total_facts = sum(len(v) for v in self.facts.values())
        return f"TemporalAttentionStore(facts={total_facts}, keys={len(self.facts)})"
