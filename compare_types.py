"""
Smart Memory - 2x2 Matrix (Permanence x Change)

Two dimensions:
- PERMANENCE: how long we remember it
- MUTABILITY: whether it can change

Matrix:
           | Stable | Mutable
-----------+--------+--------
Temporary  | Context| Ephem
Perm       | Name  | Mutable
"""

import math


class Fact:
    def __init__(self, key, value, row, col, msg):
        self.key = key
        self.value = value
        self.row = row  # 0=temp, 1=perm  
        self.col = col  # 0=stable, 1=mutable
        self.msg = msg


class SmartMemory:
    """Memory with 2x2 fact types."""
    
    # Decay rates for each quadrant (halflife in messages)
    DECAY = {
        (0, 0): 30,   # Context: temp + stable
        (0, 1): 5,    # Ephemeral: temp + mutable
        (1, 0): float('inf'),  # Name: perm + stable
        (1, 1): 500,  # Mutable: perm + mutable
    }
    
    def __init__(self):
        self.facts = {}
        self.msg = 0
        self.focus = "general"
    
    def put(self, key, value, row, col):
        """row: 0=temp, 1=perm | col: 0=stable, 1=mutable"""
        f = Fact(key, value, row, col, self.msg)
        self.facts.setdefault(key, []).append(f)
    
    def get(self, key):
        if key not in self.facts:
            return None
        
        candidates = []
        for f in self.facts[key]:
            decay = self._decay(f)
            candidates.append((decay, f))
        
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1] if candidates else None
    
    def history(self, key, limit=10):
        if key not in self.facts:
            return []
        return sorted(self.facts[key], key=lambda f: -f.msg)[:limit]
    
    def _decay(self, f):
        hl = self.DECAY[(f.row, f.col)]
        if hl == float('inf'):
            return 1.0
        return math.exp(-0.693 * (self.msg - f.msg) / hl)
    
    def advance(self):
        self.msg += 1


class FloatMemory:
    """Memory with single permanence float (0-1)."""
    
    def __init__(self):
        self.facts = {}
        self.msg = 0
    
    def put(self, key, value, permanence=0.5):
        # Convert 0-1 permanence to halflife
        # 0 = instant decay (halflife=1)
        # 1 = never decays (halflife=inf)
        if permanence >= 1:
            hl = float('inf')
        else:
            hl = 1 + permanence * 500  # 1 to 501
        
        self.facts.setdefault(key, []).append({
            'value': value,
            'msg': self.msg,
            'halflife': hl
        })
    
    def get(self, key):
        if key not in self.facts:
            return None
        
        candidates = []
        for f in self.facts[key]:
            decay = self._decay(f)
            candidates.append((decay, f))
        
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]['value'] if candidates else None
    
    def _decay(self, f):
        if f['halflife'] == float('inf'):
            return 1.0
        return math.exp(-0.693 * (self.msg - f['msg']) / f['halflife'])
    
    def advance(self):
        self.msg += 1


# Test both
print("="*60)
print("COMPARING 2x2 MATRIX vs FLOAT")
print("="*60)

# ===== 2x2 =====
print("\n[2x2 MATRIX]")
m = SmartMemory()

# Test facts
m.put("user_name", "Zach", row=1, col=0)  # perm + stable = name
m.put("project", "Alpha", row=1, col=1)      # perm + mutable
m.put("food", "pizza", row=0, col=0)        # temp + stable = context
m.put("weather", "sunny", row=0, col=1)      # temp + mutable = ephem

# Change project
m.put("project", "Beta", row=1, col=1)  # update

# Advance 100 messages
for _ in range(100):
    m.advance()

print(f"After 100 messages:")
print(f"  Name (perm+stable): {m.get('user_name').value}")
print(f"  Project (perm+mutable): {m.get('project').value}")
print(f"  History: {[f.value for f in m.history('project')]}")
print(f"  Food (temp+stable): {m.get('food').value}")
print(f"  Weather (temp+mutable): {m.get('weather').value}")

# ===== FLOAT =====
print("\n[FLOAT PERMANENCE]")
f = FloatMemory()

f.put("user_name", "Zach", permanence=1.0)   # max permanence
f.put("project", "Alpha", permanence=0.8)    # high permanence  
f.put("food", "pizza", permanence=0.5)       # medium
f.put("weather", "sunny", permanence=0.1)    # low permanence
f.put("project", "Beta", permanence=0.8)

for _ in range(100):
    f.advance()

print(f"After 100 messages:")
print(f"  Name (1.0): {f.get('user_name')}")
print(f"  Project (0.8): {f.get('project')}")
print(f"  Food (0.5): {f.get('food')}")
print(f"  Weather (0.1): {f.get('weather')}")

print("\n" + "="*60)
print("COMPARISON:")
print("="*60)
print("""
2x2 Matrix:
- 4 types: name, mutable, context, ephem
- Clear mental model
- Harder to auto-detect

Float:
- 1 number (0-1)
- Easier auto-detect from content
- Harder to explain "what is this?"
""")
