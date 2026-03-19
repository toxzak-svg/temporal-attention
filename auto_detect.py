"""
Smart Memory - Auto-detect both 2x2 AND float
"""

import math


# Auto-detect for 2x2
def detect_2x2(key, value):
    """Returns (row, col) for the 2x2 matrix."""
    
    key_lower = key.lower()
    value_str = str(value).lower()
    
    # ROW: permanence (0=temp, 1=perm)
    if any(k in key_lower for k in ['name', 'user', 'email', 'phone']):
        row = 1  # permanent
    elif any(k in key_lower for k in ['weather', 'temp', 'status', 'current']):
        row = 0  # temporary
    elif any(k in value_str for k in ['yes', 'no', 'true', 'false']):
        row = 0  # ephemeral facts
    else:
        row = 0.5  # unknown
    
    # COL: mutability (0=stable, 1=mutable)
    if any(k in key_lower for k in ['project', 'job', 'title', 'role', 'status']):
        col = 1  # mutable - can change
    elif value.istitle() and len(value) < 15:
        col = 0.5  # might be name
    else:
        col = 0  # stable
    
    return (row, col)


# Auto-detect for float
def detect_float(key, value):
    """Returns permanence 0-1."""
    
    key_lower = key.lower()
    value_str = str(value).lower()
    
    permanence = 0.5  # default
    
    # From key
    if any(k in key_lower for k in ['name', 'user', 'email']):
        permanence = max(permanence, 1.0)
    elif any(k in key_lower for k in ['project', 'job', 'role']):
        permanence = max(permanence, 0.8)
    elif any(k in key_lower for k in ['weather', 'temp', 'status', 'current']):
        permanence = min(permanence, 0.1)
    elif any(k in key_lower for k in ['pref', 'like', 'hate']):
        permanence = max(permanence, 0.7)
    
    # From value
    if value.istitle() and len(value) < 15:
        permanence = max(permanence, 0.9)
    elif value_str in ['yes', 'no', 'true', 'false']:
        permanence = min(permanence, 0.3)
    elif value_str.isdigit() and len(value_str) == 4:  # year
        permanence = max(permanence, 0.8)
    
    return permanence


# Test cases
tests = [
    ("user_name", "Zach"),
    ("project_name", "Alpha"),
    ("weather", "sunny"),
    ("food_preference", "pizza"),
    ("email", "zach@email.com"),
    ("current_status", "active"),
]

print("="*60)
print("AUTO-DETECTION TEST")
print("="*60)

for key, value in tests:
    row, col = detect_2x2(key, value)
    flo = detect_float(key, value)
    
    row_name = {0: "temp", 0.5: "?", 1: "perm"}[row]
    col_name = {0: "stable", 0.5: "?", 1: "mutable"}[col]
    
    print(f"{key:20} = {value:10}")
    print(f"  2x2: ({row_name}, {col_name})")
    print(f"  float: {flo}")
    print()
