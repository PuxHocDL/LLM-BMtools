import re

def flatten(obj, prefix=""):
    """
    Flatten JSON object to a list of (path, value) pairs.
    Arrays use [i] for specific index.
    """
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            out.extend(flatten(v, new_prefix))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            out.extend(flatten(item, new_prefix))
    else:
        out.append((prefix, obj))
    return out

def parse_path(path):
    """Parse a path string like 'data.search_results[0].vehicle_id' into parts."""
    parts = []
    current = ""
    for char in path:
        if char == '.':
            if current:
                parts.append(current)
                current = ""
        elif char == '[':
            if current:
                parts.append(current)
                current = ""
        elif char == ']':
            if current:
                parts.append(int(current))
                current = ""
        else:
            current += char
    if current:
        parts.append(current)
    return parts

def get_value_by_path(obj, path):
    """Navigate path to get value."""
    parts = parse_path(path)
    cur = obj
    for p in parts:
        if isinstance(p, int):
            if isinstance(cur, list) and 0 <= p < len(cur):
                cur = cur[p]
            else:
                return None
        else:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
    return cur

def ancestors_of(path):
    """
    Given a path 'a.b[0].c', return set of ancestor paths:
    {'a', 'a.b', 'a.b[0]'}
    """
    ans = set()
    current = ""
    for char in path:
        if char == '.' and current:
            ans.add(current)
        elif char == '[' and current:
            ans.add(current)
        current += char
    return ans
