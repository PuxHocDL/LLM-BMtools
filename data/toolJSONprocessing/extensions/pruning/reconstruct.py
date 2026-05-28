from .flattening import parse_path

def reconstruct(json_obj, keep_paths):
    """
    Reconstruct a JSON subtree from the original json_obj,
    keeping only the elements whose path (or any descendant path)
    is in `keep_paths`.
    """
    # A path is kept if it is in keep_paths, OR if one of its descendants is in keep_paths.
    # To make this easy, we can check if the current path is a prefix of any keep_paths.
    
    # We can optimize this by building a tree of keep_paths.
    path_tree = {}
    for path in keep_paths:
        parts = parse_path(path)
        cur = path_tree
        for p in parts:
            if p not in cur:
                cur[p] = {}
            cur = cur[p]
            
    def build(obj, tree_node):
        if not tree_node:
            # If there's no deeper tree but this path was kept, we keep the whole subtree (obj).
            # Wait, if a path to a leaf is kept, tree_node is empty, we return obj.
            # If a path to an array is kept, tree_node is empty, we return obj.
            return obj
        
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if k in tree_node:
                    new_obj[k] = build(v, tree_node[k])
            return new_obj
            
        elif isinstance(obj, list):
            # In an array, we must maintain order. We only keep elements whose index is in tree_node.
            # But wait, what if an array was kept entirely?
            # If the index is not in tree_node, we skip it.
            new_list = []
            for i, item in enumerate(obj):
                if i in tree_node:
                    new_list.append(build(item, tree_node[i]))
            return new_list
            
        else:
            return obj
            
    return build(json_obj, path_tree)
