from .flattening import flatten

def coverage(pruned_obj, gt_paths):
    """Calculate the fraction of ground truth paths covered in the pruned object."""
    if not gt_paths:
        return 0.0
    flat = flatten(pruned_obj)
    
    covered = 0
    for gtp in gt_paths:
        norm_gtp = gtp.replace(".[", "[").strip()
        if any(norm_gtp in fk for fk in flat.keys()):
            covered += 1
    return float(covered) / len(gt_paths)
