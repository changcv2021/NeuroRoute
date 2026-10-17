"""Group aggregation with identical definitions for both communication directions."""
from collections import defaultdict


def aggregate_type_balanced(edges, neurons, row_groups, column_groups, threshold=5,
                            normalization="type_pair_mean_input_fraction"):
    """Average input fractions over cell pairs, then over type pairs.

    neurons maps exact string root ID to (group, type, side-of-matrix).
    Missing annotations must already have been excluded and reported. Absent edges
    between annotated cells are measured zeros. Never allocate a neuron NxN array.
    """
    import numpy as np
    counts = defaultdict(int)
    types = defaultdict(set)
    for group, cell_type, side in neurons.values():
        counts[(side, group, cell_type)] += 1
        types[(side, group)].add(cell_type)
    totals = defaultdict(float)
    for pre, post, count, post_count in edges:
        src, dst = neurons.get(str(pre)), neurons.get(str(post))
        if src is None or dst is None or src[2] != "row" or dst[2] != "column":
            continue
        if count < 0 or post_count <= 0 or count > post_count:
            raise ValueError("Invalid synapse count or postsynaptic denominator")
        if count < threshold:
            continue
        totals[(src[0], src[1], dst[0], dst[1])] += count / post_count
    matrix = np.zeros((len(row_groups), len(column_groups)), dtype=float)
    for i, row in enumerate(row_groups):
        for j, col in enumerate(column_groups):
            rt, ct = types[("row", row)], types[("column", col)]
            if not rt or not ct:
                raise ValueError(f"Unobserved group {row}/{col}; cannot impute zero")
            if normalization == "type_pair_mean_input_fraction":
                matrix[i, j] = sum(totals[(row, a, col, b)] /
                    (counts[("row", row, a)] * counts[("column", col, b)])
                    for a in rt for b in ct) / (len(rt) * len(ct))
            elif normalization == "cell_pair_mean_input_fraction":
                matrix[i, j] = sum(totals[(row, a, col, b)] for a in rt for b in ct) / (
                    sum(counts[("row", row, a)] for a in rt) *
                    sum(counts[("column", col, b)] for b in ct))
            else:
                raise ValueError(f"Unknown normalization: {normalization}")
    return matrix
