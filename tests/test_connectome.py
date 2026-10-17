"""Tiny synthetic unit fixtures only; never used as biological training priors."""
import numpy as np
import pytest

from neuroroute.connectome.derive_prior import derive, effective_rank
from neuroroute.connectome.extract import select_neurons
from neuroroute.connectome.make_controls import make_controls
from neuroroute.connectome.normalize import aggregate_type_balanced


def test_rank_and_direction_comparability():
    assert effective_rank(np.eye(3))[0] == pytest.approx(3)
    assert effective_rank(np.ones((3, 4)))[0] == pytest.approx(1)
    with pytest.raises(ValueError):
        effective_rank(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="same anatomical"):
        derive(np.eye(2), np.eye(2), [["front", "hind"], ["a", "b"]],
               [["b", "a"], ["front", "hind"]])
    result = derive(np.eye(2), np.ones((2, 2)), [["front", "hind"], ["a", "b"]],
                    [["a", "b"], ["front", "hind"]])
    assert result["pi_bio"] == pytest.approx([2 / 3, 1 / 3])
    assert make_controls(result)["M_REVERSE"]["pi"] == pytest.approx([1 / 3, 2 / 3])


def test_type_balancing_not_population_weighting_and_unknown_not_zero():
    neurons = {"1": ("leg", "A", "row"), "2": ("leg", "A", "row"),
               "3": ("leg", "B", "row"), "9": ("cluster", "T", "column")}
    edges = [(1, 9, 10, 100), (2, 9, 10, 100), (3, 9, 60, 100), (88, 9, 20, 100)]
    matrix = aggregate_type_balanced(edges, neurons, ["leg"], ["cluster"])
    assert matrix[0, 0] == pytest.approx((0.1 + 0.6) / 2)
    matrix = aggregate_type_balanced(edges, neurons, ["leg"], ["cluster"],
        normalization="cell_pair_mean_input_fraction")
    assert matrix[0, 0] == pytest.approx(0.8 / 3)
    with pytest.raises(ValueError, match="Unobserved group"):
        aggregate_type_balanced(edges, neurons, ["leg", "unknown"], ["cluster"])


def test_selection_preserves_exact_ids_and_reports_missing_annotations():
    config = {"id_column": "banc_888_id", "type_column": "cell_type", "role_column": "super_class",
        "sensory_body_column": "body_part_sensory", "motor_body_column": "body_part_effector",
        "sensory_roles": ["sensory"], "body_groups": ["front_leg", "middle_leg", "hind_leg"],
        "metadata_file": "real_meta.feather", "cluster_file": "real_clusters.csv"}
    aid, did = "720575941351919408", "720575941351919409"
    records = [{"banc_888_id": int(aid), "cell_type": "A", "super_class": "ascending"},
        {"banc_888_id": int(did), "cell_type": "D", "super_class": "descending"},
        {"banc_888_id": 3, "cell_type": "NA", "super_class": "sensory", "body_part_sensory": "front_leg"}]
    clusters = [{"id": aid, "super_class": "ascending", "super_cluster": "walking"},
                {"id": did, "super_class": "descending", "super_cluster": "walking"}]
    selected, coverage = select_neurons(records, clusters, config)
    assert [r["root_id"] for r in selected] == [aid, did]
    assert coverage["excluded"]["unknown_annotation:sensory"] == 1
    assert coverage["shared_clusters"] == ["walking"]


def test_multitag_body_mapping_and_explicit_quality_filter():
    from neuroroute.connectome.extract import body_group
    groups = ["front_leg", "middle_leg", "hind_leg"]
    assert body_group("coxa,front_leg", groups) == "coxa,front_leg"
    assert body_group("coxa,front_leg", groups, "unique_leg_token") == "front_leg"
    assert body_group("front_leg, hind_leg", groups, "unique_leg_token") == "ambiguous_leg_annotation"
    assert body_group(None, groups, "unique_leg_token") == "unknown"
    config = {"id_column": "id", "type_column": "type", "role_column": "role",
        "sensory_body_column": "body", "motor_body_column": "body", "sensory_roles": ["sensory"],
        "body_groups": groups, "metadata_file": "fixture", "cluster_file": "fixture",
        "annotation_policy": {"body_tag_policy": "unique_leg_token",
            "quality_any": [{"field": "proofread", "accepted_values": ["TRUE"]}]}}
    rows = [{"id": 1, "role": "sensory", "type": "A", "body": "coxa,front_leg", "proofread": "TRUE"},
            {"id": 2, "role": "sensory", "type": "A", "body": "front_leg", "proofread": "FALSE"},
            {"id": 3, "role": "sensory", "type": "A", "body": "front_leg", "proofread": None}]
    selected, coverage = select_neurons(rows, [], config)
    assert [r["root_id"] for r in selected] == ["1"]
    assert selected[0]["group"] == "front_leg"
    assert coverage["excluded"]["quality_filter:sensory"] == 2
