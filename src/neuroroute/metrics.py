"""Physical-unit and failure-aware metrics; no learning or test-based selection."""
import math
import os
import socket


def tracking_metrics(velocity_xy, yaw_rate, command, *, horizon_steps, dt_s,
                     failure_step=None, sigma_v=0.5, sigma_w=0.5):
    """failure_step is the first invalid post-step sample; tail earns zero.

    Physical RMSE uses observed alive samples only and reports their count.
    Incomplete traces without an explicit failure are rejected, not called success.
    """
    import numpy as np
    v, w, c = np.asarray(velocity_xy, float), np.asarray(yaw_rate, float), np.asarray(command, float)
    n = len(w)
    if v.shape != (n, 2) or c.shape != (n, 3) or n > horizon_steps or horizon_steps <= 0:
        raise ValueError("Expected aligned [T,2], [T], [T,3] traces within fixed horizon")
    if any(not math.isfinite(x) or x <= 0 for x in (dt_s, sigma_v, sigma_w)):
        raise ValueError("Time and scoring scales must be positive")
    if failure_step is not None and (not isinstance(failure_step, int) or not 0 <= failure_step <= horizon_steps):
        raise ValueError("Failure step must be an integer within the horizon")
    alive = horizon_steps if failure_step is None else failure_step
    if n < alive:
        raise ValueError("Incomplete alive trace; do not invent missing observations")
    if not np.isfinite(v[:alive]).all() or not np.isfinite(w[:alive]).all() or not np.isfinite(c[:alive]).all():
        raise ValueError("Nonfinite alive observation")
    ve2 = np.sum((v[:alive] - c[:alive, :2]) ** 2, axis=-1)
    we2 = (w[:alive] - c[:alive, 2]) ** 2
    score = np.zeros(horizon_steps)
    score[:alive] = np.exp(-ve2 / sigma_v ** 2 - we2 / sigma_w ** 2)
    return {"score": float(score.mean()), "score_trace": score.tolist(),
        "velocity_rmse_m_s": float(np.sqrt(ve2.mean())) if alive else None,
        "yaw_rmse_rad_s": float(np.sqrt(we2.mean())) if alive else None,
        "physical_metric_observed_steps": alive, "physical_metric_full_horizon": alive == horizon_steps,
        "survival_s": alive * dt_s, "completed": failure_step is None,
        "failure_step": failure_step, "horizon_steps": horizon_steps,
        "score_sigma_v_m_s": sigma_v, "score_sigma_w_rad_s": sigma_w}


def recovery_metrics(score_trace, *, event_step, dt_s, failure_step=None,
                     score_threshold=0.8, hold_s=0.5):
    """Area of dimensionless tracking deficit over full postevent horizon.

    Recovery means score >= threshold throughout the specified hold interval.
    No event trial is dropped when failure occurs before the event. A failure is
    labelled as such, not interpreted as noninformative survival censoring.
    """
    import numpy as np
    score = np.asarray(score_trace, float).copy()
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("Expected a finite bounded score trace")
    if not 0 <= event_step < len(score) or not math.isfinite(dt_s) or dt_s <= 0 or not 0 < score_threshold <= 1:
        raise ValueError("Invalid event time, dt, or recovery threshold")
    hold = round(hold_s / dt_s)
    if hold < 1 or not math.isclose(hold * dt_s, hold_s, abs_tol=1e-9):
        raise ValueError("Recovery hold must be a positive exact number of control steps")
    end = len(score) if failure_step is None else failure_step
    if not isinstance(end, int) or not 0 <= end <= len(score):
        raise ValueError("Failure step is outside the horizon")
    score[end:] = 0
    prefailed = failure_step is not None and failure_step <= event_step
    recovered = None
    if not prefailed:
        for start in range(event_step, end - hold + 1):
            if np.all(score[start:start + hold] >= score_threshold):
                recovered = start
                break
    status = "recovered" if recovered is not None else ("failed_before_event" if prefailed else
              "failed_after_event" if failure_step is not None else "unrecovered_at_horizon")
    return {"postevent_deficit_area_s": float(np.sum(1-score[event_step:]) * dt_s),
        "postevent_horizon_s": (len(score)-event_step) * dt_s,
        "recovery_time_s": (recovered-event_step) * dt_s if recovered is not None else None,
        "recovery_confirmed_at_s": (recovered+hold-event_step) * dt_s if recovered is not None else None,
        "right_censored": recovered is None, "censor_time_s": max(0, end-event_step) * dt_s if recovered is None else None,
        "status": status, "failed_before_event": prefailed,
        "censoring_note": "failure and administrative end are distinct; do not treat failures as noninformative censoring"}


def retention(fault_score, healthy_score, *, minimum_healthy_score=1e-6):
    """Always retain absolute scores and differences; never clip ratios above1."""
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (fault_score, healthy_score)):
        raise ValueError("Retention requires bounded task scores, not raw rewards")
    if not math.isfinite(minimum_healthy_score) or minimum_healthy_score < 0:
        raise ValueError("Minimum healthy score must be finite and nonnegative")
    return {"fault_score": fault_score, "healthy_score": healthy_score,
            "difference": fault_score-healthy_score,
            "ratio": fault_score/healthy_score if healthy_score > minimum_healthy_score else None}


def bandwidth_metrics(sent_gates, *, dt_s, delivered_gates=None, received_flags=None,
                      planned_horizon_steps=None, budget_bits_per_step=None):
    """Trace shape[T,direction(up,down),leg], one flag per slot plus64 payload bits.

    Sent and received traces are separate. Delayed delivery need not coincide
    with a send at that same step. Unknown reception remains null, never assumed.
    """
    import numpy as np

    def boolean_trace(value):
        a = np.asarray(value)
        if a.ndim != 3 or a.shape[1:] != (2, 4) or not np.isin(a, [0, 1]).all():
            raise ValueError("Expected actual binary gate/flag trace [T,2,4]")
        return a.astype(bool)

    sent = boolean_trace(sent_gates)
    n = len(sent)
    if not n or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Nonempty trace and positive control dt required")
    horizon = n if planned_horizon_steps is None else planned_horizon_steps
    if horizon < n:
        raise ValueError("Planned horizon shorter than the observed trace")
    flags = np.full((n, 2), 4)
    payload = sent.sum(axis=-1) * 64
    bits = flags + payload
    result = {"sent_flag_bits": int(flags.sum()), "sent_payload_bits": int(payload.sum()),
        "sent_total_bits": int(bits.sum()), "active_bits_per_step": float(bits.sum()/n),
        "active_bits_per_s": float(bits.sum()/(n*dt_s)),
        "planned_horizon_bits_per_s": float(bits.sum()/(horizon*dt_s)),
        "active_steps": n, "planned_horizon_steps": horizon,
        "upward_sent_bits": int(bits[:, 0].sum()), "downward_sent_bits": int(bits[:, 1].sum()),
        "upward_payload_fraction": float(payload[:, 0].sum()/payload.sum()) if payload.sum() else None,
        "bits_per_step_quantiles": {str(q): float(np.quantile(bits.sum(axis=-1), q)) for q in [0.5, 0.9, 0.99]},
        "received_payload_bits": None, "received_flag_bits": None, "received_total_bits": None}
    if delivered_gates is not None:
        delivered = boolean_trace(delivered_gates)
        if delivered.shape != sent.shape:
            raise ValueError("Delivery and sending traces must align")
        if np.any(delivered.sum(axis=0) > sent.sum(axis=0)):
            raise ValueError("Delivered more packets than sent from an initially empty link")
        result["received_payload_bits"] = int(delivered.sum()*64)
    if received_flags is not None:
        received = boolean_trace(received_flags)
        if received.shape != sent.shape:
            raise ValueError("Received flags and sending traces must align")
        result["received_flag_bits"] = int(received.sum())
    if result["received_payload_bits"] is not None and result["received_flag_bits"] is not None:
        result["received_total_bits"] = result["received_payload_bits"] + result["received_flag_bits"]
    if budget_bits_per_step is not None:
        if budget_bits_per_step <= 0:
            raise ValueError("Budget must be positive")
        result["mean_budget_excess_bits_per_step"] = max(0., result["active_bits_per_step"]-budget_bits_per_step)
        result["mean_budget_ratio"] = result["active_bits_per_step"]/budget_bits_per_step
    return result


def hierarchical_bootstrap(seed_cases, *, replicates=5000, confidence=0.95, seed=20260912):
    """Independent training seeds outer, episodes inner; equal weight per seed.

    This is a single-method interval. Same numeric seed across methods does not
    make neural initializations paired; do not use this to invent method pairing.
    """
    import numpy as np
    if replicates > 100 and (not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")):
        raise RuntimeError("Production hierarchical bootstrap requires a compute allocation")
    if replicates < 1 or not 0 < confidence < 1 or not seed_cases:
        raise ValueError("Invalid bootstrap settings or empty training-seed sample")
    labels = sorted(seed_cases, key=str)
    values = [np.asarray(seed_cases[k], dtype=float) for k in labels]
    if any(v.ndim != 1 or not len(v) or not np.isfinite(v).all() for v in values):
        raise ValueError("Each training seed needs finite case metrics; missing cases must be reported")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(replicates)
    for b in range(replicates):
        selected = rng.integers(0, len(values), size=len(values))
        samples[b] = np.mean([values[i][rng.integers(0, len(values[i]), len(values[i]))].mean() for i in selected])
    alpha = (1-confidence)/2
    per_seed = {str(k): float(v.mean()) for k, v in zip(labels, values)}
    return {"mean": float(np.mean(list(per_seed.values()))), "per_seed": per_seed,
        "interval": np.quantile(samples, [alpha, 1-alpha]).tolist(), "confidence": confidence,
        "training_seed_count": len(values), "case_counts": {str(k):len(v) for k,v in zip(labels,values)},
        "replicates": replicates, "bootstrap_seed": seed, "unit_order": ["training_seed", "case"]}
