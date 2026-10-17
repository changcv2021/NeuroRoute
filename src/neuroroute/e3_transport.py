"""E3 receiver interventions, explicitly separate from the trained protocol.

Selected packets acquire five control steps of delay. Per-link FIFO ordering
can additionally delay later unselected packets; that exposure is logged too.
Packets ready in the same slot are received in send order and the newest one
updates the receiver cache. No transport timestamp/queue state enters actor
inputs or requires an uncharged sender header: these are harness metadata.
"""

from dataclasses import dataclass

from flax import struct
import jax.numpy as jnp
import numpy as np

from .communication import deliver

DELAY_STEPS = 5
CAPACITY = DELAY_STEPS + 1


@struct.dataclass
class LinkQueue:
    payload: jnp.ndarray       # [..., 4, 6, 8]
    send_step: jnp.ndarray     # [..., 4, 6]
    due_step: jnp.ndarray
    valid: jnp.ndarray


@struct.dataclass
class TransportState:
    up: LinkQueue
    down: LinkQueue
    step: jnp.ndarray


@struct.dataclass
class DeliveryLog:
    sent: jnp.ndarray
    sent_payload: jnp.ndarray
    wire_payload: jnp.ndarray
    send_step: jnp.ndarray
    scheduled_arrival_step: jnp.ndarray
    selected_for_delay: jnp.ndarray
    actual_delay_steps: jnp.ndarray
    content_replaced: jnp.ndarray
    content_changed: jnp.ndarray
    arrived: jnp.ndarray       # [..., 4, 6], every receipt, not just newest
    arrived_send_step: jnp.ndarray
    arrived_payload: jnp.ndarray
    arrival_step: jnp.ndarray
    cache_updated: jnp.ndarray
    cache_source_send_step: jnp.ndarray
    overflow: jnp.ndarray


def empty_transport(batch_shape=(), *, start_step=0):
    shape = tuple(batch_shape) + (4, CAPACITY)
    queue = LinkQueue(jnp.zeros(shape + (8,), jnp.uint8),
                      jnp.full(shape, -1, jnp.int32), jnp.full(shape, -1, jnp.int32),
                      jnp.zeros(shape, jnp.bool_))
    return TransportState(queue, queue,
                          jnp.full(tuple(batch_shape), start_step, jnp.int32))


def reset_transport(state, mask, *, start_step=0):
    mask = jnp.asarray(mask, jnp.bool_)
    fresh = empty_transport(mask.shape, start_step=start_step)

    def reset_queue(old, new):
        return LinkQueue(
            jnp.where(mask[..., None, None, None], new.payload, old.payload),
            jnp.where(mask[..., None, None], new.send_step, old.send_step),
            jnp.where(mask[..., None, None], new.due_step, old.due_step),
            jnp.where(mask[..., None, None], new.valid, old.valid))

    return TransportState(reset_queue(state.up, fresh.up), reset_queue(state.down, fresh.down),
                          jnp.where(mask, fresh.step, state.step))


def resolve_delay(dt):
    """E3 v1 locks 100 ms = five 20 ms control steps; fail on other dt."""
    if not np.isclose(float(dt) * DELAY_STEPS, 0.1, rtol=0, atol=1e-10):
        raise ValueError("E3 transport v1 requires five steps to equal 100 ms")
    return DELAY_STEPS


def exchange(queue, cache, gate, payload, step, *, delay_mask=None,
             replacement_payload=None, replacement_mask=None):
    """One directional slot group. ``cache`` must already have been ticked.

    Closed gates enqueue nothing. Capacity is exact for a maximum five-step
    delay and one send per link per step. ``overflow`` must cause harness
    failure; it can only occur after malformed/nonconsecutive use, and is
    never silently interpreted as a packet drop.
    """
    gate = jnp.asarray(gate, jnp.bool_)
    batch = gate.shape[:-1]
    step = jnp.broadcast_to(jnp.asarray(step, jnp.int32), batch)
    now = jnp.broadcast_to(step[..., None], gate.shape)
    selected = jnp.zeros_like(gate) if delay_mask is None else jnp.broadcast_to(delay_mask, gate.shape).astype(jnp.bool_)
    selected = selected & gate
    replaced = jnp.zeros_like(gate) if replacement_mask is None else jnp.broadcast_to(replacement_mask, gate.shape).astype(jnp.bool_)
    replaced = replaced & gate
    if replacement_payload is None:
        wire_payload = payload
        replaced = jnp.zeros_like(gate)
    else:
        wire_payload = jnp.where(replaced[..., None], replacement_payload, payload).astype(jnp.uint8)
    last_due = jnp.max(jnp.where(queue.valid, queue.due_step, -1), axis=-1)
    due = jnp.maximum(now + jnp.where(selected, DELAY_STEPS, 0), last_due)
    overflow = queue.valid[..., 0]
    pending = LinkQueue(
        jnp.concatenate((queue.payload[..., 1:, :], wire_payload[..., None, :]), axis=-2),
        jnp.concatenate((queue.send_step[..., 1:], now[..., None]), axis=-1),
        jnp.concatenate((queue.due_step[..., 1:], due[..., None]), axis=-1),
        jnp.concatenate((queue.valid[..., 1:], gate[..., None]), axis=-1))
    arrived = pending.valid & (pending.due_step <= step[..., None, None])
    newest_slot = jnp.max(jnp.where(arrived, jnp.arange(CAPACITY, dtype=jnp.int32), -1), axis=-1)
    any_arrived = newest_slot >= 0
    take_slot = jnp.maximum(newest_slot, 0)
    newest_payload = jnp.take_along_axis(pending.payload, take_slot[..., None, None], axis=-2)[..., 0, :]
    newest_send = jnp.take_along_axis(pending.send_step, take_slot[..., None], axis=-1)[..., 0]
    next_cache = deliver(cache, any_arrived, newest_payload)
    log = DeliveryLog(gate, payload, wire_payload, now, jnp.where(gate, due, -1), selected,
                       jnp.where(gate, due - now, 0), replaced,
                       replaced & jnp.any(wire_payload != payload, axis=-1), arrived,
                       jnp.where(arrived, pending.send_step, -1), pending.payload,
                       jnp.where(arrived, step[..., None, None], -1), any_arrived,
                       jnp.where(any_arrived, newest_send, -1), overflow)
    return pending.replace(valid=pending.valid & ~arrived), next_cache, log


def transmission_cost(up_log, down_log):
    """Charge source sends, including all selected/queued/replaced packets."""
    up = 4 + 64 * jnp.sum(up_log.sent.astype(jnp.int32), axis=-1)
    down = 4 + 64 * jnp.sum(down_log.sent.astype(jnp.int32), axis=-1)
    return {"total_bits": up + down, "up_bits": up, "down_bits": down}


def exposure_summary(log, dt=0.02, *, initial_pending_payload_bits=0):
    """Host-side summary of a stacked [T,...,4] directional trace."""
    sent = np.asarray(log.sent, dtype=bool)
    selected = np.asarray(log.selected_for_delay, dtype=bool)
    delay = np.asarray(log.actual_delay_steps)
    received = int(np.asarray(log.arrived).sum() * 64)
    return {"sent_payload_bits": int(sent.sum() * 64),
            "selected_payload_bits": int(selected.sum() * 64),
            "actually_delayed_payload_bits": int(((delay > 0) & sent).sum() * 64),
            "payload_bit_seconds_delayed": float((delay * sent).sum() * 64 * dt),
            "max_packet_delay_s": float(delay.max(initial=0) * dt),
            "received_payload_bits": received,
            "pending_payload_bits": int(initial_pending_payload_bits + sent.sum() * 64 - received),
            "replaced_payload_bits": int(np.asarray(log.content_replaced).sum() * 64),
            "changed_payload_bits": int(np.asarray(log.content_changed).sum() * 64)}


def compare_exposure(first, second, *, first_window_steps, second_window_steps):
    keys = ("selected_payload_bits", "actually_delayed_payload_bits", "payload_bit_seconds_delayed")
    differences = {key: [first[key], second[key]] for key in keys if not np.isclose(first[key], second[key], rtol=0, atol=1e-9)}
    if first_window_steps != second_window_steps:
        differences["window_steps"] = [first_window_steps, second_window_steps]
    if first.get("pending_payload_bits", 0) or second.get("pending_payload_bits", 0):
        differences["pending_receipts_censor_exposure"] = [first.get("pending_payload_bits", 0), second.get("pending_payload_bits", 0)]
    matched = not differences
    return {"status": "matched" if matched else "unmatched", "differences": differences,
            "clean_direction_causal_claim_allowed": matched,
            "scope": "exposure equivalence only; does not establish all other causal assumptions"}


def choose_random_matched_window(reference_sent, target_onset, allowed_onsets, seed, *, window_steps=5):
    """Use a frozen intact send trace, never reward, to match exposure count.

    Actual closed-loop exposure must still pass ``compare_exposure`` after
    execution. Candidates and seed are supplied by the preregistered harness.
    """
    flags = np.asarray(reference_sent, dtype=bool)
    if flags.ndim != 2 or flags.shape[1] != 4:
        raise ValueError("Expected a single direction's [T,4] reference send log")
    if window_steps <= 0 or target_onset < 0 or target_onset + window_steps > len(flags):
        raise ValueError("Target window falls outside the frozen reference trace")
    target_count = int(flags[target_onset:target_onset + window_steps].sum())
    eligible = [int(t) for t in allowed_onsets if 0 <= t and t + window_steps <= len(flags)
                and (t + window_steps <= target_onset or t >= target_onset + window_steps)
                and int(flags[t:t + window_steps].sum()) == target_count]
    if not eligible:
        return {"status": "unavailable", "onset": None, "matched_packet_count": target_count}
    return {"status": "reference_matched_requires_actual_audit",
            "onset": int(np.random.default_rng(seed).choice(eligible)),
            "window_steps": window_steps, "matched_packet_count": target_count}


@dataclass(frozen=True)
class PayloadSemantics:
    checkpoint_sha256: str
    direction: str
    leg: int
    codec: str
    support_stratum: str

    def validate(self):
        if (len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256)
                or self.direction not in {"up", "down"}
                or self.leg not in range(4) or not self.codec or not self.support_stratum):
            raise ValueError("Incomplete matched-payload semantics/provenance")


def permute_matched_payloads(payloads, metadata, seed, *, sent_flags):
    """Permute whole sent packets within one audited direction/leg/codec stratum.

    Payload pool formation must use actual gate=1 packets. No swapping of
    byte positions, legs, directions, checkpoints, or unmatched input strata.
    At least two packets are needed; a random cyclic offset avoids fixed points.
    """
    payloads = np.asarray(payloads)
    if payloads.dtype != np.uint8 or payloads.ndim != 2 or payloads.shape[1] != 8:
        raise ValueError("Expected actual uint8 packet pool [N,8]")
    if len(metadata) != len(payloads) or len(payloads) < 2:
        raise ValueError("At least two provenance-matched sent packets are required")
    sent_flags = np.asarray(sent_flags, dtype=bool)
    if sent_flags.shape != (len(payloads),) or not sent_flags.all():
        raise ValueError("Every donor packet must have an actual gate=1 send record")
    for item in metadata:
        item.validate()
        if item != metadata[0]:
            raise ValueError("Cross-direction/leg/codec/support payload replacement is forbidden")
    offset = int(np.random.default_rng(seed).integers(1, len(payloads)))
    indices = (np.arange(len(payloads)) + offset) % len(payloads)
    return payloads[indices], indices
