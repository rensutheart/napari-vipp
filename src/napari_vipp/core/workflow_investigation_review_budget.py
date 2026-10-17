"""Finite first-review allocation, not additional execution authority."""

from __future__ import annotations

REVIEW_ACTION_RESERVE = 7
REVIEW_CALL_RESERVE = 14
RESERVATION_FIELDS = {
    "version",
    "state",
    "total_trial_limit",
    "initial_trial_limit",
    "reserved_trial_attempts",
    "reserved_actions",
    "reserved_model_calls",
    "reason",
}


def review_reservation(limits, continue_requests, *, planned, accepted_baseline):
    """Allocate one original trial and a useful continuation/closing allowance.

    Seven actions cover plan, trial, two evidence actions, assessment, Finish and
    recovery (unplanned callers need no prospective-plan action). Fourteen calls
    allow the existing two format attempts per step. Failed responses consume
    original allowance; actual remaining budgets/time/context/images stay
    authoritative. This policy cannot guarantee future useful work.
    """
    initial_steps = 1 if accepted_baseline else 4 if planned else 3
    # A goal-started session must still reach its first trial before the strict
    # ordinary closing/recovery threshold fires. An accepted seed can instead
    # reach a genuine Finish immediately, without inventing a new candidate.
    initial_slack = 0 if accepted_baseline else 1
    eligible = (
        limits.max_attempts >= 2
        and limits.max_actions >= initial_steps + REVIEW_ACTION_RESERVE + initial_slack
        and limits.max_model_calls
        >= 2 * initial_steps + REVIEW_CALL_RESERVE + initial_slack
    )
    state = "released" if continue_requests else "held" if eligible else "unavailable"
    held = state == "held"
    reason = (
        "One original trial is held and 7 actions/14 calls are allocated to the "
        "first Keep improving decision. Failed responses consume original "
        "allowance; current remaining budgets and continuation status govern. "
        "Finish is only a genuine model nomination; actual time, context, image "
        "and authority limits still apply."
        if held
        else "The first successful Keep improving released its allocation within "
        "the original limits; nothing was reset or added."
        if continue_requests
        else "The configured original limits cannot reserve an initial review "
        "plus a useful continuation. The sole/available initial trials remain "
        "usable, but Keep improving may be unavailable; no allowance is added."
    )
    return {
        "version": 1,
        "state": state,
        "total_trial_limit": limits.max_attempts,
        "initial_trial_limit": limits.max_attempts - int(eligible),
        "reserved_trial_attempts": int(held),
        "reserved_actions": REVIEW_ACTION_RESERVE if held else 0,
        "reserved_model_calls": REVIEW_CALL_RESERVE if held else 0,
        "reason": reason,
    }


def validate_review_reservation(value, policy, budgets):
    """Validate bounded policy disclosure, never issue work/evidence authority."""
    if (
        type(budgets) is not dict
        or type(value) is not dict
        or set(value) != RESERVATION_FIELDS
        or type(value["version"]) is not int
        or value["version"] != 1
        or type(value["state"]) is not str
        or value["state"] not in {"held", "released", "unavailable"}
        or type(value["reason"]) is not str
        or not value["reason"]
        or len(value["reason"]) > 1000
        or any(
            type(value[key]) is not int
            for key in (
                "total_trial_limit",
                "initial_trial_limit",
                "reserved_trial_attempts",
                "reserved_actions",
                "reserved_model_calls",
            )
        )
    ):
        raise ValueError("Invalid completion-review reservation.")
    total, initial = value["total_trial_limit"], value["initial_trial_limit"]
    held = value["state"] == "held"
    remaining = budgets.get("attempts_remaining")
    if (
        not 1 <= total <= 5
        or initial not in {total, total - 1}
        or initial < 1
        or type(remaining) is not int
        or not 0 <= remaining <= total
        or value["reserved_trial_attempts"] != int(held)
        or value["reserved_actions"] != (REVIEW_ACTION_RESERVE if held else 0)
        or value["reserved_model_calls"] != (REVIEW_CALL_RESERVE if held else 0)
        or (held and (total < 2 or initial != total - 1 or remaining < 1))
        or (value["state"] == "unavailable" and initial != total)
        or (value["state"] == "released") != (policy["continue_requests"] > 0)
    ):
        raise ValueError("Inconsistent completion-review reservation.")
