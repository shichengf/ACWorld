"""Focused offline validation for data, tasks, oracles, and score separation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from large_catalog.database import CatalogDataError, CatalogDatabase
from large_catalog.models import LargeCatalogTask
from large_catalog.oracle import CartOracle, independently_verify_cart
from large_catalog.policies import ERROR_POLICY_NAMES, FixedErrorPolicy
from large_catalog.reference import ReferencePolicy
from large_catalog.runtime import LargeCatalogWorld, run_episode
from large_catalog.scoring import score_run


@dataclass(frozen=True, slots=True)
class ValidationReport:
    tasks: int
    reference_full: int
    process_reward_equalities: int
    process_rewards_with_event_links: int
    deterministic_rescores: int
    global_single_item_checks: int
    independent_cart_checks: int
    bounded_cart_query_checks: int
    mutation_runs: int
    score_buckets: tuple[int, int, int, int]
    nonreference_mean_max: float
    scenarios_with_four_scores: int
    scenarios_with_five_error_modes: int
    baseline_means: dict[str, float]
    baseline_score_levels: int
    passed: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_suite(
    database_path: str | Path,
    tasks: Sequence[LargeCatalogTask],
) -> ValidationReport:
    issues: list[str] = []
    reference_full = 0
    reward_equalities = 0
    linked_rewards = 0
    rescores = 0
    global_single_checks = 0
    cart_checks = 0
    bounded_cart_checks = 0
    mutation_scores: list[float] = []
    by_scenario: dict[str, set[float]] = {}
    by_scenario_modes: dict[str, set[str]] = {}
    policy_means: dict[str, list[float]] = {name: [] for name in ERROR_POLICY_NAMES}
    with CatalogDatabase(database_path) as database:
        cart_task = next(task for task in tasks if "requirements" in task.public_context)
        try:
            LargeCatalogWorld(database, cart_task)._search_carts(
                {
                    "requirements": [
                        {"query": "", "filters": {"in_stock": True}},
                    ],
                    "constraints": {},
                    "page_size": 20,
                }
            )
        except CatalogDataError:
            bounded_cart_checks = 1
        for task in tasks:
            if task.oracle.get("kind") == "selection":
                preference = dict(
                    task.public_context.get("preference")
                    or task.public_context.get("latest_preference")
                    or {}
                )
                if preference.get("kind", "lowest_price") == "lowest_price":
                    accepted = tuple(task.oracle.get("accepted_refs", ()))
                    if accepted:
                        listing = database.listing(str(accepted[0]))
                        global_single_checks += int(
                            listing is not None
                            and database.not_exists_better(
                                query=str(task.public_context["query"]),
                                filters=dict(task.public_context["filters"]),
                                chosen_price_minor=listing.price_minor,
                            )
                        )
                    else:
                        global_single_checks += int(
                            not database.full_candidates(
                                query=str(task.public_context["query"]),
                                filters=dict(task.public_context["filters"]),
                            )
                        )
            if (
                "requirements" in task.public_context
                and task.oracle.get("kind") in {"cart", "quote_decision"}
            ):
                expected = CartOracle(
                    feasible_count=int(task.oracle.get("feasible_count", 0)),
                    objective_total_minor=task.oracle.get("objective_total_minor"),
                    accepted_carts=tuple(
                        tuple(str(ref) for ref in cart)
                        for cart in task.oracle.get("accepted_carts", ())
                    ),
                )
                cart_checks += int(
                    independently_verify_cart(
                        database,
                        requirements=task.public_context["requirements"],
                        constraints=task.public_context.get("constraints", {}),
                        expected=expected,
                    )
                )
            run = run_episode(task=task, database=database, policy=ReferencePolicy())
            result = score_run(task, run, model_id="reference")
            reference_full += int(result.strict_success and result.score == 1.0)
            reward_score = sum(row.points for row in result.process_rewards) / sum(
                row.maximum for row in result.process_rewards
            )
            reward_equalities += int(reward_score == result.score)
            linked_rewards += int(
                all(
                    reward.event_ref is not None
                    for reward in result.process_rewards
                    if reward.points > 0
                )
            )
            repeated = score_run(task, run, model_id="reference")
            rescores += int(
                repeated.score == result.score
                and repeated.strict_success == result.strict_success
                and repeated.process_rewards == result.process_rewards
            )
            for mode in ERROR_POLICY_NAMES:
                mutation = run_episode(
                    task=task,
                    database=database,
                    policy=FixedErrorPolicy(task, mode),
                )
                scored = score_run(task, mutation, model_id=f"error:{mode}")
                policy_means[mode].append(scored.score)
                if scored.score < 1.0:
                    mutation_scores.append(scored.score)
                    by_scenario.setdefault(task.scenario, set()).add(round(scored.score, 8))
                    by_scenario_modes.setdefault(task.scenario, set()).add(mode)
    if reference_full != len(tasks):
        issues.append(f"reference policy full-credit count is {reference_full}/{len(tasks)}")
    if reward_equalities != len(tasks):
        issues.append("process rewards do not sum to the task score")
    if linked_rewards != len(tasks):
        issues.append("at least one earned process reward lacks a trace event")
    if rescores != len(tasks):
        issues.append("deterministic rescore changed at least one result")
    expected_single_checks = sum(
        task.oracle.get("kind") == "selection"
        and (
            task.public_context.get("preference")
            or task.public_context.get("latest_preference")
            or {}
        ).get("kind", "lowest_price")
        == "lowest_price"
        for task in tasks
    )
    if global_single_checks != expected_single_checks:
        issues.append(
            f"global single-item checks pass for "
            f"{global_single_checks}/{expected_single_checks} tasks"
        )
    expected_cart_checks = sum(
        "requirements" in task.public_context
        and task.oracle.get("kind") in {"cart", "quote_decision"}
        for task in tasks
    )
    if cart_checks != expected_cart_checks:
        issues.append(
            f"independent cart checks pass for {cart_checks}/{expected_cart_checks} tasks"
        )
    if bounded_cart_checks != 1:
        issues.append("an unbounded cart query was not rejected before materialization")
    buckets = (
        sum(score < 0.25 for score in mutation_scores),
        sum(0.25 <= score < 0.5 for score in mutation_scores),
        sum(0.5 <= score < 0.75 for score in mutation_scores),
        sum(0.75 <= score < 1.0 for score in mutation_scores),
    )
    if any(value == 0 for value in buckets):
        issues.append(f"mutation scores do not cover all four sub-full buckets: {buckets}")
    scenarios_with_four = sum(len(scores) >= 4 for scores in by_scenario.values())
    if scenarios_with_four != 20:
        issues.append(
            f"only {scenarios_with_four}/20 scenarios produce four distinct mutation scores"
        )
    scenarios_with_five_modes = sum(
        len(modes) >= 5 for modes in by_scenario_modes.values()
    )
    if scenarios_with_five_modes != 20:
        issues.append(
            f"only {scenarios_with_five_modes}/20 scenarios expose five error modes"
        )
    high_share = (
        sum(score >= 0.8 for score in mutation_scores) / len(mutation_scores)
        if mutation_scores
        else 1.0
    )
    if high_share > 0.25:
        issues.append(f"{high_share:.1%} of fixed error runs score at least 0.8")
    baseline_names = (
        "first_result",
        "constraint_blind_cheapest",
        "evidence_free",
    )
    baseline_means = {
        name: mean(policy_means[name]) for name in baseline_names
    }
    baseline_levels = len(
        {round(value, 3) for value in (*baseline_means.values(), 1.0)}
    )
    if baseline_levels < 3:
        issues.append(
            f"offline baselines produce only {baseline_levels} distinct mean-score levels"
        )
    nonreference_max = max(baseline_means.values())
    if nonreference_max > 0.85:
        issues.append(f"a fixed error policy averages {nonreference_max:.3f}")
    return ValidationReport(
        tasks=len(tasks),
        reference_full=reference_full,
        process_reward_equalities=reward_equalities,
        process_rewards_with_event_links=linked_rewards,
        deterministic_rescores=rescores,
        global_single_item_checks=global_single_checks,
        independent_cart_checks=cart_checks,
        bounded_cart_query_checks=bounded_cart_checks,
        mutation_runs=len(mutation_scores),
        score_buckets=buckets,
        nonreference_mean_max=nonreference_max,
        scenarios_with_four_scores=scenarios_with_four,
        scenarios_with_five_error_modes=scenarios_with_five_modes,
        baseline_means=baseline_means,
        baseline_score_levels=baseline_levels,
        passed=not issues,
        issues=tuple(issues),
    )
