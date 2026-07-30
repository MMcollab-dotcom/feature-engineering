"""Gross-constrained quadratic optimizer for smoothed portfolio entries."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


@dataclass(frozen=True, slots=True)
class QuadraticOptimisation:
    allocation: np.ndarray
    dual: np.ndarray
    expected_return: float
    turnover: float
    execution_cost: float
    future_decay_cost: float
    objective_value: float
    iterations: int
    gross_constraint_active: bool


def optimise_linear_position(
    *,
    expected_returns: np.ndarray,
    older_position: np.ndarray,
    position_map: np.ndarray,
    pretrade_position: np.ndarray,
    market_beta: np.ndarray,
    linear_execution_cost: float,
    future_decay_cost: float,
    quadratic_penalty: float,
    gross_budget: float,
    initial_dual: np.ndarray,
    tolerance: float = 1.0e-10,
) -> QuadraticOptimisation:
    """Optimize a desired portfolio under beta neutrality and a hard gross cap."""

    mu = np.asarray(expected_returns, dtype="float64")
    older = np.asarray(older_position, dtype="float64")
    matrix = np.asarray(position_map, dtype="float64")
    pretrade = np.asarray(pretrade_position, dtype="float64")
    beta = np.asarray(market_beta, dtype="float64")
    dual = np.asarray(initial_dual, dtype="float64").copy()
    size = mu.size
    if (
        older.shape != (size,)
        or pretrade.shape != (size,)
        or beta.shape != (size,)
        or dual.shape != (2 * size,)
        or matrix.shape != (size, size)
    ):
        raise ValueError("optimizer inputs must share one square asset dimension")
    if gross_budget < 0.0 or not np.isfinite(gross_budget):
        raise ValueError("gross_budget must be finite and non-negative")
    if quadratic_penalty <= 0.0 or not np.isfinite(quadratic_penalty):
        raise ValueError("quadratic_penalty must be finite and positive")
    if linear_execution_cost < 0.0 or not np.isfinite(linear_execution_cost):
        raise ValueError("linear_execution_cost must be finite and non-negative")
    if future_decay_cost < 0.0 or not np.isfinite(future_decay_cost):
        raise ValueError("future_decay_cost must be finite and non-negative")

    offset = older - pretrade
    if gross_budget == 0.0:
        return _result(
            allocation=np.zeros_like(mu),
            dual=np.zeros_like(dual),
            mu=mu,
            offset=offset,
            matrix=matrix,
            execution_cost_rate=linear_execution_cost,
            future_decay_cost_rate=future_decay_cost,
            quadratic_penalty=quadratic_penalty,
            iterations=0,
            gross_constraint_active=True,
        )

    beta_norm_squared = float(beta @ beta)
    projection = np.eye(size, dtype="float64")
    if beta_norm_squared > np.finfo("float64").eps:
        projection -= np.outer(beta, beta) / beta_norm_squared
    projected_mu = projection @ mu
    projected_matrix = np.concatenate((matrix @ projection, projection), axis=0)
    projected_offset = np.concatenate((offset, np.zeros(size)))
    dual_bound = np.concatenate(
        (
            np.full(size, linear_execution_cost),
            np.full(size, future_decay_cost),
        )
    )

    iterations = 0
    if linear_execution_cost == 0.0 and future_decay_cost == 0.0:
        allocation = projected_mu / quadratic_penalty
        dual.fill(0.0)
    else:
        np.clip(dual, -dual_bound, dual_bound, out=dual)
        dual, iterations = _identify_dual(
            matrix=projected_matrix,
            expected_returns=projected_mu,
            offset=projected_offset,
            quadratic_penalty=quadratic_penalty,
            bound=dual_bound,
            initial=dual,
        )
        if projected_matrix.shape[0] <= size:
            dual = _polish_dual(
                matrix=projected_matrix,
                expected_returns=projected_mu,
                offset=projected_offset,
                quadratic_penalty=quadratic_penalty,
                bound=dual_bound,
                initial=dual,
            )
        allocation = (projected_mu - projected_matrix.T @ dual) / quadratic_penalty

    constraint_tolerance = max(1.0e-8, 10.0 * tolerance)
    gross = float(np.abs(allocation).sum())
    gross_constraint_active = gross > gross_budget + constraint_tolerance
    if gross_constraint_active:
        allocation, capped_iterations = _solve_gross_constrained_qp(
            expected_returns=mu,
            offset=offset,
            matrix=matrix,
            market_beta=beta,
            execution_cost=linear_execution_cost,
            future_decay_cost=future_decay_cost,
            quadratic_penalty=quadratic_penalty,
            gross_budget=gross_budget,
            initial_allocation=allocation * (gross_budget / gross),
        )
        iterations += capped_iterations

    gross = float(np.abs(allocation).sum())
    beta_exposure = float(beta @ allocation)
    beta_norm = float(np.sqrt(beta_norm_squared))
    if gross > gross_budget + constraint_tolerance:
        raise RuntimeError("quadratic portfolio optimisation violated gross constraint")
    if beta_norm and abs(beta_exposure) > constraint_tolerance * beta_norm:
        raise RuntimeError("quadratic portfolio optimisation violated beta neutrality")
    return _result(
        allocation=allocation,
        dual=dual,
        mu=mu,
        offset=offset,
        matrix=matrix,
        execution_cost_rate=linear_execution_cost,
        future_decay_cost_rate=future_decay_cost,
        quadratic_penalty=quadratic_penalty,
        iterations=iterations,
        gross_constraint_active=gross_constraint_active,
    )


def _result(
    *,
    allocation: np.ndarray,
    dual: np.ndarray,
    mu: np.ndarray,
    offset: np.ndarray,
    matrix: np.ndarray,
    execution_cost_rate: float,
    future_decay_cost_rate: float,
    quadratic_penalty: float,
    iterations: int,
    gross_constraint_active: bool,
) -> QuadraticOptimisation:
    trade = offset + matrix @ allocation
    expected_return = float(mu @ allocation)
    turnover = float(np.abs(trade).sum())
    execution_cost = execution_cost_rate * turnover
    future_cost = future_decay_cost_rate * float(np.abs(allocation).sum())
    quadratic_cost = 0.5 * quadratic_penalty * float(allocation @ allocation)
    return QuadraticOptimisation(
        allocation=allocation,
        dual=dual,
        expected_return=expected_return,
        turnover=turnover,
        execution_cost=execution_cost,
        future_decay_cost=future_cost,
        objective_value=(
            expected_return - quadratic_cost - execution_cost - future_cost
        ),
        iterations=iterations,
        gross_constraint_active=gross_constraint_active,
    )


def _solve_gross_constrained_qp(
    *,
    expected_returns: np.ndarray,
    offset: np.ndarray,
    matrix: np.ndarray,
    market_beta: np.ndarray,
    execution_cost: float,
    future_decay_cost: float,
    quadratic_penalty: float,
    gross_budget: float,
    initial_allocation: np.ndarray,
) -> tuple[np.ndarray, int]:
    size = expected_returns.size
    identity = np.eye(size, dtype="float64")
    zeros = np.zeros((size, size), dtype="float64")
    absolute_rows = np.concatenate(
        (
            np.concatenate((-matrix, identity, zeros), axis=1),
            np.concatenate((matrix, identity, zeros), axis=1),
            np.concatenate((-identity, zeros, identity), axis=1),
            np.concatenate((identity, zeros, identity), axis=1),
        ),
        axis=0,
    )
    absolute_bounds = np.concatenate((offset, -offset, np.zeros(2 * size)))
    gross_row = np.concatenate((np.zeros(2 * size), np.ones(size)))[None, :]
    beta_row = np.concatenate((market_beta, np.zeros(2 * size)))[None, :]
    initial_trade = np.abs(offset + matrix @ initial_allocation)
    initial = np.concatenate(
        (initial_allocation, initial_trade, np.abs(initial_allocation))
    )

    def objective(values: np.ndarray) -> float:
        allocation = values[:size]
        turnover = values[size : 2 * size]
        return float(
            0.5 * quadratic_penalty * (allocation @ allocation)
            - expected_returns @ allocation
            + execution_cost * turnover.sum()
            + future_decay_cost * values[2 * size :].sum()
        )

    def gradient(values: np.ndarray) -> np.ndarray:
        allocation = values[:size]
        return np.concatenate(
            (
                quadratic_penalty * allocation - expected_returns,
                np.full(size, execution_cost),
                np.full(size, future_decay_cost),
            )
        )

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(
            np.concatenate((np.full(size, -np.inf), np.zeros(2 * size))),
            np.full(3 * size, np.inf),
        ),
        constraints=(
            LinearConstraint(absolute_rows, absolute_bounds, np.inf),
            LinearConstraint(gross_row, -np.inf, gross_budget),
            LinearConstraint(beta_row, 0.0, 0.0),
        ),
        options={"ftol": 1.0e-11, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"gross-constrained QP failed: {result.message}")
    return np.asarray(result.x[:size], dtype="float64"), int(result.nit)


def _identify_dual(
    *,
    matrix: np.ndarray,
    expected_returns: np.ndarray,
    offset: np.ndarray,
    quadratic_penalty: float,
    bound: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, int]:
    spectral_norm = float(np.linalg.norm(matrix, ord=2))
    if spectral_norm <= np.finfo("float64").eps:
        return np.clip(initial, -bound, bound), 0
    step_size = quadratic_penalty / spectral_norm**2
    dual = np.clip(initial, -bound, bound)
    accelerated = dual.copy()
    momentum = 1.0
    convergence_tolerance = max(1.0e-14, float(np.max(bound)) * 1.0e-8)
    for iteration in range(1, 51):
        transformed = matrix.T @ accelerated - expected_returns
        gradient = matrix @ transformed / quadratic_penalty - offset
        updated = np.clip(accelerated - step_size * gradient, -bound, bound)
        if float(np.max(np.abs(updated - dual))) <= convergence_tolerance:
            return updated, iteration
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        accelerated = updated + (momentum - 1.0) / next_momentum * (updated - dual)
        dual = updated
        momentum = next_momentum
    return dual, iteration


def _polish_dual(
    *,
    matrix: np.ndarray,
    expected_returns: np.ndarray,
    offset: np.ndarray,
    quadratic_penalty: float,
    bound: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    initial = np.clip(initial, -bound, bound)
    initial_q = (expected_returns - matrix.T @ initial) / quadratic_penalty
    initial_trade = offset + matrix @ initial_q
    inferred = np.where(
        np.abs(initial) >= 0.999 * bound,
        np.sign(initial).astype("int8"),
        0,
    )

    def solve(status: np.ndarray) -> np.ndarray | None:
        zero_trade = status == 0
        nonzero_trade = ~zero_trade
        polished = np.zeros(matrix.shape[0], dtype="float64")
        polished[nonzero_trade] = bound[nonzero_trade] * status[nonzero_trade]
        q0 = (
            expected_returns - matrix[nonzero_trade].T @ polished[nonzero_trade]
        ) / quadratic_penalty
        if np.any(zero_trade):
            equality = matrix[zero_trade]
            error = equality @ q0 + offset[zero_trade]
            multiplier = (
                quadratic_penalty
                * np.linalg.lstsq(equality @ equality.T, error, rcond=1.0e-12)[0]
            )
            q = q0 - equality.T @ multiplier / quadratic_penalty
            polished[zero_trade] = multiplier
        else:
            q = q0
        trade = offset + matrix @ q
        stationarity = quadratic_penalty * q - expected_returns + matrix.T @ polished
        if np.any(np.abs(polished) > bound + 1.0e-10):
            return None
        if np.any(zero_trade) and np.max(np.abs(trade[zero_trade])) > 1.0e-9:
            return None
        if (
            np.any(nonzero_trade)
            and np.min(status[nonzero_trade] * trade[nonzero_trade]) < -1.0e-9
        ):
            return None
        if np.max(np.abs(stationarity)) > 1.0e-10:
            return None
        return polished

    polished = solve(inferred)
    if polished is not None:
        return polished
    alternatives: dict[int, int] = {}
    for index, state in enumerate(inferred):
        if state:
            alternatives[index] = 0
        else:
            trade_sign = int(np.sign(initial_trade[index]))
            if trade_sign:
                alternatives[index] = trade_sign
    changed_indices = tuple(alternatives)
    for change_count in range(1, len(changed_indices) + 1):
        for changed in combinations(changed_indices, change_count):
            candidate = inferred.copy()
            for index in changed:
                candidate[index] = alternatives[index]
            polished = solve(candidate)
            if polished is not None:
                return polished
    for candidate in product((-1, 0, 1), repeat=matrix.shape[0]):
        polished = solve(np.asarray(candidate, dtype="int8"))
        if polished is not None:
            return polished
    raise RuntimeError("could not identify a valid turnover-dual active set")


__all__ = ["QuadraticOptimisation", "optimise_linear_position"]
