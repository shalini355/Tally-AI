"""Bounded parallel execution for blocking structured LLM evaluations."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    from .llm_matcher import MatchDecision
except ImportError:
    from llm_matcher import MatchDecision


@dataclass(frozen=True)
class CandidatePair:
    """One ERP row and one bank row sent to the evaluator."""

    erp_id: str
    bank_ref: str
    erp_row: Mapping[str, Any]
    bank_row: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateEvaluation:
    """A candidate pair together with its validated model decision."""

    candidate: CandidatePair
    decision: MatchDecision


def _evaluate_one(
    candidate: CandidatePair,
    evaluator: Callable[..., MatchDecision],
    provider: str,
    model: str | None,
) -> CandidateEvaluation:
    try:
        decision = evaluator(
            candidate.erp_row,
            candidate.bank_row,
            provider=provider,
            model=model,
        )
    except Exception as error:
        decision = MatchDecision(
            is_match=False,
            confidence_score=0.0,
            match_type="no_match",
            reasoning=f"AI provider unavailable after fallback: {str(error)[:180]}",
        )
    return CandidateEvaluation(candidate=candidate, decision=decision)


async def _evaluate_async(
    candidates: list[CandidatePair],
    evaluator: Callable[..., MatchDecision],
    provider: str,
    model: str | None,
    max_workers: int,
) -> list[CandidateEvaluation]:
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            loop.run_in_executor(
                executor,
                _evaluate_one,
                candidate,
                evaluator,
                provider,
                model,
            )
            for candidate in candidates
        ]
        return list(await asyncio.gather(*futures))


def evaluate_candidates_parallel(
    candidates: list[CandidatePair],
    evaluator: Callable[..., MatchDecision],
    *,
    provider: str,
    model: str | None,
    max_workers: int = 8,
) -> list[CandidateEvaluation]:
    """Evaluate candidates concurrently with bounded blocking-worker fan-out.

    The function remains synchronous at the pipeline boundary, so it can be called
    from Streamlit, a CLI, or a synchronous task worker. The provider client call is
    blocking, therefore threads avoid blocking the event loop while asyncio manages
    completion and preserves result ordering.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if not candidates:
        return []
    return asyncio.run(
        _evaluate_async(candidates, evaluator, provider, model, max_workers)
    )
