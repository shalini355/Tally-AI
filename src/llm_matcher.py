"""Structured LLM evaluation for unresolved reconciliation candidates."""

from __future__ import annotations

import json
import os
import time
from enum import Enum

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class MatchType(str, Enum):
    """Allowed explanations for a potential ERP-to-bank match."""

    EXACT = "exact"
    FUZZY_NAME_MATCH = "fuzzy_name_match"
    ROUNDING_DISCREPANCY = "rounding_discrepancy"
    NO_MATCH = "no_match"


class MatchDecision(BaseModel):
    """The validated, auditable decision returned by the matching agent."""

    model_config = ConfigDict(extra="forbid")

    is_match: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    match_type: MatchType
    reasoning: str = Field(min_length=1)


SYSTEM_PROMPT = """You are a conservative fintech reconciliation analyst.
Evaluate one possible match between an internal ERP transaction and a bank settlement.
Use business logic, not just literal string equality:
- Treat common merchant-name variations as possible matches, such as 'RZPAY' and
  'Razorpay', abbreviations, legal suffixes, and settlement descriptors.
- A total amount difference smaller than ₹1.00 can be a rounding discrepancy, but
  only classify it that way when the merchant identity and transaction context support
  the match. Do not invent a match solely because the amount is close.
- Consider the ERP amount, tax line item, bank net amount, and fee deducted together.
- Return no_match when identity or financial evidence is insufficient.
- Be conservative: confidence must reflect the evidence, and uncertain candidates
  should have is_match=false.
- Keep reasoning brief, factual, and single-line without quotes or escaping issues.
Return only the structured decision required by the response schema."""


def _get_api_key(provider: str) -> str | None:
    """Read credentials from local environment variables or Streamlit secrets."""
    key_name = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(key_name)
    if api_key:
        return api_key

    try:
        import streamlit as st

        return st.secrets.get(key_name)
    except (ImportError, FileNotFoundError, KeyError):
        return None


def _json_safe_row(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert dictionaries, Pandas rows, and scalar values into JSON-safe data."""
    values = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    return json.loads(json.dumps(values, default=str))


def _create_client(provider: str, model: str | None = None) -> Any:
    """Create an Instructor client lazily so importing this module makes no API call."""
    provider = provider.lower()
    if provider in {"gemini", "groq"}:
        import instructor
        from openai import OpenAI

        if provider == "gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            base_url = "https://api.groq.com/openai/v1"
        api_key = _get_api_key(provider)
        if not api_key:
            raise RuntimeError(f"{provider.upper()}_API_KEY is not set")
        return instructor.from_openai(
            OpenAI(api_key=api_key, base_url=base_url),
            mode=instructor.Mode.JSON,
        )
    raise ValueError("provider must be either 'gemini' or 'groq'")


def _is_rate_limit_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return "429" in error_text or "rate_limit" in error_text or "quota" in error_text


def evaluate_potential_match(
    erp_row: Mapping[str, Any] | Any,
    bank_row: Mapping[str, Any] | Any,
    *,
    provider: str = "groq",
    model: str | None = None,
    client: Any | None = None,
    max_retries: int = 3,
) -> MatchDecision:
    """Ask an Instructor client to evaluate one unresolved candidate pair.

    Pass an Instructor-compatible ``client`` in tests to avoid network calls. When no
    client is supplied, the selected provider key is read lazily from ``.env``.
    Automatically falls back to Groq if Gemini hits 429 rate limits.
    """
    default_models = {
        "gemini": "gemini-3.6-flash",
        "groq": "openai/gpt-oss-20b",
    }
    selected_provider = provider.lower()
    selected_model = model or default_models.get(selected_provider, "openai/gpt-oss-20b")
    erp_data = _json_safe_row(erp_row)
    bank_data = _json_safe_row(bank_row)
    user_prompt = (
        "Evaluate this candidate pair. The ERP row is Dataset A and the bank row is "
        "Dataset B. Keep reasoning under 20 words as a single sentence without quotes.\n\n"
        f"ERP row:\n{json.dumps(erp_data, indent=2, ensure_ascii=False)}\n\n"
        f"Bank row:\n{json.dumps(bank_data, indent=2, ensure_ascii=False)}"
    )

    max_attempts = 4
    fallback_used = False
    for attempt in range(1, max_attempts + 1):
        try:
            active_client = client if client is not None else _create_client(selected_provider, selected_model)
            return active_client.chat.completions.create(
                model=selected_model,
                response_model=MatchDecision,
                max_retries=max_retries,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                fallback_provider = "gemini" if selected_provider == "groq" else "groq"
                if client is None and not fallback_used and _get_api_key(fallback_provider):
                    selected_provider = fallback_provider
                    selected_model = default_models[selected_provider]
                    fallback_used = True
                    continue
                if attempt < max_attempts:
                    time.sleep(2.5 * attempt)
                    continue
            if attempt == max_attempts:
                if _is_rate_limit_error(exc):
                    raise RuntimeError(
                        f"{selected_provider.upper()} rate limit reached. "
                        "Add the other provider key in Streamlit Secrets, wait for the quota reset, "
                        "or use a provider with available quota."
                    ) from exc
                raise exc
    raise RuntimeError("Failed to evaluate potential match after retries")



