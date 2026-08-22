"""Generate deliberately messy synthetic data for reconciliation testing."""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


SEED = 20260822
ERP_RECORD_COUNT = 60
BANK_RECORD_COUNT = 60
MATCHED_RECORD_COUNT = 52

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ERP_PATH = DATA_DIR / "erp_ledger.csv"
BANK_PATH = DATA_DIR / "bank_statement.csv"
GROUND_TRUTH_PATH = DATA_DIR / ".ground_truth_mappings.json"

MERCHANTS = [
    ("Razorpay Software", "RZPAY PVT LTD"),
    ("Acme Retail India", "ACME RETAIL"),
    ("Nimbus Cloud Services", "NIMBUS CLOUD"),
    ("Green Basket Foods", "GREEN BASKET"),
    ("Northstar Electronics", "NORTHSTAR ELEC"),
    ("Urban Mobility Labs", "URBAN MOBILITY"),
    ("Mango Tree Wellness", "MANGO TREE WELLNESS"),
    ("Bluebird Education", "BLUEBIRD EDU"),
]
CURRENCIES = ["INR", "INR", "INR", "INR", "USD"]


def money(value: Decimal | float | int) -> float:
    """Round monetary values consistently before writing CSV files."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def make_erp_records(rng: random.Random) -> list[dict[str, object]]:
    records = []
    start_date = date(2026, 7, 1)

    for index in range(ERP_RECORD_COUNT):
        merchant_name, _ = MERCHANTS[index % len(MERCHANTS)]
        currency = CURRENCIES[index % len(CURRENCIES)]
        amount = Decimal(rng.randint(1_500, 125_000)) / 100
        tax_rate = Decimal("0.18") if currency == "INR" else Decimal("0.08")
        tax = (amount * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        records.append(
            {
                "erp_id": f"ERP-{index + 1:04d}",
                "date": (start_date + timedelta(days=index % 22)).isoformat(),
                "merchant_name": merchant_name,
                "amount": money(amount),
                "currency": currency,
                "tax_line_item": money(tax),
            }
        )

    return records


def make_bank_records(
    erp_records: list[dict[str, object]], rng: random.Random
) -> tuple[list[dict[str, object]], dict[str, object]]:
    bank_records: list[dict[str, object]] = []
    matches: list[dict[str, object]] = []
    erp_exceptions: list[str] = []
    bank_exceptions: list[str] = []
    start_date = date(2026, 7, 2)

    for index, erp in enumerate(erp_records[:MATCHED_RECORD_COUNT]):
        _, bank_name = MERCHANTS[index % len(MERCHANTS)]
        match_type = ("exact", "fuzzy_name", "rounding_difference")[index % 3]
        if match_type == "exact":
            description = f"{erp['merchant_name']} {erp['erp_id']} SETTLEMENT"
            settlement_date = str(erp["date"])
            fee = Decimal("0.00")
            net_amount = Decimal(str(erp["amount"]))
        elif match_type == "fuzzy_name":
            description = f"{bank_name} SETTLEMENT"
            settlement_date = (
                date.fromisoformat(str(erp["date"])) + timedelta(days=1)
            ).isoformat()
            fee = (Decimal(str(erp["amount"])) * Decimal("0.0125")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            net_amount = (
                Decimal(str(erp["amount"]))
                + Decimal(str(erp["tax_line_item"]))
                - fee
            )
        else:
            description = f"{bank_name} / CARD PAYOUT"
            settlement_date = (
                date.fromisoformat(str(erp["date"])) + timedelta(days=2)
            ).isoformat()
            fee = (Decimal(str(erp["amount"])) * Decimal("0.01")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            net_amount = (
                Decimal(str(erp["amount"]))
                + Decimal(str(erp["tax_line_item"]))
                - fee
                + Decimal("0.01")
            )

        bank_ref = f"BNK-{index + 1:05d}"
        bank_records.append(
            {
                "bank_ref": bank_ref,
                "settlement_date": settlement_date,
                "description": description,
                "net_amount": money(net_amount),
                "fee_deducted": money(fee),
            }
        )
        matches.append(
            {
                "erp_id": erp["erp_id"],
                "bank_ref": bank_ref,
                "match_type": match_type,
                "expected_fee_deducted": money(fee),
            }
        )

    for index, erp in enumerate(erp_records[MATCHED_RECORD_COUNT:], start=1):
        erp_exceptions.append(str(erp["erp_id"]))
        bank_ref = f"BNK-UNMATCHED-{index:03d}"
        bank_records.append(
            {
                "bank_ref": bank_ref,
                "settlement_date": (
                    start_date + timedelta(days=30 + index)
                ).isoformat(),
                "description": f"UNKNOWN MERCHANT {index:02d}",
                "net_amount": money(Decimal(rng.randint(2_000, 90_000)) / 100),
                "fee_deducted": money(Decimal(rng.randint(0, 500)) / 100),
            }
        )
        bank_exceptions.append(bank_ref)

    rng.shuffle(bank_records)
    ground_truth = {
        "seed": SEED,
        "record_counts": {
            "erp": len(erp_records),
            "bank": len(bank_records),
            "matched_pairs": len(matches),
        },
        "matches": matches,
        "exceptions": {
            "erp_only": erp_exceptions,
            "bank_only": bank_exceptions,
        },
    }
    return bank_records, ground_truth


def generate_dataset(
    seed: int = SEED,
    output_dir: str | Path = DATA_DIR,
    erp_count: int = ERP_RECORD_COUNT,
    matched_count: int = MATCHED_RECORD_COUNT,
) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    erp_csv = output_path / "erp_ledger.csv"
    bank_csv = output_path / "bank_statement.csv"
    ground_truth_json = output_path / ".ground_truth_mappings.json"

    rng = random.Random(seed)
    erp_records = make_erp_records_with_count(rng, erp_count)
    bank_records, ground_truth = make_bank_records_with_count(erp_records, rng, matched_count, seed)

    pd.DataFrame(erp_records).to_csv(erp_csv, index=False)
    pd.DataFrame(bank_records).to_csv(bank_csv, index=False)
    ground_truth_json.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    print(f"Wrote {len(erp_records)} ERP records to {erp_csv}")
    print(f"Wrote {len(bank_records)} bank records to {bank_csv}")
    print(f"Wrote {len(ground_truth['matches'])} mappings to {ground_truth_json}")

    return erp_csv, bank_csv, ground_truth_json


def make_erp_records_with_count(rng: random.Random, count: int) -> list[dict[str, object]]:
    records = []
    start_date = date(2026, 7, 1)

    for index in range(count):
        merchant_name, _ = MERCHANTS[index % len(MERCHANTS)]
        currency = CURRENCIES[index % len(CURRENCIES)]
        amount = Decimal(rng.randint(1_500, 125_000)) / 100
        tax_rate = Decimal("0.18") if currency == "INR" else Decimal("0.08")
        tax = (amount * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        records.append(
            {
                "erp_id": f"ERP-{index + 1:04d}",
                "date": (start_date + timedelta(days=index % 22)).isoformat(),
                "merchant_name": merchant_name,
                "amount": money(amount),
                "currency": currency,
                "tax_line_item": money(tax),
            }
        )

    return records


def make_bank_records_with_count(
    erp_records: list[dict[str, object]],
    rng: random.Random,
    matched_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    bank_records: list[dict[str, object]] = []
    matches: list[dict[str, object]] = []
    erp_exceptions: list[str] = []
    bank_exceptions: list[str] = []
    start_date = date(2026, 7, 2)

    actual_matched = min(matched_count, len(erp_records))

    for index, erp in enumerate(erp_records[:actual_matched]):
        _, bank_name = MERCHANTS[index % len(MERCHANTS)]
        match_type = ("exact", "fuzzy_name", "rounding_difference")[index % 3]
        if match_type == "exact":
            description = f"{erp['merchant_name']} {erp['erp_id']} SETTLEMENT"
            settlement_date = str(erp["date"])
            fee = Decimal("0.00")
            net_amount = Decimal(str(erp["amount"]))
        elif match_type == "fuzzy_name":
            description = f"{bank_name} SETTLEMENT"
            settlement_date = (
                date.fromisoformat(str(erp["date"])) + timedelta(days=1)
            ).isoformat()
            fee = (Decimal(str(erp["amount"])) * Decimal("0.0125")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            net_amount = (
                Decimal(str(erp["amount"]))
                + Decimal(str(erp["tax_line_item"]))
                - fee
            )
        else:
            description = f"{bank_name} / CARD PAYOUT"
            settlement_date = (
                date.fromisoformat(str(erp["date"])) + timedelta(days=2)
            ).isoformat()
            fee = (Decimal(str(erp["amount"])) * Decimal("0.01")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            net_amount = (
                Decimal(str(erp["amount"]))
                + Decimal(str(erp["tax_line_item"]))
                - fee
                + Decimal("0.01")
            )

        bank_ref = f"BNK-{index + 1:05d}"
        bank_records.append(
            {
                "bank_ref": bank_ref,
                "settlement_date": settlement_date,
                "description": description,
                "net_amount": money(net_amount),
                "fee_deducted": money(fee),
            }
        )
        matches.append(
            {
                "erp_id": erp["erp_id"],
                "bank_ref": bank_ref,
                "match_type": match_type,
                "expected_fee_deducted": money(fee),
            }
        )

    for index, erp in enumerate(erp_records[actual_matched:], start=1):
        erp_exceptions.append(str(erp["erp_id"]))
        bank_ref = f"BNK-UNMATCHED-{index:03d}"
        bank_records.append(
            {
                "bank_ref": bank_ref,
                "settlement_date": (
                    start_date + timedelta(days=30 + index)
                ).isoformat(),
                "description": f"UNKNOWN MERCHANT {index:02d}",
                "net_amount": money(Decimal(rng.randint(2_000, 90_000)) / 100),
                "fee_deducted": money(Decimal(rng.randint(0, 500)) / 100),
            }
        )
        bank_exceptions.append(bank_ref)

    rng.shuffle(bank_records)
    ground_truth = {
        "seed": seed,
        "record_counts": {
            "erp": len(erp_records),
            "bank": len(bank_records),
            "matched_pairs": len(matches),
        },
        "matches": matches,
        "exceptions": {
            "erp_only": erp_exceptions,
            "bank_only": bank_exceptions,
        },
    }
    return bank_records, ground_truth


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Output directory")
    parser.add_argument("--records", type=int, default=ERP_RECORD_COUNT, help="Number of ERP records")
    parser.add_argument("--matched", type=int, default=MATCHED_RECORD_COUNT, help="Number of matched records")
    args = parser.parse_args()

    generate_dataset(
        seed=args.seed,
        output_dir=args.output_dir,
        erp_count=args.records,
        matched_count=args.matched,
    )


if __name__ == "__main__":
    main()

