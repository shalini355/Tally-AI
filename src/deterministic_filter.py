"""High-speed deterministic first pass for fintech reconciliation."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

import pandas as pd


ERP_REQUIRED_COLUMNS = {
    "erp_id",
    "date",
    "merchant_name",
    "amount",
    "currency",
    "tax_line_item",
}
BANK_REQUIRED_COLUMNS = {
    "bank_ref",
    "settlement_date",
    "description",
    "net_amount",
    "fee_deducted",
}


class DeterministicMatcher:
    """Match ERP and bank rows using exact embedded ID and amount rules."""

    def __init__(
        self,
        erp_path: str | Path,
        bank_path: str | Path,
    ) -> None:
        self.erp_path = Path(erp_path)
        self.bank_path = Path(bank_path)
        self.erp = pd.DataFrame()
        self.bank = pd.DataFrame()
        self.matched_pairs: list[dict[str, Any]] = []
        self.unresolved_erp = pd.DataFrame()
        self.unresolved_bank = pd.DataFrame()

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load both CSV files and validate their required columns."""
        erp = pd.read_csv(self.erp_path)
        bank = pd.read_csv(self.bank_path)
        self._validate_columns(erp, ERP_REQUIRED_COLUMNS, "Dataset A")
        self._validate_columns(bank, BANK_REQUIRED_COLUMNS, "Dataset B")

        erp["erp_id"] = erp["erp_id"].astype("string").str.strip()
        bank["description"] = bank["description"].astype("string").fillna("")
        erp["amount"] = pd.to_numeric(erp["amount"], errors="raise")
        bank["net_amount"] = pd.to_numeric(bank["net_amount"], errors="raise")
        self.erp = erp
        self.bank = bank
        return self.erp, self.bank

    def run_first_pass(self) -> list[dict[str, Any]]:
        """Return exact matches and store the remaining rows as unresolved pools."""
        if self.erp.empty and self.bank.empty:
            self.load_data()

        if self.erp.empty or self.bank.empty:
            self.matched_pairs = []
            self.unresolved_erp = self.erp.copy()
            self.unresolved_bank = self.bank.copy()
            return self.matched_pairs

        bank_indexed = self.bank.copy()
        erp_ids = self.erp["erp_id"].dropna().drop_duplicates().tolist()

        def extract_unambiguous_id(description: str) -> str | None:
            candidates = [
                erp_id
                for erp_id in erp_ids
                if re.search(
                    rf"(?<![A-Za-z0-9_-]){re.escape(erp_id)}(?![A-Za-z0-9_-])",
                    description,
                )
            ]
            return candidates[0] if len(candidates) == 1 else None

        bank_indexed["_erp_id"] = bank_indexed["description"].map(
            extract_unambiguous_id
        )
        bank_indexed["_amount"] = bank_indexed["net_amount"]

        erp_key_counts = self.erp.groupby(["erp_id", "amount"], dropna=False).size()
        bank_key_counts = bank_indexed.groupby(["_erp_id", "_amount"], dropna=False).size()
        unique_erp_keys = erp_key_counts[erp_key_counts == 1].index
        unique_bank_keys = bank_key_counts[bank_key_counts == 1].index

        candidates = (
            self.erp[self.erp.set_index(["erp_id", "amount"]).index.isin(unique_erp_keys)]
            .reset_index(names="_erp_row")
            .merge(
                bank_indexed[
                    bank_indexed.set_index(["_erp_id", "_amount"]).index.isin(unique_bank_keys)
                ].reset_index(names="_bank_row"),
                left_on=["erp_id", "amount"],
                right_on=["_erp_id", "_amount"],
                how="inner",
                sort=False,
            )
        )
        candidates = candidates.drop_duplicates(subset=["_erp_row", "_bank_row"])

        self.matched_pairs = [
            {
                "erp": row["erp_id"],
                "bank": row["bank_ref"],
                "erp_row": int(row["_erp_row"]),
                "bank_row": int(row["_bank_row"]),
                "amount": float(row["amount"]),
            }
            for row in candidates[
                ["erp_id", "bank_ref", "_erp_row", "_bank_row", "amount"]
            ].to_dict("records")
        ]

        matched_erp_rows = candidates["_erp_row"].astype(int).unique()
        matched_bank_rows = candidates["_bank_row"].astype(int).unique()
        self.unresolved_erp = self.erp.drop(index=matched_erp_rows).copy()
        self.unresolved_bank = self.bank.drop(index=matched_bank_rows).copy()
        return self.matched_pairs

    def print_false_positives(self, ground_truth_path: str | Path) -> list[dict[str, Any]]:
        """Print deterministic pairs absent from the exact ground-truth mappings."""
        with Path(ground_truth_path).open(encoding="utf-8") as ground_truth_file:
            ground_truth = json.load(ground_truth_file)
        expected = {
            (str(row["erp_id"]), str(row["bank_ref"]))
            for row in ground_truth.get("matches", [])
            if row.get("match_type") == "exact"
        }
        false_positives = [
            pair
            for pair in self.matched_pairs
            if (str(pair["erp"]), str(pair["bank"])) not in expected
        ]
        print(f"Deterministic false positives: {len(false_positives)}")
        for pair in false_positives:
            erp_row = self.erp.iloc[pair["erp_row"]]
            bank_row = self.bank.iloc[pair["bank_row"]]
            print(
                f"ERP {pair['erp']} amount={erp_row['amount']} -> "
                f"BANK {pair['bank']} description={bank_row['description']!r} "
                f"net_amount={bank_row['net_amount']} "
                f"trigger: exact ID={pair['erp']}, exact amount={pair['amount']}"
            )
        return false_positives

    def unresolved_pool(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return unmatched ERP and bank rows for semantic or fuzzy matching."""
        if self.erp.empty and self.bank.empty:
            self.run_first_pass()
        return self.unresolved_erp.copy(), self.unresolved_bank.copy()

    @staticmethod
    def _validate_columns(
        frame: pd.DataFrame, required: set[str], dataset_name: str
    ) -> None:
        missing = required.difference(frame.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")


if __name__ == "__main__":
    import argparse
    from generate_data import BANK_PATH, ERP_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path)
    args = parser.parse_args()
    matcher = DeterministicMatcher(ERP_PATH, BANK_PATH)
    pairs = matcher.run_first_pass()
    unresolved_erp, unresolved_bank = matcher.unresolved_pool()
    print(f"Deterministic matches: {len(pairs)}")
    print(f"Unresolved ERP rows: {len(unresolved_erp)}")
    print(f"Unresolved bank rows: {len(unresolved_bank)}")
    if args.ground_truth:
        matcher.print_false_positives(args.ground_truth)
