"""CSV parsing and contract checks never execute user-supplied Python or SQL."""

import csv
import io
import math
from collections import Counter
from datetime import date

from app.schemas import Contract

ALLOWED = {"trim_whitespace", "drop_exact_duplicates", "escape_formulas"}


def parse(text: str) -> tuple[list[str], list[list[str]]]:
    if len(text.encode()) > 1000000:
        raise ValueError("CSV exceeds 1 MB")
    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error:
        raise ValueError("Malformed CSV") from None
    if not records or not records[0] or any(not h.strip() for h in records[0]):
        raise ValueError("CSV requires nonempty column names")
    header, rows = records[0], records[1:]
    if len(set(header)) != len(header) or len(header) > 50:
        raise ValueError("Duplicate headers or more than 50 columns")
    if len(rows) > 5000:
        raise ValueError("CSV exceeds 5000 rows")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("CSV row width differs from header")
    return header, rows


def analyze(text: str, contract: Contract) -> dict:
    header, rows = parse(text)
    issues = []

    def issue(code, count, column=None):
        if count:
            issues.append({"code": code, "count": count, "column": column})

    names = {c.name for c in contract.columns}
    issue("missing_columns", len(names - set(header)))
    if not contract.allow_extra_columns:
        issue("unexpected_columns", len(set(header) - names))
    issue("exact_duplicate_rows", len(rows) - len({tuple(r) for r in rows}))
    issue("whitespace_cells", sum(v != v.strip() for row in rows for v in row))
    issue("spreadsheet_formula_headers", count_formula_headers(header))
    issue(
        "spreadsheet_formula_cells",
        sum(
            v.lstrip().startswith(("=", "+", "-", "@")) and not _is_number(v)
            for row in rows
            for v in row
        ),
    )
    for column in contract.columns:
        if column.name not in header:
            continue
        values = [row[header.index(column.name)] for row in rows]
        present = [v for v in values if v.strip()]
        issue("null_values", len(values) - len(present) if column.required else 0, column.name)
        if column.unique:
            issue(
                "duplicate_key_values", sum(n - 1 for n in Counter(present).values()), column.name
            )
        bad_type, bad_range = 0, 0
        for value in present:
            try:
                if column.dtype == "integer":
                    int(value)
                elif column.dtype == "date":
                    date.fromisoformat(value)
                elif column.dtype == "number":
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise ValueError()
                if column.dtype in {"number", "integer"}:
                    n = float(value)
                    bad_range += int(
                        (column.minimum is not None and n < column.minimum)
                        or (column.maximum is not None and n > column.maximum)
                    )
            except (ValueError, OverflowError):
                bad_type += 1
        issue("type_errors", bad_type, column.name)
        issue("range_errors", bad_range, column.name)
        issue(
            "enum_errors",
            sum(v not in column.allowed_values for v in present) if column.allowed_values else 0,
            column.name,
        )
    return {"rows": len(rows), "columns": header, "issues": issues, "passes_contract": not issues}


def _is_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def count_formula_headers(header: list[str]) -> int:
    """Count formula-like column names without changing contract identifiers.

    Use the same prefix and finite-number policy as data-cell escaping. Header
    repairs require explicit schema changes rather than an automatic rename.
    """
    return sum(
        value.lstrip().startswith(("=", "+", "-", "@")) and not _is_number(value)
        for value in header
    )


def proposed_operations(profile: dict) -> list[str]:
    codes = {i["code"] for i in profile["issues"]}
    result = []
    if "whitespace_cells" in codes:
        result.append("trim_whitespace")
    if "exact_duplicate_rows" in codes:
        result.append("drop_exact_duplicates")
    if "spreadsheet_formula_cells" in codes:
        result.append("escape_formulas")
    return result


def repair(text: str, operations: list[str]) -> tuple[str, dict]:
    if len(operations) != len(set(operations)) or not set(operations) <= ALLOWED:
        raise ValueError("Unsupported repair operation")
    header, rows = parse(text)
    before = len(rows)
    changed = 0
    if "trim_whitespace" in operations:
        changed += sum(v != v.strip() for row in rows for v in row)
        rows = [[v.strip() for v in row] for row in rows]
    if "drop_exact_duplicates" in operations:
        rows = [list(row) for row in dict.fromkeys(tuple(row) for row in rows)]
    if "escape_formulas" in operations:
        for row in rows:
            for i, value in enumerate(row):
                if value.lstrip().startswith(("=", "+", "-", "@")) and not _is_number(value):
                    row[i] = "'" + value
                    changed += 1
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue(), {
        "rows_before": before,
        "rows_after": len(rows),
        "rows_removed": before - len(rows),
        "cells_changed": changed,
    }
