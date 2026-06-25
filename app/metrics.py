"""Lightweight markdown stats, tuned for comparing invoice line-item tables."""
import re


def markdown_table_stats(md: str) -> dict:
    """Count tables and rows in markdown.

    For invoices the useful signal is ``max_table_rows`` (~ number of captured
    line items) and ``num_tables`` (a single line-item table getting split into
    several is a structure break, the classic marker failure on long invoices).
    """
    if not md:
        return {
            "chars": 0, "lines": 0, "num_tables": 0,
            "total_table_rows": 0, "max_table_rows": 0,
        }

    lines = md.splitlines()
    tables = []  # rows per contiguous table block
    cur = 0
    for ln in lines:
        s = ln.strip()
        is_row = s.startswith("|") and s.count("|") >= 2
        if is_row:
            cur += 1
        elif cur:
            tables.append(cur)
            cur = 0
    if cur:
        tables.append(cur)

    # Each markdown table has a header row + a |---|---| separator; data rows ~= rows - 2.
    data_rows = [max(0, t - 2) for t in tables]

    return {
        "chars": len(md),
        "lines": len(lines),
        "num_tables": len(tables),
        "total_table_rows": sum(data_rows),
        "max_table_rows": max(data_rows) if data_rows else 0,
    }


_HEADING_RE = re.compile(r"^#{1,6}\s")


def quick_overview(md: str) -> dict:
    """A couple more cheap signals for eyeballing output quality."""
    stats = markdown_table_stats(md)
    if md:
        stats["headings"] = sum(1 for ln in md.splitlines() if _HEADING_RE.match(ln))
    else:
        stats["headings"] = 0
    return stats
