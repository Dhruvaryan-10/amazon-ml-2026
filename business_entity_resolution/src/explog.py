"""Append one row to outputs/experiments.csv (columns match the Google Sheet)."""
import csv, datetime, os
LOG = os.path.join(os.path.dirname(__file__), "..", "outputs", "experiments.csv")
COLS = ["exp_id", "date", "change", "blocking_recall", "avg_candidates", "cv_f05",
        "public_lb", "submission_hash", "notes"]


def log(change, notes="", **metrics):
    rows = list(csv.DictReader(open(LOG, encoding="utf-8"))) if os.path.exists(LOG) else []
    row = {c: "" for c in COLS}
    ids = [int(r["exp_id"].split("_")[-1]) for r in rows if r.get("exp_id", "").split("_")[-1].isdigit()]
    row.update(exp_id=f"exp_{(max(ids) + 1 if ids else 0):03d}", date=str(datetime.date.today()), change=change, notes=notes)
    row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})
    new = not os.path.exists(LOG) or os.path.getsize(LOG) == 0
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"logged {row['exp_id']}: {change}")
