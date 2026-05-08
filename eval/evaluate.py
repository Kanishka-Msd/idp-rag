import json
from pathlib import Path

GROUND = Path("eval/ground_truth")
PRED = Path("eval/predictions")
REPORT = Path("eval/reports")
REPORT.mkdir(parents=True, exist_ok=True)

FIELDS = ["invoice_number", "invoice_date", "vendor_name", "total_amount", "currency"]

def norm_str(x):
    if x is None:
        return None
    return " ".join(str(x).strip().lower().split())

def norm_currency(x):
    if x is None:
        return None
    x = norm_str(x)
    mapping = {"rs.": "inr", "rs": "inr", "₹": "inr", "inr": "inr", "usd": "usd", "eur": "eur"}
    return mapping.get(x, x)

def norm_amount(x):
    try:
        return round(float(x), 2)
    except:
        return None

def match_field(k, pred, truth):
    p = pred.get(k)
    t = truth.get(k)

    if k == "total_amount":
        return norm_amount(p) == norm_amount(t)
    if k == "currency":
        return norm_currency(p) == norm_currency(t)
    return norm_str(p) == norm_str(t)

def score_invoice(inv_id: str):
    gt_path = GROUND / f"{inv_id}.json"
    pr_path = PRED / f"{inv_id}.json"
    if not gt_path.exists() or not pr_path.exists():
        return None

    truth = json.loads(gt_path.read_text(encoding="utf-8"))
    pred = json.loads(pr_path.read_text(encoding="utf-8"))

    per_field = {}
    correct = 0
    for k in FIELDS:
        ok = match_field(k, pred, truth)
        per_field[k] = bool(ok)
        correct += 1 if ok else 0

    return {"invoice_id": inv_id, "accuracy": correct / len(FIELDS), "per_field": per_field}

def main():
    inv_ids = sorted([p.stem for p in GROUND.glob("*.json")])
    results = []

    for inv_id in inv_ids:
        r = score_invoice(inv_id)
        if r:
            results.append(r)

    if not results:
        print("No scored invoices. Make sure prediction + ground truth filenames match.")
        return

    avg = sum(r["accuracy"] for r in results) / len(results)

    report = {"num_invoices": len(results), "avg_accuracy": avg, "results": results}
    REPORT.joinpath("report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Invoices scored: {len(results)}")
    print(f"Average accuracy: {avg:.3f}")
    print("Saved report to eval/reports/report.json")

if __name__ == "__main__":
    main()