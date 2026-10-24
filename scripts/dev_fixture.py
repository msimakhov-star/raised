"""Development fixture for the UI: two synthetic charts with their true tables, one planted disagreement.
Never shipped as a demo sample (index.json marks it dev-only and the banner says so).

    python -m scripts.dev_fixture
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYN = ROOT / "bench" / "data" / "synth" / "probe"
OUT = ROOT / "data" / "samples" / "dev"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figs = []
    for k, idx in enumerate(["0000", "0003"]):
        truth = json.loads((SYN / f"{idx}.json").read_text())
        fid = f"f{k + 1}"
        shutil.copy(SYN / f"{idx}.jpg", OUT / f"{fid}-chart.jpg")
        sem, table = truth["semantics"], truth["table"]
        name = next(iter(table["series"]))
        points = []
        status, reason = "verified", "Both readings agree and the redraw matches the original."
        if k == 1:
            v = table["series"][name][2]
            points.append({"series": name, "x": table["x"][2], "value": v, "reading_a": round(v * 1.12, 2), "reading_b": v,
                           "status": "disputed", "note": "Pixel match prefers the measured value."})
            status, reason = "review", "1 value where the two readings disagree."
        figs.append({"id": fid, "page": k + 1, "figure_number": f"9.{k + 1}", "caption": sem.get("title", ""), "chart_type": sem["chart_type"],
                     "title": sem.get("title", ""), "semantics": sem, "image": f"/samples/dev/{fid}-chart.jpg", "status": status, "reason": reason,
                     "iou": 0.91, "table": table, "points": points,
                     "description": {"long_description": "Development fixture. The real description is written by Nemotron 3 Ultra and checked by the number guard.",
                                     "exploration_guide": ["Start at the first point.", "Listen for the highest pitch."],
                                     "audit": [{"claim_quote": "the series rises over the period", "verdict": "unverifiable", "fact_ids": []}],
                                     "alt_text": sem.get("title", "")}})
    result = {"job": "sample-dev", "recorded": True, "document": {"name": "dev fixture"}, "figures": figs,
              "cost": {"total_usd": 0, "calls": []}}
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    idx = ROOT / "data" / "samples" / "index.json"
    entries = [e for e in (json.loads(idx.read_text()) if idx.exists() else []) if e["id"] != "dev"]
    entries.append({"id": "dev", "title": "Development fixture", "subtitle": "Synthetic charts, not a real run", "thumb": "/samples/dev/f1-chart.jpg",
                    "note": "development fixture built from synthetic charts with known values, not a model run"})
    idx.write_text(json.dumps(entries, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
