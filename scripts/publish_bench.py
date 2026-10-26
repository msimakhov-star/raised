"""Turn bench/results/results.json into data/bench/results.json for the /bench page. Numbers are copied, never edited."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = json.loads((ROOT / "bench" / "results" / "results.json").read_text())
rows, by_type = [], []
for key, rec in sorted(src.items()):
    config, split = key.split("-", 1)
    s = rec["summary"]
    rows.append({"config": config, "split": split, "n": s["n"], "rms_f1": s.get("rms_f1", 0) / 100 if s.get("rms_f1") is not None else None,
                 "median_error": s.get("median_err_pct") / 100 if s.get("median_err_pct") is not None else None,
                 "within_2pct": s.get("within_2pct"), "usd_per_chart": s.get("usd_per_figure"), "seconds_per_chart": s.get("seconds_per_figure")})
    if split == "synth-bench":
        for t, x in s.get("by_chart_type", {}).items():
            by_type.append({"config": config, "type": t.replace("_", " "), "median_error": x["median_err_pct"] / 100})
best = next((r for r in rows if r["split"] == "synth-bench"), None)
headline = (f"On {best['n']} synthetic charts with known values, pixel measurement alone reads values with a median error of "
            f"{100 * best['median_error']:.2f}% of the axis range. Model configurations are added when they have been run.") if best else ""
out = ROOT / "data" / "bench"
out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(json.dumps({"generated": time.strftime("%d %b %Y"), "headline": headline, "rows": rows, "by_type": by_type}, indent=1))
print(headline)
