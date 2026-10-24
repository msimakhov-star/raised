"""Fetch a stratified sample of the ChartQA test split into bench/data/chartqa/ (gitignored, never committed).

    python -m bench.fetch_chartqa [--n 200] [--seed 0] [--force]

Source: Hugging Face dataset ahmed-masry/ChartQA, file "ChartQA Dataset.zip" (875 MB; also the layout described in
github.com/vis-nlp/ChartQA README: <split>/png, <split>/tables, <split>/annotations). We never download the zip: its
index is read and only the members we need are fetched with HTTP range requests (about 10 MB for 200 charts).
Licence: GPL-3.0 on both the Hugging Face dataset card and the GitHub repo (checked 2026-10-24). The files stay in a
gitignored cache and are not redistributed; Raised itself is MIT.

Sources, from the test split's file names (checked by eye on images: OWID logo, "PEW RESEARCH CENTER" footer, OECD
names): two_col_* / multi_col_* are Statista (1207 charts), 14-digit names are Our World in Data (143), short numeric
names are Pew (123), OECD_* are OECD (36). The README warns that chart annotations are noisy (Pew ones were
hand-made); we use only the CSV tables as truth and the annotation's "type" (v_bar, h_bar, line, pie) as chart type.

Outputs: index.json (zip member offsets), meta.json (every test chart's CSV and type), png/<id>.png, sample.json.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import random
import re
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from raised.model import Table, parse_number

ROOT = Path(__file__).resolve().parent / "data" / "chartqa"
URL = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
PREFIX = "ChartQA Dataset/test/"
TYPES = {"v_bar": "bar", "h_bar": "hbar", "line": "line", "pie": "pie"}  # annotation "type" values per the README


def source_of(chart_id: str) -> str:
    if chart_id.startswith("OECD"):
        return "oecd"
    if chart_id.startswith(("two_col", "multi_col")):
        return "statista"
    return "owid" if re.fullmatch(r"\d{14}", chart_id) else "pew"


# ---------------------------------------------------------------- reading members of the remote zip


class Remote:
    """Range reads on the zip. The signed CDN URL expires, so a 403 re-resolves it once."""

    def __init__(self):
        self.client = httpx.Client(follow_redirects=True, timeout=120)
        self._resolve()

    def _resolve(self):
        r = self.client.head(URL)
        r.raise_for_status()
        self.url, self.size = str(r.url), int(r.headers["content-length"])

    def get(self, start: int, end: int) -> bytes:
        """Bytes [start, end), end clamped to the file size."""
        end = min(end, self.size)
        for attempt in (0, 1):
            r = self.client.get(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
            if r.status_code == 403 and attempt == 0:
                self._resolve()
                continue
            r.raise_for_status()
            return r.content

    def as_file(self):
        """A file object over the zip for zipfile (central directory only: a few large reads)."""
        remote = self

        class F(io.RawIOBase):
            pos = 0

            def seekable(self): return True
            def readable(self): return True
            def tell(self): return self.pos

            def seek(self, off, whence=0):
                self.pos = {0: off, 1: self.pos + off, 2: remote.size + off}[whence]
                return self.pos

            def readinto(self, b):
                data = remote.get(self.pos, self.pos + len(b)) if self.pos < remote.size else b""
                b[:len(data)] = data
                self.pos += len(data)
                return len(data)

        return io.BufferedReader(F(), 1 << 20)


def load_index(remote: Remote) -> dict[str, list[int]]:
    """{member name: [local header offset, compressed size, compression]} for the test split (cached)."""
    p = ROOT / "index.json"
    if p.exists():
        return json.loads(p.read_text())
    import zipfile
    zf = zipfile.ZipFile(remote.as_file())
    idx = {i.filename: [i.header_offset, i.compress_size, i.compress_type] for i in zf.infolist()
           if i.filename.startswith(PREFIX) and not i.is_dir()}
    p.write_text(json.dumps(idx))
    return idx


def fetch_members(remote: Remote, idx: dict, names: list[str], gap: int, workers: int = 8) -> dict[str, bytes]:
    """Contents of the named members. Members closer than `gap` bytes share one range request."""
    ordered = sorted(names, key=lambda n: idx[n][0])
    groups: list[list[str]] = []
    for n in ordered:
        if groups and idx[n][0] - (idx[groups[-1][-1]][0] + idx[groups[-1][-1]][1]) <= gap:
            groups[-1].append(n)
        else:
            groups.append([n])

    def one(group: list[str]) -> dict[str, bytes]:
        first, last = idx[group[0]][0], idx[group[-1]]
        blob = remote.get(first, last[0] + 30 + len(group[-1].encode()) + 1024 + last[1])  # +1 KB for the local extra field
        out = {}
        for n in group:
            off, csize, method = idx[n]
            o = off - first
            fnlen, extralen = int.from_bytes(blob[o + 26:o + 28], "little"), int.from_bytes(blob[o + 28:o + 30], "little")
            raw = blob[o + 30 + fnlen + extralen:o + 30 + fnlen + extralen + csize]
            assert len(raw) == csize, f"short read for {n}"
            out[n] = zlib.decompress(raw, -15) if method == 8 else raw
        return out

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for part in ex.map(one, groups) for k, v in part.items()}


# ---------------------------------------------------------------- tables


def csv_to_table(text: str) -> tuple[Table, str] | None:
    """First column = row labels (x), every other column = one series; (table, header of the first column).
    None when no numeric series or fewer than two rows remain."""
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    if len(rows) < 3:
        return None
    header, body = rows[0], rows[1:]
    series: dict[str, list] = {}
    for j in range(1, len(header)):
        col = [parse_number(r[j]) if j < len(r) else None for r in body]
        if any(v is not None for v in col):
            name = header[j]
            while name in series:
                name += " (2)"
            series[name] = col
    return (Table([r[0] for r in body], series), header[0]) if series else None


# ---------------------------------------------------------------- selection


def quotas(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Equal share per group, small groups give their unused share to the larger ones."""
    out, left = {}, n
    for i, k in enumerate(sorted(sizes, key=sizes.get)):
        out[k] = min(sizes[k], left // (len(sizes) - i))
        left -= out[k]
    return out


def allocate(sizes: dict[str, int], q: int) -> dict[str, int]:
    """Proportional split of q over groups (largest remainder)."""
    tot = sum(sizes.values())
    raw = {k: q * v / tot for k, v in sizes.items()}
    out = {k: int(r) for k, r in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:q - sum(out.values())]:
        out[k] += 1
    return out


def select(cands: list[dict], n: int, seed: int) -> list[dict]:
    """n charts: equal per source (capped by availability), proportional over chart types inside a source,
    seeded. The result alternates between sources so any prefix (--limit) stays balanced."""
    rng = random.Random(seed)
    by_src: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in cands:
        by_src[c["source"]][c["chart_type"]].append(c)
    picked = {}
    for src, q in quotas({s: sum(map(len, t.values())) for s, t in by_src.items()}, n).items():
        take = allocate({t: len(v) for t, v in by_src[src].items()}, q)
        picked[src] = [c for t, v in sorted(by_src[src].items()) for c in rng.sample(sorted(v, key=lambda c: c["id"]), take[t])]
        rng.shuffle(picked[src])
    return [c for group in itertools.zip_longest(*picked.values()) for c in group if c]


# ---------------------------------------------------------------- main


def fetch(n: int = 200, seed: int = 0, force: bool = False) -> list[dict]:
    ROOT.mkdir(parents=True, exist_ok=True)
    sample_p = ROOT / "sample.json"
    if sample_p.exists() and not force:
        sample = json.loads(sample_p.read_text())
        if len(sample) == n and all((ROOT / c["image"]).exists() for c in sample):
            return sample  # idempotent: nothing to do
    remote = Remote()
    idx = load_index(remote)
    meta_p = ROOT / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
    else:
        ids = sorted(k[len(PREFIX + "tables/"):-4] for k in idx if k.startswith(PREFIX + "tables/") and k.endswith(".csv"))
        names = [f"{PREFIX}tables/{i}.csv" for i in ids] + [f"{PREFIX}annotations/{i}.json" for i in ids]
        blobs = fetch_members(remote, idx, names, gap=1 << 20)
        meta = {i: {"csv": blobs[f"{PREFIX}tables/{i}.csv"].decode("utf-8", "replace"),
                    "type": json.loads(blobs[f"{PREFIX}annotations/{i}.json"]).get("type")} for i in ids}
        meta_p.write_text(json.dumps(meta))
    cands = []
    for i, m in meta.items():
        parsed = csv_to_table(m["csv"])
        if parsed and m["type"] in TYPES:
            table, xh = parsed
            cands.append({"id": i, "source": source_of(i), "chart_type": TYPES[m["type"]], "x_header": xh,
                          "image": f"png/{i}.png", "table": table.to_dict()})
    sample = select(cands, n, seed)
    missing = [c for c in sample if not (ROOT / c["image"]).exists()]
    (ROOT / "png").mkdir(exist_ok=True)
    for name, data in fetch_members(remote, idx, [f"{PREFIX}png/{c['id']}.png" for c in missing], gap=1 << 16).items():
        (ROOT / "png" / name.rsplit("/", 1)[1]).write_bytes(data)
    sample_p.write_text(json.dumps(sample, indent=1))
    return sample


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="reselect and refetch even if sample.json is complete")
    a = ap.parse_args()
    s = fetch(a.n, a.seed, a.force)
    by = defaultdict(int)
    for c in s:
        by[(c["source"], c["chart_type"])] += 1
    print(f"{len(s)} charts in {ROOT}")
    for k, v in sorted(by.items()):
        print(f"  {k[0]:9s} {k[1]:5s} {v}")
