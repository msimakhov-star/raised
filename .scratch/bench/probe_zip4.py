import httpx, zipfile, io, os
exec(open(".scratch/bench/probe_zip.py").read().split("t=time.time()")[0].replace('if __name__ != "__main__": raise SystemExit',''))
z = zipfile.ZipFile(RangeFile(URL))
want = ["two_col_100025","multi_col_10","00339007006077","10099","10146","OECD_POVERTY_RATE_GBR_SVN_000002"]
for w in want:
    for sub, ext in (("png",".png"),("tables",".csv"),("annotations",".json")):
        n = f"ChartQA Dataset/test/{sub}/{w}{ext}"
        try: data = z.read(n)
        except KeyError: print("missing", n); continue
        open(f".scratch/bench/probe/{w}{ext}","wb").write(data)
