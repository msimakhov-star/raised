import httpx, zipfile, io, collections, re, json, pickle, sys
exec(open(".scratch/bench/probe_zip.py").read().split("t=time.time()")[0].replace('if __name__ != "__main__": raise SystemExit',''))
z = zipfile.ZipFile(RangeFile(URL))
names = z.namelist()
open(".scratch/bench/names.txt","w").write("\n".join(names))
c = collections.Counter("/".join(n.split("/")[:-1]) for n in names)
for k,v in sorted(c.items()): print(k, v)
