import httpx, zipfile, io, collections, time
URL = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"

class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.c = httpx.Client(follow_redirects=True, timeout=60)
        r = self.c.head(url); r.raise_for_status()
        self.url = str(r.url); self.size = int(r.headers["content-length"]); self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        r = self.c.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos+n-1}"}); r.raise_for_status()
        self.pos += len(r.content); return r.content
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

t=time.time()
z = zipfile.ZipFile(RangeFile(URL))
names = z.namelist()
print(len(names), time.time()-t)
c = collections.Counter("/".join(n.split("/")[:4]) for n in names)
for k,v in sorted(c.items()): print(k, v)
