import json, sys, time
import pytesseract
from PIL import Image
for i in sys.argv[1:]:
    p = f"bench/data/synth/probe/{int(i):04d}"
    im = Image.open(p + ".jpg").convert("L")
    d = json.load(open(p + ".json"))
    big = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    for psm in (11, 6):
        t = time.time()
        r = pytesseract.image_to_data(big, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
        words = [(r["text"][k], int(float(r["conf"][k])), r["left"][k] // 2, r["top"][k] // 2) for k in range(len(r["text"])) if r["text"][k].strip()]
        print(i, psm, f"{time.time()-t:.2f}s", d["semantics"]["x_ticks"], d["semantics"]["y_ticks"])
        print("   ", words)
