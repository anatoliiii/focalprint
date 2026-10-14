"""NIST MINEX III validation imagery (public domain): raw 8-bit .gray -> .tif.
Usage: convert_minex.py <validation dir> <out dir>"""
import re, sys, pathlib
import numpy as np, cv2
src, out = map(pathlib.Path, sys.argv[1:3])
header = (src / 'minexiii_validation_data.h').read_text()
dims = {n: (int(w), int(h)) for n, w, h in re.findall(r'\{"(\w+\.gray)", \{(\d+), (\d+),', header)}
out.mkdir(parents=True, exist_ok=True)
n = 0
for name, (w, h) in dims.items():
    raw = (src / 'validation_imagery_raw' / name).read_bytes()
    if len(raw) != w * h:
        print('skip', name, len(raw), w, h); continue
    cv2.imwrite(str(out / name.replace('.gray', '.tif')), np.frombuffer(raw, np.uint8).reshape(h, w)); n += 1
print('converted', n)
