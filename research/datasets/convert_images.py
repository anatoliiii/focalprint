"""Any image files -> 8-bit grayscale .tif, names kept. Usage: convert_images.py <out dir> <files…>"""
import sys, pathlib
import cv2
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = 0
for f in map(pathlib.Path, sys.argv[2:]):
    im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if im is not None:
        cv2.imwrite(str(out / (f.stem + '.tif')), im); n += 1
print('converted', n)
