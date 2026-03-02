python - <<'PY'
import os
from PIL import Image

root="/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/data22/viame/images"
bad=[]
for fn in sorted(os.listdir(root)):
    if not fn.lower().endswith(".png"): 
        continue
    p=os.path.join(root, fn)
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception:
        bad.append(p)

print(f"checked: {len([f for f in os.listdir(root) if f.lower().endswith('.png')])}")
print(f"bad: {len(bad)}")
for p in bad[:50]:
    print(p)
if bad:
    out="/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/data22/bad_pngs.txt"
    with open(out,"w") as f:
        f.write("\n".join(bad) + "\n")
    print("wrote:", out)
PY
