import zipfile
from pathlib import Path

z = Path(r"C:\git\mamintoosi-papers-codes\risk-optimal-momentum\data\large\blogfeedback.zip")
print("blog exists", z.exists(), "size", z.stat().st_size if z.exists() else None)
if z.exists():
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        print("n members", len(names))
        print(names[:30])
        for n in names[:5]:
            info = zf.getinfo(n)
            print(n, info.file_size)
