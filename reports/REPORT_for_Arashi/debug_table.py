import re
tex = open('paper/sn-article.tex', encoding='utf-8').read()
m = re.search(r"\\label\{tab:opt_noise\}.*?\\begin\{tabular\}\{lccccr\}(.*?)\\bottomrule", tex, re.S)
BS = chr(92) * 2
for line in m.group(1).split(BS):
    line = line.strip()
    if "&" not in line or "pm" in line or "multicolumn" in line:
        print("SKIP(reason):", repr(line[:90]))
        continue
    cells = [c.replace("$", "").strip().replace(chr(92) + "tau=", "") for c in line.split("&")]
    if len(cells) != 6:
        print("SKIP(ncells=%d):" % len(cells), repr(line[:90]), cells)
        continue
    try:
        lvl = float(cells[0])
        nums = [float(c) for c in cells[1:5]]
    except ValueError as e:
        print("SKIP(float):", repr(cells[0]), e)
        continue
    dt = re.match(r"([+-][0-9.]+)\s*\(([+-][0-9.]+)\)$", cells[5].replace(" ", ""))
    print("ROW lvl=%s dt=%s cells5=%r" % (lvl, bool(dt), cells[5]))
