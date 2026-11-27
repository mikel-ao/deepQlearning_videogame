"""Resumen de los entrenamientos en curso: una fila por log.
Uso:  python resumen.py   |   watch -n 60 python resumen.py
"""
import glob, os, re, sys

EP = re.compile(r"ep\s+(\d+) \| r=\s*(-?\d+) \| media100=\s*([\d.]+) \| ε=([\d.]+) \| pasos=(\d+) \| ([\d,]+) pasos/s")
EV = re.compile(r"EVAL ep (\d+): media\d+=([\d.]+)")

pattern = sys.argv[1] if len(sys.argv) > 1 else "runs/ss4_long_*.log"
rows = []
for path in sorted(glob.glob(pattern)):
    last, evals, best_m100 = None, [], 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if (m := EP.search(line)):
                last = m
                best_m100 = max(best_m100, float(m.group(3)))
            elif (m := EV.search(line)):
                evals.append((int(m.group(1)), float(m.group(2))))
    if last is None:
        continue
    best_ev = max(evals, key=lambda e: e[1]) if evals else None
    rows.append((os.path.basename(path).removesuffix(".log"),
                 int(last.group(1)), float(last.group(3)), best_m100, float(last.group(4)),
                 f"{evals[-1][1]:.0f} (ep {evals[-1][0]})" if evals else "-",
                 f"{best_ev[1]:.0f} (ep {best_ev[0]})" if best_ev else "-",
                 last.group(6)))

fmt = "{:<18} {:>6} {:>9} {:>13} {:>6} {:>15} {:>15} {:>8}"
print(fmt.format("run", "ep", "media100", "max media100", "ε", "último EVAL", "mejor EVAL", "pasos/s"))
print("-" * 98)
for r in rows:
    print(fmt.format(r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.3f}", r[5], r[6], r[7]))
