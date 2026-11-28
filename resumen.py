"""Resumen de los entrenamientos en curso: una fila por log.
"pared" = % medio de pasos pegado a una pared en los últimos 100 episodios (solo logs nuevos; alto = se esconde).

Uso:  python resumen.py                 -> todos los runs/ss4_long_*.log
      python resumen.py "runs/*.log"    -> otro patrón
      watch -n 60 python resumen.py     -> se refresca cada minuto (Ctrl+c para salir)
"""
import glob
import os
import re
import sys

EP = re.compile(r"ep\s+(\d+) \| r=\s*(-?\d+) \| media100=\s*([\d.]+) \| ε=([\d.]+) \| pasos=(\d+) \| ([\d,]+) pasos/s")
WALL = re.compile(r"\| pared (\d+)%")
EV = re.compile(r"EVAL ep (\d+): media\d+=([\d.]+)")

pattern = sys.argv[1] if len(sys.argv) > 1 else "runs/ss4_long_*.log"
rows = []
for path in sorted(glob.glob(pattern)):
    last, evals, best_m100, walls = None, [], 0.0, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if (m := EP.search(line)):
                last = m
                best_m100 = max(best_m100, float(m.group(3)))
                if (w := WALL.search(line)):
                    walls.append(int(w.group(1)))
            elif (m := EV.search(line)):
                evals.append((int(m.group(1)), float(m.group(2))))
    if last is None:
        continue
    best_ev = max(evals, key=lambda e: e[1]) if evals else None
    rows.append((
        os.path.basename(path).removesuffix(".log"),
        int(last.group(1)), float(last.group(3)), best_m100, float(last.group(4)),
        f"{evals[-1][1]:.0f} (ep {evals[-1][0]})" if evals else "-",
        f"{best_ev[1]:.0f} (ep {best_ev[0]})" if best_ev else "-",
        last.group(6),
        f"{sum(walls[-100:]) / len(walls[-100:]):.0f}%" if walls else "-",
    ))

hdr = ("run", "ep", "media100", "max media100", "ε", "último EVAL", "mejor EVAL", "pasos/s", "pared")
fmt = "{:<18} {:>6} {:>9} {:>13} {:>6} {:>15} {:>15} {:>8} {:>6}"
print(fmt.format(*hdr))
print("-" * 105)
for r in rows:
    print(fmt.format(r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.3f}", r[5], r[6], r[7], r[8]))
