"""
Compara grupos de entrenamientos a partir de las reevaluaciones de snapshots (eval_snapshots.py, 30 partidas).

Para cada grupo junta sus semillas y usa solo los snapshots 200, 400, ..., 3000 (los mismos para todos):
  - media de snapshots   media de todas las evaluaciones (¿cómo juega en general a lo largo del entrenamiento?)
  - mejor                el mejor snapshot de cada semilla (lo que nos quedaríamos)
  - buenos               nº de snapshots con media > 750
  - escondidos           nº de snapshots que se esconden: partidas de > 15.000 pasos y mediana <= 100
  - pared                % medio del tiempo pegado a una pared

Uso:  python compare_groups.py
      python compare_groups.py --groups b3_a3 b3_demo ss4_long --max-ep 3000
"""
import argparse
import csv
import glob
import math
import os
import re
from statistics import mean

p = argparse.ArgumentParser()
p.add_argument("--groups", nargs="+",
               default=["ss4_long", "b3_a3", "b3_demo", "b3_dqfd", "b3_start", "b3_corner"])
p.add_argument("--max-ep", type=int, default=3000)
p.add_argument("--every", type=int, default=200)
p.add_argument("--seeds", nargs="+", default=["s1", "s2", "s3"])
p.add_argument("--good", type=float, default=750)
args = p.parse_args()

NAMES = {"ss4_long": "base (6 acciones)", "b3_a3": "3 acciones", "b3_demo": "+ mis partidas",
         "b3_dqfd": "+ partidas + margen", "b3_start": "+ partidas + mis estados", "b3_corner": "3 acc. + castigo esquinas"}


def load(run):
    path = os.path.join("runs", f"{run}_snapshot_eval_g30.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            ep = int(r["episodio"])
            if ep % args.every or ep > args.max_ep:
                continue
            num = lambda k: float(r[k]) if r.get(k) not in (None, "") else math.nan
            rows.append({"ep": ep, "media": num("media"), "desv": num("desv"), "mediana": num("mediana"),
                         "pared": num("pared"), "pasos": num("pasos"), "n": int(float(r["partidas"]))})
    return rows


hdr = f"{'grupo':<27} {'semillas':>8} {'snaps':>5} {'media snaps':>11} {'mejor por semilla':<24} {'buenos':>7} {'escond.':>7} {'pared':>6}"
print(hdr)
print("-" * len(hdr))
for g in args.groups:
    seeds, allrows, bests = [], [], []
    for s in args.seeds:
        rows = load(f"{g}_{s}")
        if not rows:
            continue
        seeds.append(s)
        allrows += rows
        b = max(rows, key=lambda r: r["media"])
        bests.append(f"{b['media']:.0f}")
    if not allrows:
        print(f"{NAMES.get(g, g):<27} (sin evaluaciones)")
        continue
    good = sum(r["media"] > args.good for r in allrows)
    hidden = [r for r in allrows if not math.isnan(r["pasos"])]
    n_hid = sum(r["pasos"] > 15000 and r["mediana"] <= 100 for r in hidden)
    walls = [r["pared"] for r in allrows if not math.isnan(r["pared"])]
    print(f"{NAMES.get(g, g):<27} {len(seeds):>8} {len(allrows):>5} {mean(r['media'] for r in allrows):>11.0f}"
          f" {' / '.join(bests):<24} {good:>3}/{len(allrows):<3} "
          f"{(f'{n_hid}/{len(hidden)}' if hidden else '-'):>7} {(f'{mean(walls):.0f}%' if walls else '-'):>6}")

print(f"\nSnapshots {args.every}..{args.max_ep} cada {args.every}, 30 partidas cada uno, ε=0."
      f" 'buenos' = media > {args.good:.0f}; 'escond.' = partidas > 15.000 pasos y mediana <= 100.")
