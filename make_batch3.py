"""
Tanda 3: ¿ayudan tus partidas? Genera los bloques en hyperparameters.yml (ejecutar UNA vez).

Todos parten del mismo bloque base y cambian solo lo indicado (3 semillas cada grupo, 15 entrenamientos):
  b3_a3      3 acciones (FIRE, RIGHTFIRE, LEFTFIRE), sin demostraciones   -> efecto de las 3 acciones
  b3_demo    3 acciones + 25 % de cada mini-batch con tus partidas         -> efecto de tus datos
  b3_dqfd    lo anterior + pérdida de margen de DQfD (imitar tu acción)    -> efecto de imitar
  b3_start   3 acciones + demos + 30 % de episodios empiezan en un estado  -> practicar oleadas difíciles
             tuyo con >= 1000 puntos
  b3_corner  3 acciones + castigo de 0,01 por paso pegado a una pared       -> ¿deja de esconderse?
             más de 2 s seguidos (sin demostraciones, se compara con b3_a3)
La comparación sin nada de esto ya existe: las semillas 1-3 del bloque base.

Uso:  python make_batch3.py                      # base ss4_long_s1 (lr 1e-4)
      python make_batch3.py --base b2_lr5e5_s2   # si la tanda 2 dice que otra configuración es mejor
"""
import argparse

import yaml

p = argparse.ArgumentParser()
p.add_argument("--base", default="ss4_long_s1")
p.add_argument("--episodes", type=int, default=3000)
p.add_argument("--file", default="hyperparameters.yml")
args = p.parse_args()

d = yaml.safe_load(open(args.file))
base = dict(d[args.base])
for k in ("resume_from", "init_model"):
    base.pop(k, None)

A3 = {"action_set": [1, 4, 5]}
DEMO = {"demo_glob": "demos/demo_*.npz", "demo_ratio": 0.25}
groups = {
    "b3_a3": {**A3},
    "b3_demo": {**A3, **DEMO},
    "b3_dqfd": {**A3, **DEMO, "demo_margin": 1.0},
    "b3_start": {**A3, **DEMO, "demo_start_prob": 0.3, "demo_start_min_score": 1000},
    "b3_corner": {**A3, "corner_penalty": 0.01, "corner_patience": 30},
}
new = {}
for g, extra in groups.items():
    for s in (1, 2, 3):
        b = dict(base)
        b.update(seed=s, max_episodes=args.episodes, **extra)
        new[f"{g}_s{s}"] = b

clash = set(new) & set(d)
assert not clash, f"ya existen: {sorted(clash)}"
with open(args.file, "a") as f:
    f.write(f"\n# ---- tanda 3: tus partidas (base {args.base}, generada con make_batch3.py) ----\n")
    yaml.safe_dump(new, f, sort_keys=False, allow_unicode=True)
print(" ".join(new))
