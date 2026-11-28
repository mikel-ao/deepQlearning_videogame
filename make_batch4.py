"""
Tanda 4: mejoras de estabilidad sobre la base (ss4_long_s1: 6 acciones, sin partidas humanas, sin castigo).
Cribado: 6 grupos x 2 semillas = 12 entrenamientos de 3.000 episodios. Ejecutar UNA vez.

  b4_clip    gradient clipping (norma máx. 0.1; el log muestra el % de actualizaciones recortadas)
  b4_tau     red objetivo suave (Polyak, τ = 0.0005 por actualización ≈ misma escala que copiar cada 10.000 pasos)
  b4_n3      retornos de 3 pasos
  b4_duel    Dueling DQN
  b4_mdqn    Munchausen-DQN (α = 0.9, τ = 0.03, l0 = -1)
  b4_lrdec   learning rate decreciente: 1e-4 -> 1e-5 (exponencial a lo largo de los 3.000 episodios)
La comparación sin cambios: ss4_long_s1/s2/s3, ya reevaluados (eval_snapshots, cada 200 episodios).

Uso:  python make_batch4.py
"""
import argparse

import yaml


class NoAlias(yaml.SafeDumper):
    """Sin anclas (&id001): dos volcados con el mismo ancla rompen el YAML entero."""
    def ignore_aliases(self, data):
        return True


p = argparse.ArgumentParser()
p.add_argument("--base", default="ss4_long_s1")
p.add_argument("--episodes", type=int, default=3000)
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
p.add_argument("--file", default="hyperparameters.yml")
args = p.parse_args()

d = yaml.safe_load(open(args.file))
base = dict(d[args.base])
for k in ("resume_from", "init_model"):
    base.pop(k, None)

groups = {
    "b4_clip": {"grad_clip": 0.1},
    "b4_tau": {"target_tau": 0.0005},
    "b4_n3": {"n_step": 3},
    "b4_duel": {"dueling": True},
    "b4_mdqn": {"munchausen": True, "m_alpha": 0.9, "m_tau": 0.03, "m_l0": -1.0},
    "b4_lrdec": {"lr_end": 0.00001},
}
new = {}
for g, extra in groups.items():
    for s in args.seeds:
        b = dict(base)
        b.update(seed=s, max_episodes=args.episodes, **extra)
        new[f"{g}_s{s}"] = b

clash = set(new) & set(d)
assert not clash, f"ya existen: {sorted(clash)}"
with open(args.file, "a") as f:
    f.write(f"\n# ---- tanda 4: estabilidad (base {args.base}, generada con make_batch4.py) ----\n")
    yaml.dump(new, f, Dumper=NoAlias, sort_keys=False, allow_unicode=True)
print(" ".join(new))
