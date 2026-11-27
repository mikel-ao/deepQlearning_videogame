import sys, yaml
path = sys.argv[1] if len(sys.argv) > 1 else "hyperparameters.yml"
d = yaml.safe_load(open(path))
base = d["ss4_long_s1"]
new = {}
def add(name, seed, **kw):
    b = dict(base); b.update(seed=seed, max_episodes=3000, **kw); new[name] = b
for s in (2, 3):    add(f"b2_lr5e5_s{s}", s, learning_rate=0.00005)
for s in (1, 2, 3): add(f"b2_lr25e6_s{s}", s, learning_rate=0.000025)
for s in (1, 2, 3): add(f"b2_pen_s{s}", s, life_loss="terminal_penalty")
for s in (1, 2):    add(f"b2_sync30k_s{s}", s, network_sync_rate=30000)
assert not set(new) & set(d), "ya existen bloques b2_* en el YAML"
with open(path, "a") as f:
    f.write("\n# ---- tanda 2: contra la inestabilidad (generada con make_batch2.py) ----\n")
    yaml.safe_dump(new, f, sort_keys=False, allow_unicode=True)
print(" ".join(new))
