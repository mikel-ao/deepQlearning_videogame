"""
Evalúa los snapshots guardados con muchas partidas (ε=0, sin entrenar) para confirmar
cuáles son buenos de verdad. Guarda una tabla CSV y copia el mejor como modelo confirmado.

Uso (desde la carpeta del proyecto, con dql activado):
    python eval_snapshots.py demonattack_stack4 --episodes 200 400          # solo esos dos
    python eval_snapshots.py demonattack_stack4 --every 50 --games 20        # todos (≈ 10 min)

Salida:
    runs/<set>_snapshot_eval[_eps<ε>]_g<partidas>.csv   episodio, media, desviación, min, mediana, max
    runs/<set>_best_confirmed[_eps<ε>]_g<partidas>.pt    el snapshot con mejor media
Para verlo jugar:
    python agent.py <set> --model runs/<set>_best_confirmed_g30.pt
"""
import argparse
import random
import csv
import glob
import os
import re
import shutil
import time

import numpy as np
import torch
import yaml

from agent import make_env
from dqn import DQN

torch.set_num_threads(1)          # no robar CPU a un entrenamiento que esté en marcha


def play(net, env, games, seed0, epsilon=0.0, actions=None):
    """actions: lista de acciones del juego si el modelo usa action_set (p. ej. [1, 4, 5])."""
    scores = []
    for g in range(games):
        state, _ = env.reset(seed=seed0 + g)          # mismas semillas para todos los snapshots
        done, score = False, 0.0
        while not done:
            if random.random() < epsilon:                  # ε de evaluación (el paper de DQN usa 0.05)
                action = random.randrange(len(actions)) if actions else env.action_space.sample()
            else:
                with torch.no_grad():
                    action = net(torch.as_tensor(state).unsqueeze(0)).argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(actions[action] if actions else action)
            score += reward
            done = terminated or truncated
        scores.append(score)
    return np.array(scores)


def main():
    p = argparse.ArgumentParser(description="Evaluar snapshots con muchas partidas")
    p.add_argument("hyperparameters", help="bloque de hyperparameters.yml")
    p.add_argument("--episodes", type=int, nargs="*", help="episodios concretos (p. ej. 200 400)")
    p.add_argument("--every", type=int, default=None, help="solo episodios múltiplos de este número")
    p.add_argument("--games", type=int, default=30, help="partidas por snapshot (por defecto 30)")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--epsilon", type=float, default=0.0, help="ε durante la evaluación (0 = greedy; 0.05 como el paper)")
    args = p.parse_args()

    with open("hyperparameters.yml") as f:
        hp = yaml.safe_load(f)[args.hyperparameters]

    snap_dir = os.path.join("runs", f"{args.hyperparameters}_snapshots")
    snaps = {int(re.search(r"ep_(\d+)", s).group(1)): s
             for s in glob.glob(os.path.join(snap_dir, "ep_*.pt"))}
    if args.episodes:
        missing = [e for e in args.episodes if e not in snaps]
        if missing:
            raise SystemExit(f"No existen snapshots para los episodios {missing}")
        snaps = {e: snaps[e] for e in args.episodes}
    elif args.every:
        snaps = {e: s for e, s in snaps.items() if e % args.every == 0}
    if not snaps:
        raise SystemExit(f"No hay snapshots en {snap_dir}")

    env = make_env(hp["env_id"], hp.get("frame_stack", 1))
    actions = hp.get("action_set")
    net = DQN(env.observation_space.shape[0], len(actions) if actions else env.action_space.n, hp["hidden_dim"])
    net.eval()

    random.seed(args.seed)
    rows = []
    print(f"{len(snaps)} snapshots × {args.games} partidas (ε={args.epsilon})\n")
    print(f"{'episodio':>8} | {'media':>7} | {'desv':>6} | {'min':>5} | {'mediana':>7} | {'max':>5} | t")
    for ep in sorted(snaps):
        t0 = time.time()
        net.load_state_dict(torch.load(snaps[ep], map_location="cpu"))
        s = play(net, env, args.games, args.seed, args.epsilon, actions)
        rows.append([ep, s.mean(), s.std(), s.min(), np.median(s), s.max()])
        print(f"{ep:8d} | {s.mean():7.1f} | {s.std():6.1f} | {s.min():5.0f} | {np.median(s):7.0f} | {s.max():5.0f} | {time.time() - t0:4.0f}s")
    env.close()

    # el nombre incluye ε y nº de partidas; el CSV ACUMULA: evaluar otros episodios añade filas
    # (si un episodio ya estaba, se actualiza) -> nunca se pierde una evaluación anterior
    tag = ("" if args.epsilon == 0 else f"_eps{args.epsilon:g}") + f"_g{args.games}"
    out = os.path.join("runs", f"{args.hyperparameters}_snapshot_eval{tag}.csv")
    merged = {}
    if os.path.exists(out):
        with open(out) as f:
            for row in csv.DictReader(f):
                e = int(row["episodio"])
                merged[e] = [e] + [float(row[k]) for k in ("media", "desv", "min", "mediana", "max")]
    for r in rows:
        merged[r[0]] = r
    rows = [merged[e] for e in sorted(merged)]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "media", "desv", "min", "mediana", "max", "partidas"])
        for r in rows:
            w.writerow([r[0]] + [round(x, 1) for x in r[1:]] + [args.games])

    rows = [r for r in rows if r[0] in snaps or os.path.exists(os.path.join(snap_dir, f"ep_{r[0]:05d}.pt"))]
    best = max(rows, key=lambda r: r[1])
    snaps.setdefault(best[0], os.path.join(snap_dir, f"ep_{best[0]:05d}.pt"))
    best_path = os.path.join("runs", f"{args.hyperparameters}_best_confirmed{tag}.pt")
    shutil.copy(snaps[best[0]], best_path)
    sem = best[2] / np.sqrt(args.games)
    print(f"\nMejor de todas las evaluadas ({len(rows)} snapshots): episodio {best[0]} con media {best[1]:.1f} ± {sem:.1f} (error estándar, {args.games} partidas)")
    print(f"Tabla: {out}\nModelo: {best_path}")


if __name__ == "__main__":
    main()
