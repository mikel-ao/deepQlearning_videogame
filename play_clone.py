"""
Ver o evaluar un clon (train_bc.py --fire-always) cambiando CÓMO elige la acción.

El clon con argmax elige casi siempre la acción más probable ("quieto disparando").
Dos alternativas:
  --sample         elegir al azar según las probabilidades del clon (si duda 40/30/30, a veces se mueve)
  --stay-bias X    restar X al logit de "quieto" (FIRE) antes de decidir: le cuesta más quedarse quieto

Uso:
    python play_clone.py models/bc_fire.pt --sample --watch          # verlo jugar
    python play_clone.py models/bc_fire.pt --sample --games 30       # evaluar (mismas semillas que siempre)
    python play_clone.py models/bc_fire.pt --stay-bias 0.5 --games 30
"""
import argparse

import numpy as np
import torch

from agent import make_env
from dqn import DQN

torch.set_num_threads(1)


def choose(net, state, sample, stay_bias, rng):
    with torch.no_grad():
        logits = net(torch.as_tensor(state).unsqueeze(0))[0].clone()
    logits[1] -= stay_bias                                   # 1 = FIRE (quieto disparando)
    allowed = torch.tensor([1, 4, 5])                        # FIRE, RIGHTFIRE, LEFTFIRE
    sub = logits[allowed]
    if sample:
        p = torch.softmax(sub, 0).numpy()
        return int(allowed[rng.choice(3, p=p)])
    return int(allowed[sub.argmax()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--stay-bias", type=float, default=0.0)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--watch", action="store_true", help="ventana con el juego (no evalúa)")
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    env = make_env("ALE/DemonAttack-v5", 4, render_mode="human" if args.watch else None)
    net = DQN(512, 6, 256)
    net.load_state_dict(torch.load(args.model, map_location="cpu"))
    net.eval()
    rng = np.random.default_rng(args.seed)

    scores, moves = [], np.zeros(6, dtype=int)
    for g in range(10**9 if args.watch else args.games):
        s, _ = env.reset(seed=args.seed + g)
        done, tot = False, 0.0
        while not done:
            a = choose(net, s, args.sample, args.stay_bias, rng)
            moves[a] += 1
            s, r, te, tr, _ = env.step(a)
            tot += r
            done = te or tr
        scores.append(tot)
        if args.watch:
            print(f"Partida {g + 1}: {tot:.0f} puntos")
    s = np.array(scores)
    total = moves[[1, 4, 5]].sum()
    print(f"media {s.mean():.1f} ± {s.std() / np.sqrt(len(s)):.1f} (EE, {len(s)} partidas) | mediana {np.median(s):.0f}"
          f" | min {s.min():.0f} | max {s.max():.0f}")
    print(f"quieto {moves[1] / total:.0%} | derecha {moves[4] / total:.0%} | izquierda {moves[5] / total:.0%}"
          "   (tú: 38% / 31% / 31%)")


if __name__ == "__main__":
    main()
