"""
Behavior cloning: entrena la misma red del DQN para imitar tus partidas (demos/*.npz).

Es aprendizaje supervisado: estado de la RAM (512 bytes) -> acción que pulsaste.
La salida es un .pt compatible con agent.py, así que se puede ver jugar igual que los demás:
    python agent.py ss4_long_s1 --model models/bc.pt

Uso:
    python train_bc.py                    # entrena y evalúa con 30 partidas (ε=0)
    python train_bc.py --balance          # pesa menos NOOP si lo pulsas casi siempre
    python train_bc.py --eval-games 0     # solo entrenar

Validación: las ÚLTIMAS partidas grabadas (no frames al azar: los frames seguidos se parecen
tanto que una validación aleatoria daría un acierto falso).
"""
import argparse
import glob
import os

import numpy as np
import torch
from torch import nn

from agent import make_env
from dqn import DQN
from eval_snapshots import play

torch.manual_seed(0)
np.random.seed(0)


def load_demos(frame_stack):
    files = sorted(glob.glob("demos/demo_*.npz"))
    if not files:
        raise SystemExit("No hay demostraciones en demos/. Graba primero con record_human.py")
    S, A, T, scores = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if d["states"].shape[1] != 128 * frame_stack:
            print(f"  (salto {f}: frame_stack distinto)")
            continue
        S.append(d["states"]); A.append(d["actions"]); T.append(d["terminated"])
        scores += list(d["scores"])
    return np.concatenate(S), np.concatenate(A), np.concatenate(T), scores, len(files)


def main():
    p = argparse.ArgumentParser(description="Behavior cloning sobre demostraciones humanas")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.15, help="fracción final de los datos para validar")
    p.add_argument("--balance", action="store_true", help="pesos de clase inversos a la frecuencia")
    p.add_argument("--eval-games", type=int, default=30)
    p.add_argument("--fire-always", action="store_true",
                   help="NOOP->FIRE, RIGHT->RIGHTFIRE, LEFT->LEFTFIRE: el clon solo decide hacia dónde moverse")
    p.add_argument("--lag", type=int, default=0,
                   help="emparejar el estado t con tu acción t+lag (tu tiempo de reacción, ~3 pasos = 200 ms)")
    p.add_argument("--out", default="models/bc.pt")
    args = p.parse_args()

    S, A, T, scores, nfiles = load_demos(args.frame_stack)
    n_actions = 6
    if args.fire_always:
        A = np.array([1, 1, 4, 5, 4, 5])[A]          # todas las acciones pasan a su versión con disparo
    if args.lag:
        # estado t -> acción t+lag, sin cruzar el final de una partida
        ok = np.ones(len(A), dtype=bool)
        for e in np.where(T)[0]:
            ok[max(0, e - args.lag + 1):e + 1] = False
        ok[-args.lag:] = False
        idx = np.where(ok)[0]
        S, A, T = S[idx], A[idx + args.lag], T[idx + args.lag]
    print(f"{len(A)} pasos de {nfiles} sesiones, {len(scores)} partidas completas"
          + (f" (tu media: {np.mean(scores):.0f}, máx {np.max(scores):.0f})" if scores else ""))
    counts = np.bincount(A, minlength=n_actions)
    print("Acciones:", counts.tolist())

    # corte en el último fin de partida antes del val_frac final -> validación = partidas enteras
    cut = int(len(A) * (1 - args.val_frac))
    ends = np.where(T[:cut])[0]
    cut = ends[-1] + 1 if len(ends) else cut
    Xtr, Ytr = torch.as_tensor(S[:cut]), torch.as_tensor(A[:cut])
    Xva, Yva = torch.as_tensor(S[cut:]), torch.as_tensor(A[cut:])
    print(f"Entrenamiento {len(Ytr)} pasos | validación {len(Yva)} pasos")

    weight = None
    if args.balance:
        c = np.bincount(A[:cut], minlength=n_actions).astype(np.float32)
        weight = torch.as_tensor(np.where(c > 0, c.sum() / (n_actions * np.maximum(c, 1)), 0.0),
                                 dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    net = DQN(128 * args.frame_stack, n_actions, args.hidden_dim)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best_val, best_state = np.inf, None
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            loss = loss_fn(net(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tr_acc = (net(Xtr).argmax(1) == Ytr).float().mean().item()
            logits = net(Xva) if len(Yva) else None
            va_loss = loss_fn(logits, Yva).item() if len(Yva) else float("nan")
            va_acc = (logits.argmax(1) == Yva).float().mean().item() if len(Yva) else float("nan")
        if len(Yva) and va_loss < best_val:
            best_val, best_state = va_loss, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"época {epoch:3d} | acierto train {tr_acc:.3f} | val {va_acc:.3f} | pérdida val {va_loss:.3f}")

    if best_state is not None:                      # el de menor pérdida de validación (antes de sobreajustar)
        net.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"\nGuardado {args.out}")

    if args.eval_games:
        env = make_env("ALE/DemonAttack-v5", args.frame_stack)
        s = play(net, env, args.eval_games, seed0=1000)          # mismo protocolo que eval_snapshots.py
        se = s.std() / np.sqrt(len(s))
        print(f"Evaluación clon: media {s.mean():.1f} ± {se:.1f} (EE, {len(s)} partidas)"
              f" | mediana {np.median(s):.0f} | min {s.min():.0f} | max {s.max():.0f}")
        if scores:
            print(f"Tú: media {np.mean(scores):.0f} en {len(scores)} partidas")


if __name__ == "__main__":
    main()
