"""
Vídeo de UNA partida completa de un modelo, eligiendo la mejor partida entre varias semillas.

Primero juega --tries partidas sin grabar (rápido) y apunta puntos y cuánto se mueve;
luego graba la mejor partida que cumpla el mínimo de movimiento (para que no salga escondido).

Uso (desde la carpeta del repo, con dql activado):
    python best_game_video.py ss4_long_s3 models/ss4_long_s3_ep1100.pt
    python best_game_video.py ss4_long_s3 models/ss4_long_s3_ep1100.pt --tries 40 --min-move 0.4 --fps 30

--fps 15 = velocidad real (el agente decide 15 veces por segundo); 30 = x2, más ágil para redes sociales.
Salida: videos/<bloque>_mejor_partida.mp4
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from agent import make_env
from dqn import DQN
from PIL import Image, ImageDraw

from make_video import BAR_H, SCALE, load_font

MOVES = {2, 3, 4, 5}        # RIGHT, LEFT, RIGHTFIRE, LEFTFIRE (acciones del juego)


def overlay(frame, title, score, step, total, font):
    h, w, _ = frame.shape
    img = Image.fromarray(frame).resize((w * SCALE, h * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + BAR_H), (20, 12, 36))
    canvas.paste(img, (0, BAR_H))
    d = ImageDraw.Draw(canvas)
    d.text((12, 10), title, font=font, fill=(243, 238, 252))
    txt = f"{score:.0f} pts"
    d.text((canvas.width - 12 - d.textlength(txt, font=font), 10), txt, font=font, fill=(255, 116, 184))
    d.rectangle([0, BAR_H - 4, int(canvas.width * step / max(total, 1)), BAR_H - 1], fill=(98, 210, 229))
    return np.asarray(canvas)


def load(block, model):
    with open("hyperparameters.yml") as f:
        hp = yaml.safe_load(f)[block]
    actions = hp.get("action_set")
    env = make_env(hp["env_id"], hp.get("frame_stack", 1))
    net = DQN(env.observation_space.shape[0], len(actions) if actions else env.action_space.n, hp["hidden_dim"])
    net.load_state_dict(torch.load(model, map_location="cpu"))
    net.eval()
    env.close()
    return hp, actions, net


def play(net, actions, hp, seed, render=False, writer=None, font=None, title="", total=1):
    env = make_env(hp["env_id"], hp.get("frame_stack", 1), render_mode="rgb_array" if render else None)
    state, _ = env.reset(seed=seed)
    score, steps, moves, done = 0.0, 0, 0, False
    while not done:
        with torch.no_grad():
            a = net(torch.as_tensor(state).unsqueeze(0)).argmax(dim=1).item()
        a = actions[a] if actions else a
        moves += a in MOVES
        state, r, te, tr, _ = env.step(a)
        score += r
        steps += 1
        done = te or tr
        if render:
            writer.append_data(overlay(env.render(), title, score, steps, total, font))
    env.close()
    return score, steps, moves / max(steps, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("block", help="bloque de hyperparameters.yml con la configuración del modelo")
    p.add_argument("model", help=".pt del modelo")
    p.add_argument("--tries", type=int, default=30, help="partidas de prueba para elegir la mejor")
    p.add_argument("--min-move", type=float, default=0.3, help="fracción mínima de pasos moviéndose")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=None, help="grabar directamente esta semilla (sin buscar)")
    p.add_argument("--title", default="DQN · Demon Attack", help="texto arriba a la izquierda del vídeo")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    hp, actions, net = load(args.block, args.model)
    seeds = [args.seed] if args.seed is not None else range(args.tries)
    print(f"Probando {len(seeds)} partida(s)...")
    results = []
    for seed in seeds:
        s, n, mv = play(net, actions, hp, seed)
        results.append((s, n, mv, seed))
        print(f"  semilla {seed:3d}: {s:5.0f} puntos, {n:5d} pasos, se mueve el {mv:.0%}")
    ok = [r for r in results if r[2] >= args.min_move] or results
    score, steps, mv, seed = max(ok, key=lambda r: r[0])
    print(f"\nElegida semilla {seed}: {score:.0f} puntos, {steps} pasos (~{steps / args.fps:.0f} s de vídeo a {args.fps} fps),"
          f" se mueve el {mv:.0%}")

    os.makedirs("videos", exist_ok=True)
    out = args.out or os.path.join("videos", f"{args.block}_mejor_partida.mp4")
    font = load_font(22)
    with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as w:
        s2, _, _ = play(net, actions, hp, seed, render=True, writer=w, font=font, title=args.title, total=steps)
    print(f"Listo: {out} ({s2:.0f} puntos)")


if __name__ == "__main__":
    main()
