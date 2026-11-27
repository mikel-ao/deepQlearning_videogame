"""
Vídeo de progreso: una partida por snapshot, con el episodio y la puntuación encima.

Uso (desde la carpeta del proyecto, con el entorno dql activado):
    python make_video.py demonattack_long                 # todos los snapshots que haya
    python make_video.py demonattack_long --n 10          # 10 snapshots repartidos
    python make_video.py demonattack_long --every 200     # episodios 200, 400, 600...
    python make_video.py demonattack_long --max-steps 600 --fps 60
    python make_video.py demonattack_long demonattack_long2 --n 10   # junta las dos ejecuciones

Necesita:  pip install "imageio[ffmpeg]"
Se puede ejecutar mientras el entrenamiento sigue en marcha (usa los snapshots ya guardados).
"""
import argparse
import glob
import os
import re

import ale_py
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from agent import make_env
from dqn import DQN

gym.register_envs(ale_py)

SCALE = 3                     # 160x210 -> 480x630, píxeles nítidos
BAR_H = 44                    # franja superior para el texto


def pick_evenly(items, n):
    if n is None or n >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def load_font(size):
    for path in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",       # macOS
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:   # Linux
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(frame, episode, score, step, max_steps, font):
    h, w, _ = frame.shape
    img = Image.fromarray(frame).resize((w * SCALE, h * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + BAR_H), (20, 12, 36))
    canvas.paste(img, (0, BAR_H))
    d = ImageDraw.Draw(canvas)
    d.text((12, 10), f"Episodio {episode}", font=font, fill=(243, 238, 252))
    txt = f"{score:.0f} pts"
    d.text((canvas.width - 12 - d.textlength(txt, font=font), 10), txt, font=font, fill=(255, 116, 184))
    # barra fina: cuánto del clip se ha reproducido
    d.rectangle([0, BAR_H - 4, int(canvas.width * step / max_steps), BAR_H - 1], fill=(98, 210, 229))
    return np.asarray(canvas)


def play_clip(model_path, episode, hp, max_steps, seed, writer, font, device="cpu"):
    env = make_env(hp["env_id"], hp.get("frame_stack", 1), render_mode="rgb_array")
    actions = hp.get("action_set")
    net = DQN(env.observation_space.shape[0], len(actions) if actions else env.action_space.n,
              hp["hidden_dim"]).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    state, _ = env.reset(seed=seed)
    score, done, step = 0.0, False, 0
    while not done and step < max_steps:
        with torch.no_grad():
            action = net(torch.as_tensor(state, device=device).unsqueeze(0)).argmax(dim=1).item()
        state, reward, terminated, truncated, _ = env.step(actions[action] if actions else action)
        score += reward
        step += 1
        done = terminated or truncated
        writer.append_data(annotate(env.render(), episode, score, step, max_steps, font))
    env.close()
    return score, step, done


def main():
    p = argparse.ArgumentParser(description="Vídeo de progreso del DQN")
    p.add_argument("hyperparameters", nargs="+",
                   help="uno o varios bloques de hyperparameters.yml (p. ej. demonattack_long demonattack_long2)")
    p.add_argument("--every", type=int, default=None, help="usar solo los episodios múltiplos de este número (p. ej. 200)")
    p.add_argument("--n", type=int, default=None, help="nº de snapshots repartidos (por defecto, todos)")
    p.add_argument("--max-steps", type=int, default=600, help="pasos máximos por partida (600 ≈ 40 s de juego real)")
    p.add_argument("--fps", type=int, default=60, help="fps del vídeo; el juego va a ~15 decisiones/s, así que 60 = ×4")
    p.add_argument("--seed", type=int, default=0, help="misma semilla para todos los clips (comparación justa)")
    p.add_argument("--out", default=None, help="ruta del MP4 (por defecto runs/<set>_progreso.mp4)")
    args = p.parse_args()

    with open("hyperparameters.yml") as f:
        hp = yaml.safe_load(f)[args.hyperparameters[0]]        # misma arquitectura en todos

    snaps = []
    for name in args.hyperparameters:
        snaps += glob.glob(os.path.join("runs", f"{name}_snapshots", "ep_*.pt"))
    if not snaps:
        raise SystemExit("No hay snapshots todavía. Se guarda uno cada 50 episodios.")
    snaps = sorted(snaps, key=lambda s: int(re.search(r"ep_(\d+)", s).group(1)))
    if args.every:
        snaps = [s for s in snaps if int(re.search(r"ep_(\d+)", s).group(1)) % args.every == 0]
    snaps = pick_evenly(snaps, args.n)

    out = args.out or os.path.join("runs", f"{args.hyperparameters[-1]}_progreso.mp4")
    font = load_font(22)
    print(f"{len(snaps)} snapshots -> {out}")

    with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                            macro_block_size=1) as writer:
        for path in snaps:
            episode = int(re.search(r"ep_(\d+)", path).group(1))
            score, steps, finished = play_clip(path, episode, hp, args.max_steps, args.seed, writer, font)
            print(f"  ep {episode:5d}: {score:5.0f} pts en {steps} pasos"
                  f"{'' if finished else ' (recortado)'}")

    print(f"Listo: {out}")


if __name__ == "__main__":
    main()
