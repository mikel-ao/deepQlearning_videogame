"""
Juega a Demon Attack con el teclado y graba tus partidas como demostraciones.

Se graba lo mismo que ve el agente: la RAM (128 bytes) apilada 4 veces = 512 bytes,
con frameskip 4 (el agente decide 15 veces por segundo). No se guarda ninguna imagen.

Controles:  ← →  moverse   |   ESPACIO  disparar (se puede combinar con ← →)
            ESC o cerrar la ventana  terminar (se guarda todo lo jugado)

Uso (en el Mac, con dql activado):
    python record_human.py                 # velocidad real del juego (15 decisiones/s)
    python record_human.py --fps 10        # más lento = más fácil, mismas reglas
    python record_human.py --sticky 0      # sin "sticky actions" (más preciso para ti)
    python record_human.py --infinite-lives   # nunca hay game over: para llegar a las oleadas difíciles

Cada sesión se guarda en demos/demo_<fecha>.npz con:
    states (N,512) uint8, actions (N,), rewards (N,), next_states (N,512),
    terminated (N,) fin de partida, life_lost (N,) perdió una vida, scores (partidas,),
    emu_states: estado completo del emulador cada 30 pasos (para que el DQN pueda empezar
    partidas justo en las oleadas difíciles a las que tú llegas)
"""
import argparse
import os
import pickle
import time

import ale_py
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.utils.play import play

gym.register_envs(ale_py)
LIVES_RAM = 114        # byte de la RAM con las vidas de reserva en Demon Attack (ale.lives() = este valor + 1)


def make_play_env(env_id, frame_stack, sticky):
    env = gym.make(env_id, obs_type="ram", render_mode="rgb_array",
                   repeat_action_probability=sticky)
    env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
    env = gym.wrappers.FlattenObservation(env)
    return env


def main():
    p = argparse.ArgumentParser(description="Grabar demostraciones humanas (RAM)")
    p.add_argument("--env", default="ALE/DemonAttack-v5")
    p.add_argument("--frame-stack", type=int, default=4, help="igual que el bloque que vayas a entrenar")
    p.add_argument("--fps", type=int, default=15, help="15 = velocidad real con frameskip 4")
    p.add_argument("--sticky", type=float, default=0.25, help="0.25 como el agente; 0 = control exacto")
    p.add_argument("--zoom", type=float, default=3.0)
    p.add_argument("--state-every", type=int, default=30,
                   help="guardar el estado completo del emulador cada N pasos (0 = no); ~8 KB cada uno")
    p.add_argument("--infinite-lives", action="store_true",
                   help="vidas infinitas para llegar a oleadas avanzadas (las puntuaciones NO son comparables con el agente)")
    p.add_argument("--keep-partial", action="store_true", help="guardar también la partida a medias del final")
    args = p.parse_args()

    env = make_play_env(args.env, args.frame_stack, args.sticky)

    # teclas -> acciones, a partir de los nombres reales de las acciones del juego
    meanings = env.unwrapped.get_action_meanings()
    combos = {"FIRE": (pygame.K_SPACE,), "RIGHT": (pygame.K_RIGHT,), "LEFT": (pygame.K_LEFT,),
              "RIGHTFIRE": (pygame.K_RIGHT, pygame.K_SPACE), "LEFTFIRE": (pygame.K_LEFT, pygame.K_SPACE)}
    keys_to_action = {combos[m]: i for i, m in enumerate(meanings) if m in combos}

    buf = {k: [] for k in ("states", "actions", "rewards", "next_states", "terminated", "life_lost")}
    scores, ep = [], {"score": 0.0, "lives": None, "steps": 0, "deaths": 0}
    emu_states = []        # (paso global, partida, paso en la partida, puntos hasta ahí, vidas, estado ALE)

    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        if isinstance(obs_t, tuple):           # el primer paso tras reset llega como (obs, info)
            obs_t = obs_t[0]
        lives = info.get("lives")
        lost = ep["lives"] is not None and lives is not None and lives < ep["lives"]
        if args.infinite_lives:
            ale = env.unwrapped.ale
            if ale.getRAM()[LIVES_RAM] < 3:          # rellenar la reserva: nunca llega al game over
                ale.setRAM(LIVES_RAM, 3)
            lives = ale.lives()
            if lost:
                ep["deaths"] += 1
        ep["lives"] = lives
        buf["states"].append(np.asarray(obs_t, dtype=np.uint8))
        buf["actions"].append(int(action))
        buf["rewards"].append(float(rew))
        buf["next_states"].append(np.asarray(obs_tp1, dtype=np.uint8))
        buf["terminated"].append(bool(terminated))
        buf["life_lost"].append(bool(lost))
        ep["score"] += rew
        ep["steps"] += 1
        if args.state_every and ep["steps"] % args.state_every == 0 and not (terminated or truncated):
            emu_states.append((len(buf["actions"]) - 1, len(scores), ep["steps"], ep["score"], lives,
                               pickle.dumps(env.unwrapped.ale.cloneState())))
        if terminated or truncated:
            scores.append(ep["score"])
            print(f"Partida {len(scores)}: {ep['score']:.0f} puntos  "
                  f"(media {np.mean(scores):.0f}, {len(buf['actions'])} pasos grabados)")
            ep["score"], ep["lives"], ep["steps"] = 0.0, None, 0

    print("Controles: ← → moverse, ESPACIO disparar, ESC terminar. Haz clic en la ventana del juego.")
    try:
        play(env, fps=args.fps, zoom=args.zoom, callback=callback,
             keys_to_action=keys_to_action, noop=0)
    finally:
        env.close()
        # descartar la partida a medias del final (la que empieza sola tras terminar la anterior)
        ends = [i for i, t in enumerate(buf["terminated"]) if t]
        keep = ends[-1] + 1 if ends else 0       # sin ninguna partida completa -> no se guarda nada
        if not args.keep_partial and not args.infinite_lives and keep < len(buf["actions"]):
            dropped = len(buf["actions"]) - keep
            for k in buf:
                del buf[k][keep:]
            emu_states[:] = [e for e in emu_states if e[0] < keep]
            print(f"(descartados {dropped} pasos de la última partida, que quedó a medias)")
        if args.infinite_lives and buf["actions"]:
            print(f"Vidas infinitas: {ep['score']:.0f} puntos en la sesión con {ep['deaths']} muertes"
                  " (no comparable con el agente)")
        n = len(buf["actions"])
        if n == 0:
            print("No se ha grabado nada.")
            return
        os.makedirs("demos", exist_ok=True)
        out = os.path.join("demos", f"demo_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        np.savez_compressed(
            out,
            states=np.stack(buf["states"]), actions=np.array(buf["actions"], dtype=np.int64),
            rewards=np.array(buf["rewards"], dtype=np.float32),
            next_states=np.stack(buf["next_states"]),
            terminated=np.array(buf["terminated"]), life_lost=np.array(buf["life_lost"]),
            scores=np.array(scores, dtype=np.float32),
            emu_states=np.array(emu_states, dtype=object),
            meta=np.array([args.env, args.frame_stack, args.sticky, args.fps, args.infinite_lives], dtype=object),
        )
        acts = np.bincount(buf["actions"], minlength=len(meanings))
        print(f"\nGuardado {out}: {n} pasos, {len(scores)} partidas completas"
              + (f", media {np.mean(scores):.0f}" if scores else ""))
        if emu_states:
            print(f"Estados del emulador guardados: {len(emu_states)} (para empezar partidas de entrenamiento desde ellos)")
        print("Acciones:", ", ".join(f"{m}={c}" for m, c in zip(meanings, acts)))
        if not buf["terminated"][-1]:
            print("(sin ninguna partida completa: se guarda lo jugado igualmente)")


if __name__ == "__main__":
    main()
