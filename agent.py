"""
DQN sobre Demon Attack (RAM, 128 bytes) siguiendo la guía de Flappy Bird.

Entrenar:            python agent.py demonattack_long --train
Ver jugar (prueba):  python agent.py demonattack_long
"""
import argparse
import collections
import glob
import itertools
import os
import pickle
import random
import time
from datetime import datetime

import ale_py
import gymnasium as gym
import matplotlib
import numpy as np
import torch
import yaml
from torch import nn

from dqn import build_dqn
from experience_replay import ReplayMemory

matplotlib.use("Agg")          # dibujar a archivo, sin ventana
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
PLOT_EVERY_SECONDS = 60
CHECKPOINT_EVERY_EPISODES = 50
SNAPSHOT_EVERY_EPISODES = 50       # un modelo guardado cada 50 episodios para el vídeo
AVG_WINDOW = 100


def make_env(env_id, frame_stack=1, render_mode=None):
    """Entorno RAM. Con frame_stack > 1 apila los últimos N estados y los aplana (N*128 bytes)."""
    env = gym.make(env_id, obs_type="ram", render_mode=render_mode)
    if frame_stack > 1:
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)   # (N, 128)
        env = gym.wrappers.FlattenObservation(env)                              # (N*128,)
    return env


# acciones con disparo equivalentes (para usar demostraciones con action_set de 3 acciones)
FIRE_VERSION = {0: 1, 2: 4, 3: 5}          # NOOP->FIRE, RIGHT->RIGHTFIRE, LEFT->LEFTFIRE
PLAYER_X_RAM, WALL_VALUES = 22, (17, 200)  # byte 22 = 17 pegado a la izquierda, 200 pegado a la derecha


def restore_emulator(env, state_bytes):
    """Pone el emulador en un estado grabado (record_human.py) y rehace la pila de estados RAM."""
    ale = env.unwrapped.ale
    ale.restoreState(pickle.loads(state_bytes))
    ram = ale.getRAM().copy()
    w = env
    while w is not None and not isinstance(w, gym.wrappers.FrameStackObservation):
        w = getattr(w, "env", None)
    if w is None:
        return ram, ale.lives()
    for _ in range(w.stack_size):
        w.obs_queue.append(ram)
    return np.concatenate([ram] * w.stack_size), ale.lives()


class Agent:
    # ------------------------------------------------------------------ 1.12
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as file:
            all_sets = yaml.safe_load(file)
            hp = all_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.env_id = hp["env_id"]
        self.replay_memory_size = hp["replay_memory_size"]
        self.mini_batch_size = hp["mini_batch_size"]
        self.train_start = hp["train_start"]
        self.epsilon_init = hp["epsilon_init"]
        self.epsilon_decay = hp["epsilon_decay"]
        self.epsilon_min = hp["epsilon_min"]
        self.network_sync_rate = hp["network_sync_rate"]
        self.train_freq = hp["train_freq"]
        self.learning_rate = hp["learning_rate"]
        self.discount_factor = hp["discount_factor"]
        self.hidden_dim = hp["hidden_dim"]
        self.clip_reward = hp["clip_reward"]
        self.max_episodes = hp["max_episodes"]
        self.device = hp["device"]
        self.seed = hp.get("seed")                     # semilla (None = aleatoria)
        self.snapshot_every = hp.get("snapshot_every", SNAPSHOT_EVERY_EPISODES)
        if hp.get("torch_threads"):
            torch.set_num_threads(hp["torch_threads"])  # 1 = varios entrenamientos en paralelo sin pisarse
        self.resume_from = hp.get("resume_from")       # bloque cuyo checkpoint se continúa
        self.double_dqn = hp.get("double_dqn", False)  # Double DQN en el cálculo del objetivo
        self.frame_stack = hp.get("frame_stack", 1)     # nº de estados RAM apilados (1 = sin apilar)
        self.eval_every = hp.get("eval_every")          # evaluar cada N episodios (None = no)
        self.eval_episodes = hp.get("eval_episodes", 10)
        self.life_loss = hp.get("life_loss", "none")    # none | terminal | terminal_penalty
        self.loss_name = hp.get("loss", "mse")          # mse | huber
        assert self.life_loss in ("none", "terminal", "terminal_penalty"), self.life_loss

        # ---- tanda 3 (todas opcionales; sin ellas el agente es idéntico al de antes) ----
        self.action_set = hp.get("action_set")          # p. ej. [1, 4, 5] = FIRE, RIGHTFIRE, LEFTFIRE
        self.demo_glob = hp.get("demo_glob")            # p. ej. "demos/demo_*.npz"
        self.demo_ratio = hp.get("demo_ratio", 0.25)    # fracción de cada mini-batch sacada de tus partidas
        self.demo_margin = hp.get("demo_margin", 0.0)   # peso de la pérdida de margen de DQfD (0 = no)
        self.demo_start_prob = hp.get("demo_start_prob", 0.0)    # prob. de empezar un episodio en un estado tuyo
        self.demo_start_min_score = hp.get("demo_start_min_score", 0)  # solo estados con >= puntos (oleadas avanzadas)
        self.init_model = hp.get("init_model")          # .pt con el que inicializar la red
        self.corner_penalty = hp.get("corner_penalty", 0.0)   # castigo por paso pegado a una pared...
        self.corner_patience = hp.get("corner_patience", 30)  # ...a partir de N pasos seguidos (30 = 2 s)

        # ---- tanda 4: estabilidad (todas opcionales; sin ellas, el agente de siempre) ----
        self.grad_clip = hp.get("grad_clip")            # norma máxima del gradiente (p. ej. 10)
        self.target_tau = hp.get("target_tau")          # red objetivo suave: θ' <- τθ + (1-τ)θ' en cada actualización
        self.n_step = hp.get("n_step", 1)               # retornos de n pasos
        self.dueling = hp.get("dueling", False)         # Dueling DQN
        self.munchausen = hp.get("munchausen", False)   # Munchausen-DQN (objetivo suave regularizado por entropía)
        self.m_alpha = hp.get("m_alpha", 0.9)
        self.m_tau = hp.get("m_tau", 0.03)
        self.m_l0 = hp.get("m_l0", -1.0)
        self.lr_end = hp.get("lr_end")                  # lr decreciente (exponencial) hasta este valor al final
        self.clip_stats = [0, 0]                        # actualizaciones / cuántas recortó grad_clip (por episodio)

        # 2.8: Huber (SmoothL1) es menos sensible a errores grandes -> entrenamiento más estable
        self.loss_fn = nn.SmoothL1Loss() if self.loss_name == "huber" else nn.MSELoss()
        self.optimizer = None                           # se crea en run()

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.pt")
        self.CKPT_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}_ckpt.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.png")
        self.SNAP_DIR = os.path.join(RUNS_DIR, f"{hyperparameter_set}_snapshots")
        os.makedirs(self.SNAP_DIR, exist_ok=True)

    def log(self, msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}: {msg}"
        print(line)
        with open(self.LOG_FILE, "a") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------ run
    def run(self, is_training=True, render=False, model_file=None):
        env = make_env(self.env_id, self.frame_stack, "human" if render else None)
        if self.seed is not None:                       # reproducible: misma semilla, mismo entrenamiento
            random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
            env.reset(seed=self.seed); env.action_space.seed(self.seed)
        num_states = env.observation_space.shape[0]
        self.actions = list(self.action_set) if self.action_set else list(range(env.action_space.n))
        num_actions = len(self.actions)                  # la red da un valor por acción PERMITIDA

        policy_dqn = build_dqn(num_states, num_actions, self.hidden_dim, self.dueling).to(self.device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            if self.init_model:
                policy_dqn.load_state_dict(torch.load(self.init_model, map_location=self.device))
            target_dqn = build_dqn(num_states, num_actions, self.hidden_dim, self.dueling).to(self.device)
            target_dqn.load_state_dict(policy_dqn.state_dict())          # 2.6
            self.demos = self.load_demos(num_states) if self.demo_glob else None
            self.start_states = self.load_start_states() if self.demo_start_prob > 0 else []
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

            rewards_per_episode, epsilon_history = [], []
            eval_history = []                                             # (episodio, media, min, max)
            eval_env = make_env(self.env_id, self.frame_stack) if self.eval_every else None
            best_avg = -np.inf
            step_count = 0
            last_plot = time.time()
            start_episode = 0

            if self.resume_from:                                          # continuar entrenamiento
                ckpt_path = os.path.join(RUNS_DIR, f"{self.resume_from}_ckpt.pt")
                ckpt = torch.load(ckpt_path, map_location=self.device)
                policy_dqn.load_state_dict(ckpt["policy"])
                target_dqn.load_state_dict(ckpt["target"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                step_count = ckpt["step_count"]
                start_episode = ckpt["episode"] + 1
                if self.epsilon_init is None:                             # null en el YAML = seguir con el ε guardado
                    epsilon = ckpt["epsilon"]
                self.log(f"Reanudando desde {ckpt_path}: episodio {start_episode},"
                         f" {step_count} pasos, ε={epsilon:.3f} (la replay memory empieza vacía)")

            self.log(f"Inicio entrenamiento '{self.hyperparameter_set}'")
            extras = {k: v for k, v in dict(grad_clip=self.grad_clip, target_tau=self.target_tau,
                                           n_step=self.n_step if self.n_step > 1 else None,
                                           dueling=self.dueling or None, munchausen=self.munchausen or None,
                                           lr_end=self.lr_end).items() if v}
            if extras:
                self.log(f"Opciones de estabilidad: {extras}")
        else:
            start_episode = 0
            policy_dqn.load_state_dict(torch.load(model_file or self.MODEL_FILE, map_location=self.device))
            policy_dqn.eval()
            epsilon = 0.0

        for episode in itertools.count(start_episode):                  # 1.10
            state, info = env.reset()
            lives = info.get("lives")
            demo_start = False
            if is_training and self.start_states and random.random() < self.demo_start_prob:
                state, lives = restore_emulator(env, random.choice(self.start_states))
                demo_start = True
            terminated = truncated = False
            episode_reward, ep_steps, t0 = 0.0, 0, time.time()
            nbuf = collections.deque()                     # ventana para los retornos de n pasos
            if is_training and self.lr_end:
                frac = min(1.0, (episode - start_episode) / max(self.max_episodes, 1))
                lr = self.learning_rate * (self.lr_end / self.learning_rate) ** frac
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
            wall_run, wall_steps = 0, 0                    # pasos seguidos / totales pegado a una pared

            while not (terminated or truncated):
                # ---- 2.1 ε-greedy (sin forward si explora) -------------------
                if is_training and random.random() < epsilon:
                    action = (random.randrange(num_actions) if self.action_set   # índice dentro de self.actions
                              else env.action_space.sample())
                else:
                    with torch.no_grad():                                # 2.3
                        s = torch.as_tensor(state, device=self.device).unsqueeze(0)
                        action = policy_dqn(s).argmax(dim=1).item()

                new_state, reward, terminated, truncated, info = env.step(self.actions[action])
                episode_reward += reward
                ep_steps += 1
                at_wall = env.unwrapped.ale.getRAM()[PLAYER_X_RAM] in WALL_VALUES
                wall_run = wall_run + 1 if at_wall else 0
                wall_steps += at_wall

                if is_training:
                    r = float(np.sign(reward)) if self.clip_reward else float(reward)

                    # ---- perder una vida: terminal SOLO para el aprendizaje ----------
                    # la partida sigue; el objetivo deja de contar con el futuro tras morir,
                    # así morir cuesta exactamente el valor futuro que se pierde
                    if self.corner_penalty and wall_run > self.corner_patience:
                        r -= self.corner_penalty                 # escondido en una esquina: castigo pequeño por paso

                    lost_life = lives is not None and info.get("lives", lives) < lives
                    lives = info.get("lives", lives)
                    done_for_learning = terminated
                    if lost_life and self.life_loss != "none":
                        done_for_learning = True
                        if self.life_loss == "terminal_penalty":
                            r -= 1.0

                    # guardar `terminated` (o pérdida de vida), no `truncated`: truncar no es morir
                    nbuf.append((state, action, r, done_for_learning, new_state))
                    if done_for_learning or truncated:
                        while nbuf:                                # vaciar la ventana: no se cruza una muerte
                            memory.append(self._n_step_transition(nbuf))
                            nbuf.popleft()
                    elif len(nbuf) >= self.n_step:
                        memory.append(self._n_step_transition(nbuf))
                        nbuf.popleft()
                    step_count += 1

                    # ---- 2.7 optimizar cada train_freq pasos ------------------
                    if len(memory) >= self.train_start and step_count % self.train_freq == 0:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)
                        if self.target_tau:                      # red objetivo suave (Polyak)
                            with torch.no_grad():
                                for pt, p in zip(target_dqn.parameters(), policy_dqn.parameters()):
                                    pt.mul_(1 - self.target_tau).add_(p, alpha=self.target_tau)

                    # ---- 2.6 sincronizar red objetivo (por pasos) -------------
                    if not self.target_tau and step_count % self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

                state = new_state

            if not is_training:
                print(f"Episodio {episode}: recompensa={episode_reward:.0f}")
                continue

            # ---- fin de episodio: ε, registro, guardado -----------------------
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)    # 2.2
            if demo_start:          # empezó a mitad de partida: no cuenta para media100 (no sería comparable)
                self.log(f"ep {episode:5d} | r={episode_reward:6.0f} | [desde estado tuyo] | ε={epsilon:.3f}"
                         f" | pasos={step_count}")
            else:
                rewards_per_episode.append(episode_reward)
                epsilon_history.append(epsilon)

            avg = np.mean(rewards_per_episode[-AVG_WINDOW:]) if rewards_per_episode else 0.0
            sps = ep_steps / max(time.time() - t0, 1e-9)
            if not demo_start:
                self.log(f"ep {episode:5d} | r={episode_reward:6.0f} | media{AVG_WINDOW}={avg:7.1f}"
                         f" | ε={epsilon:.3f} | pasos={step_count} | {sps:,.0f} pasos/s"
                         f" | pared {wall_steps / max(ep_steps, 1):.0%}"
                         + (f" | recorte {self.clip_stats[1] / max(self.clip_stats[0], 1):.0%}" if self.grad_clip else "")
                         + (f" | lr {self.optimizer.param_groups[0]['lr']:.1e}" if self.lr_end else ""))
                self.clip_stats = [0, 0]

            if self.eval_every:
                # evaluación separada: red congelada, ε=0, sin entrenar -> métrica estable
                if (episode + 1) % self.eval_every == 0:
                    scores = self.evaluate(eval_env, policy_dqn)
                    m = float(np.mean(scores))
                    eval_history.append((episode + 1, m, min(scores), max(scores)))
                    self.log(f"  EVAL ep {episode + 1}: media{self.eval_episodes}={m:.1f}"
                             f" | min={min(scores):.0f} | max={max(scores):.0f}")
                    if m > best_avg:                                      # mejor modelo según la EVALUACIÓN
                        best_avg = m
                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        self.log(f"  ↳ nuevo mejor modelo (eval): {m:.1f}")
            # sin evaluación: mejor modelo según la media móvil (no un episodio suelto)
            elif len(rewards_per_episode) >= min(AVG_WINDOW, 20) and avg > best_avg:
                best_avg = avg
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                self.log(f"  ↳ nuevo mejor modelo: media{AVG_WINDOW}={avg:.1f}")

            # snapshot con nombre propio (no se sobrescribe) -> vídeo de progreso
            if (episode + 1) % self.snapshot_every == 0:
                torch.save(policy_dqn.state_dict(),
                           os.path.join(self.SNAP_DIR, f"ep_{episode + 1:05d}.pt"))

            # checkpoint periódico para poder reanudar
            if (episode + 1) % CHECKPOINT_EVERY_EPISODES == 0:
                torch.save({"episode": episode, "step_count": step_count, "epsilon": epsilon,
                            "policy": policy_dqn.state_dict(), "target": target_dqn.state_dict(),
                            "optimizer": self.optimizer.state_dict()}, self.CKPT_FILE)

            if time.time() - last_plot > PLOT_EVERY_SECONDS:
                self.save_graph(rewards_per_episode, epsilon_history, eval_history)
                last_plot = time.time()

            if episode + 1 - start_episode >= self.max_episodes:          # max_episodes = episodios de ESTA ejecución
                self.save_graph(rewards_per_episode, epsilon_history, eval_history)
                self.log("Fin: alcanzado max_episodes")
                break

        env.close()

    def evaluate(self, env, policy_dqn):
        """Juega eval_episodes partidas completas con ε=0 y devuelve las puntuaciones."""
        scores = []
        for _ in range(self.eval_episodes):
            state, _ = env.reset()
            done, score = False, 0.0
            while not done:
                with torch.no_grad():
                    s = torch.as_tensor(state, device=self.device).unsqueeze(0)
                    action = policy_dqn(s).argmax(dim=1).item()
                state, reward, terminated, truncated, _ = env.step(self.actions[action])
                score += reward
                done = terminated or truncated
            scores.append(score)
        return scores

    # ------------------------------------------------------------------ 3.x
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations, discounts = zip(*mini_batch)   # 3.4
        states, new_states = np.stack(states), np.stack(new_states)
        actions, rewards, terminations = np.array(actions), np.array(rewards), np.array(terminations)
        discounts = np.array(discounts, dtype=np.float32)        # γ^n de cada transición (γ si n = 1)

        n_demo = 0
        if self.demos is not None:
            # una parte del mini-batch sale de tus partidas (se quedan para siempre, no se sobrescriben)
            n_demo = max(1, int(round(self.mini_batch_size * self.demo_ratio)))
            keep = self.mini_batch_size - n_demo
            d = self.demos
            idx = np.random.randint(len(d["actions"]), size=n_demo)
            states = np.concatenate([states[:keep], d["states"][idx]])
            new_states = np.concatenate([new_states[:keep], d["next_states"][idx]])
            actions = np.concatenate([actions[:keep], d["actions"][idx]])
            rewards = np.concatenate([rewards[:keep], d["rewards"][idx]])
            terminations = np.concatenate([terminations[:keep], d["dones"][idx]])
            discounts = np.concatenate([discounts[:keep], np.full(n_demo, self.discount_factor, dtype=np.float32)])

        states = torch.as_tensor(states, device=self.device)
        new_states = torch.as_tensor(new_states, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor(terminations, dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(discounts, device=self.device)

        with torch.no_grad():                                                     # 3.5
            if self.munchausen:
                # Munchausen-DQN (Vieillard et al., 2020): objetivo "suave" con la política softmax de la red
                # objetivo, más un bonus α·τ·log π(a|s) acotado en [l0, 0]
                q_next = target_dqn(new_states)
                logpi_next = torch.log_softmax(q_next / self.m_tau, dim=1)
                next_v = (logpi_next.exp() * (q_next - self.m_tau * logpi_next)).sum(dim=1)
                logpi = torch.log_softmax(target_dqn(states) / self.m_tau, dim=1).gather(1, actions.unsqueeze(1)).squeeze(1)
                bonus = self.m_alpha * torch.clamp(self.m_tau * logpi, min=self.m_l0, max=0.0)
                target_q = rewards + bonus + (1 - terminations) * discounts * next_v
            elif self.double_dqn:
                # la red de política ELIGE la mejor acción, la red objetivo la EVALÚA
                best_actions = policy_dqn(new_states).argmax(dim=1, keepdim=True)
                next_q = target_dqn(new_states).gather(1, best_actions).squeeze(1)
            else:
                next_q = target_dqn(new_states).max(dim=1)[0]
            if not self.munchausen:
                target_q = rewards + (1 - terminations) * discounts * next_q

        all_q = policy_dqn(states)
        current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # 3.6
        assert current_q.shape == target_q.shape                                   # trampa (32,1) vs (32,)

        loss = self.loss_fn(current_q, target_q)
        if n_demo and self.demo_margin > 0:
            # pérdida de margen (DQfD): tu acción debe valer al menos 0.8 más que cualquier otra
            q_d, a_d = all_q[-n_demo:], actions[-n_demo:]
            margins = torch.full_like(q_d, 0.8)
            margins.scatter_(1, a_d.unsqueeze(1), 0.0)
            j_e = ((q_d + margins).max(dim=1)[0] - q_d.gather(1, a_d.unsqueeze(1)).squeeze(1)).mean()
            loss = loss + self.demo_margin * j_e
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            norm = torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), self.grad_clip)
            self.clip_stats[0] += 1
            self.clip_stats[1] += float(norm) > self.grad_clip
        self.optimizer.step()

    def _n_step_transition(self, nbuf):
        """Transición de n pasos desde nbuf[0]: suma descontada de recompensas hasta n pasos o una muerte."""
        state, action = nbuf[0][0], nbuf[0][1]
        ret, disc = 0.0, 1.0
        for i, (_, _, r, done, s_next) in enumerate(nbuf):
            if i >= self.n_step:
                break
            ret += disc * r
            disc *= self.discount_factor
            if done:
                return (state, action, s_next, ret, True, disc)
        return (state, action, s_next, ret, False, disc)

    # ------------------------------------------------------------------ demostraciones
    def _action_index(self, a):
        """Acción del juego (0-5) -> índice dentro de self.actions, o None si no está permitida."""
        if a in self.actions:
            return self.actions.index(a)
        if FIRE_VERSION.get(a) in self.actions:          # sin disparo -> su versión con disparo
            return self.actions.index(FIRE_VERSION[a])
        return None

    def load_demos(self, num_states):
        S, S2, A, R, D = [], [], [], [], []
        files = sorted(glob.glob(self.demo_glob))
        for f in files:
            d = np.load(f, allow_pickle=True)
            if d["states"].shape[1] != num_states:
                self.log(f"  (demo {f} descartada: frame_stack distinto)")
                continue
            idx = np.array([self._action_index(int(a)) for a in d["actions"]], dtype=object)
            ok = np.array([i is not None for i in idx])
            r = d["rewards"].astype(np.float32)
            r = np.sign(r) if self.clip_reward else r
            done = d["terminated"].copy()
            if self.life_loss != "none":                      # misma regla que el agente
                done = done | d["life_lost"]
                if self.life_loss == "terminal_penalty":
                    r = r - d["life_lost"].astype(np.float32)
            S.append(d["states"][ok]); S2.append(d["next_states"][ok]); A.append(idx[ok].astype(np.int64))
            R.append(r[ok]); D.append(done[ok].astype(np.float32))
        if not S:
            raise SystemExit(f"No hay demostraciones válidas en {self.demo_glob}")
        demos = {"states": np.concatenate(S), "next_states": np.concatenate(S2), "actions": np.concatenate(A),
                 "rewards": np.concatenate(R), "dones": np.concatenate(D)}
        self.log(f"Demostraciones: {len(demos['actions'])} transiciones de {len(files)} archivos,"
                 f" {self.demo_ratio:.0%} de cada mini-batch, margen={self.demo_margin}")
        return demos

    def load_start_states(self):
        states = []
        for f in sorted(glob.glob(self.demo_glob or "demos/demo_*.npz")):
            d = np.load(f, allow_pickle=True)
            if "emu_states" not in d.files:
                continue
            states += [e[5] for e in d["emu_states"] if e[3] >= self.demo_start_min_score]
        self.log(f"Estados de inicio: {len(states)} (puntos >= {self.demo_start_min_score}),"
                 f" prob={self.demo_start_prob}")
        return states

    def save_graph(self, rewards, epsilons, evals=()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        moving = [np.mean(rewards[max(0, i - AVG_WINDOW + 1):i + 1]) for i in range(len(rewards))]
        ax1.plot(rewards, alpha=.3, label="episodio")
        ax1.plot(moving, label=f"media {AVG_WINDOW}")
        if evals:
            ep, m, lo, hi = zip(*evals)
            ax1.fill_between(ep, lo, hi, color="C3", alpha=.12)
            ax1.plot(ep, m, "o-", color="C3", ms=3, label=f"eval (ε=0, media {self.eval_episodes})")
        ax1.set_xlabel("episodio"); ax1.set_ylabel("recompensa"); ax1.legend()
        ax2.plot(epsilons)
        ax2.set_xlabel("episodio"); ax2.set_ylabel("ε")
        fig.tight_layout(); fig.savefig(self.GRAPH_FILE); plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar o probar el DQN")
    parser.add_argument("hyperparameters", help="bloque de hyperparameters.yml")
    parser.add_argument("--train", action="store_true", help="entrenar (sin esto: modo prueba con ventana)")
    parser.add_argument("--model", default=None, help="modo prueba: .pt concreto (p. ej. un snapshot)")
    args = parser.parse_args()

    agent = Agent(args.hyperparameters)
    if args.train:
        agent.run(is_training=True, render=False)
    else:
        agent.run(is_training=False, render=True, model_file=args.model)
