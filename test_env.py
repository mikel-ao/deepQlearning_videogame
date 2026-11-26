import gymnasium as gym
import ale_py

gym.register_envs(ale_py)  # obligatorio con Gymnasium >= 1.0

env = gym.make("ALE/DemonAttack-v5", obs_type="ram", render_mode="human")
obs, info = env.reset(seed=42)
print("Obs shape:", obs.shape, "| dtype:", obs.dtype)   # (128,) uint8
print("Acciones:", env.action_space)                     # Discrete(6)
print(env.unwrapped.get_action_meanings())

total, done = 0.0, False
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    total += r
    done = term or trunc
print("Recompensa agente aleatorio:", total)
env.close()
