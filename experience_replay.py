"""
Replay memory: guarda transiciones (state, action, new_state, reward, terminated)
y devuelve muestras aleatorias para romper la correlación entre pasos consecutivos.
"""
import random
from collections import deque


class ReplayMemory:
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)   # al llenarse, descarta las más antiguas
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    import numpy as np

    mem = ReplayMemory(maxlen=5, seed=0)
    for i in range(8):
        s = np.full(128, i, dtype=np.uint8)
        mem.append((s, i % 6, s + 1, float(i), False))
    print("Tamaño:", len(mem))                                # 5 (las 3 primeras se descartan)
    print("Acciones en una muestra:", [t[1] for t in mem.sample(3)])
